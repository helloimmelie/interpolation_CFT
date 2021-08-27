#import torch
import torch
from torch.nn import functional as F
#import module
import os
import cv2
import numpy as np
from tqdm import tqdm
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
import sys
from pathlib import Path
#import custom module
from .model.pytorch_msssim import ssim_matlab
from utils import transferAudio
import time



warnings.filterwarnings("ignore")

def clear_write_buffer(user_args, write_buffer, vid_out):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item[:, :, ::-1])

def build_read_buffer(user_args, w, read_buffer, videogen):
    try:
        
        if user_args.montage:
            left = w // 4
            w = w // 2

        for frame in videogen:
             if not user_args.img is None:
                  frame = cv2.imread(os.path.join(user_args.img, frame))[:, :, ::-1].copy()
             if user_args.montage:
                  frame = frame[:, left: left + w]
             read_buffer.put(frame)

    except:
        pass
    read_buffer.put(None)

def make_inference(I0, I1, n, model, args  ):
    middle = model.inference(I0, I1, args.scale)
    if n == 1:
        return [middle]
    first_half = make_inference(I0, middle, n=n//2)
    second_half = make_inference(middle, I1, n=n//2)
    if n%2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]

def pad_image(img, padding, args):
    if(args.fp16):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)


def RIFE_inference(args):

    assert (not args.video is None or not args.img is None)
    if args.UHD and args.scale==1.0:
        args.scale = 0.5
    assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
    if not args.img is None:
        args.png = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if(args.fp16):
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    try:
        try:
            from .model.RIFE_HDv2 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v2.x HD model.")
        except:
            from train_log.RIFE_HDv3 import Model
            model = Model()
            model.load_model(args.modelDir, -1)
            print("Loaded v3.x HD model.")
    except:
        from .model.RIFE_HD import Model
        model = Model()
        model.load_model(args.modelDir, -1)
        print("Loaded v1.x HD model")
    model.eval()
    model.device()

    if not args.video is None:

        videoCapture = cv2.VideoCapture(args.video)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        videoCapture.release()
        if args.fps is None:
            fpsNotAssigned = True
            args.fps = fps * (2 ** args.exp)
        else:
            fpsNotAssigned = False
        videogen = skvideo.io.vreader(args.video)
        lastframe = next(videogen)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_path_wo_ext, ext = os.path.splitext(args.video)
        print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
        if args.png == False and fpsNotAssigned == True and not args.skip:
            print("The audio will be merged after interpolation process")
        else:
            print("Will not merge audio because using png, fps or skip flag!")
    else:
        videogen = []
        for f in os.listdir(args.img):
            if 'png' in f:
                videogen.append(f)
        tot_frame = len(videogen)
        videogen.sort(key= lambda x:int(x[:-4]))
        lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
        videogen = videogen[1:]
    h, w, _ = lastframe.shape
    vid_out_name = None
    vid_out = None
    if args.png:
        if not os.path.exists('vid_out'):
            os.mkdir('vid_out')
    else:
        if args.output is not None:
            if not os.path.isdir(os.path.join('.','RIFE_result')):
                os.mkdir(os.path.join('.','RIFE_result'))
            file_name = Path(args.output).name 
            vid_out_name = os.path.join('.','RIFE_result',file_name)
            vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))
        else:
            vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, (2 ** args.exp), int(np.round(args.fps)), args.ext)
            vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))
    


    tmp = max(32, int(32 / args.scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    pbar = tqdm(total=tot_frame)
    skip_frame = 1

    if args.montage:
        left = w // 4
        w = w // 2
        lastframe = lastframe[:, left: left + w]

    write_buffer = Queue(maxsize=500)
    read_buffer = Queue(maxsize=500)
    
    _thread.start_new_thread(build_read_buffer, (args, w, read_buffer, videogen))
    _thread.start_new_thread(clear_write_buffer, (args, write_buffer, vid_out))

    I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1, padding, args)

    while True:
        frame = read_buffer.get()
        if frame is None:
            break
        I0 = I1
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1, padding, args)
        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

        if ssim > 0.995:
            if skip_frame % 100 == 0:
                print("\nWarning: Your video has {} static frames, skipping them may change the duration of the generated video.".format(skip_frame))
            skip_frame += 1
            if args.skip:
                pbar.update(1)
                continue

        if ssim < 0.2:
            output = []
            for i in range((2 ** args.exp) - 1):
                output.append(I0)
            '''
            output = []
            step = 1 / (2 ** args.exp)
            alpha = 0
            for i in range((2 ** args.exp) - 1):
            alpha += step
            beta = 1-alpha
            output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
            '''
        else:
            output = make_inference(I0, I1, 2**args.exp-1, model, args) if args.exp else []

        if args.montage:
            write_buffer.put(np.concatenate((lastframe, lastframe), 1))
            for mid in output:
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
        else:
            write_buffer.put(lastframe)
            for mid in output:
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                write_buffer.put(mid[:h, :w])
        pbar.update(1)
        lastframe = frame

    if args.montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
    else:
        write_buffer.put(lastframe)

    while(not write_buffer.empty()):
        time.sleep(0.1)
    pbar.close()
    if not vid_out is None:
        vid_out.release()

    # move audio to new video file if appropriate
    if args.png == False and fpsNotAssigned == True and not args.skip and not args.video is None:
        try:
            transferAudio(args.video, vid_out_name)
        except:
            print("Audio transfer failed. Interpolated video will have no audio")
            targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
            os.rename(targetNoAudio, vid_out_name)


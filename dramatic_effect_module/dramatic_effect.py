#import module
import os
import cv2 as cv
from tqdm import tqdm
from pathlib import Path
#import torch
from torch.utils.data import Dataset, DataLoader
import torch
#Import Custom module
from Dataset.VideoDataset import VideoDataset
from utils import transferAudio



def read_labels(label_path):

    f = open(label_path,'r',encoding='utf-8')

    label_datas = f.readlines()
    data_list = []
    labels =[]

    for data in label_datas:
        data = data.replace('\n','')
        data = data.replace('(','')
        data = data.replace(')','')
        data_list.append(data)

    for frame in range(1, len(labels)): #data[0]은 파일이름이기 때문에 제외 
        frame_s = data_list[frame].split(',') 
        labels.append(frame_s) 
    
    return labels


def dramatic_effect(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RIFE_file_name = Path(args.output).name

    RIFE_video = os.path.join('.','RIFE_result', RIFE_file_name )

    file_name = Path(args.video).name
    
    label_path = args.label_path

    label_datas = read_labels(args.label_path)
    
    #save file 
    save_filename = args.output #file_name => 저장할 파일 이름

        

    #get original video info from "videocapture" function
    cap_original = cv.VideoCapture(args.video)
    fps_original = cap_original.get(cv.CAP_PROP_FPS)

    #get LIFE output video info from "videocapture" function
    cap_RIFE = cv.VideoCapture(os.path.join('.','RIFE_result',RIFE_file_name)) #동영상 파일 읽는 함수:cv2.VideoCapture
    width_RIFE = cap_RIFE.get(cv.CAP_PROP_FRAME_WIDTH) #cv2.CAP_PROP_FAME_WIDTH 
    height_RIFE = cap_RIFE.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps_RIFE = cap_RIFE.get(cv.CAP_PROP_FPS)

    fps_division = fps_RIFE//int(fps_original)

    #save codec
    fourcc = cv.VideoWriter_fourcc(*'mp4v') 
    out_put = cv.VideoWriter(save_filename, fourcc, fps_RIFE, (int(width_RIFE), int(height_RIFE)))

    #data load
    file_path = os.path.join('.','RIFE_result',RIFE_file_name)
    predict_dataset = VideoDataset(file_path=file_path)
    predict_dataloader = DataLoader(predict_dataset, batch_size=args.batch_size, shuffle = False, num_workers = 0)
    
    #frame_range: counts index of frame 
    frame_range = 0

    for frame in tqdm(predict_dataloader):
        
        frame_list = []
        label_list = []
        frame_range = frame_range + args.batch_size

        frame = frame.to(device)

        for label_idx in range(len(label_datas)):
            start = int(label_datas[label_idx][0])
            finish = int(label_datas[label_idx][1])
            if start > frame_range and finish < frame_range:
                label_list.append({'start': start, 'finish': finish})

        for frame_element in frame:
            frame_list.append(frame_element) 
               
        
        for label_idx in range(len(label_list)) : 
            start = int(label_list[label_idx]['start'])
            finish = int(label_list[label_idx]['finish'])

            #divide by fps difference between original and RIFE result (because RIFE result extends FPS)
            start = (start//fps_division) - frame_range
            finish = (finish//fps_division) - frame_range

            start_point = start * fps_division - frame_range

            for i in range(finish-start):
                frame_list[start_point+(i*fps_division)] = frame_list[start_point+(i*fps_division)] 
                frame_list[start_point+1+(i*fps_division)] = frame_list[start_point+(i*fps_division)]
                frame_list[start_point+2+(i*fps_division)] = frame_list[start_point+(i*fps_division)]
                frame_list[start_point+3+(i*fps_division)] = frame_list[start_point+(i*fps_division)]
            
        else:
            pass

        for frame_element in frame_list:
            out_put.write(frame_element.cpu().numpy())
        
    out_put.release()

    try:
        transferAudio(file_path, save_filename)
    
    except:
        print("Audio transfer failed. Result video will have no audio")
        targetNoAudio = os.path.splitext(save_filename)[0] + "_noaudio" + os.path.splitext(save_filename)[1]
        os.rename(targetNoAudio, save_filename)
    


#import module
import argparse
import cmd

#import custommodule
from RIFE_module.inference_video import RIFE_inference
from dramatic_effect_module.dramatic_effect import dramatic_effect

class VideoShell(cmd.Cmd):
    
    intro = 'Welcome to the video interpolation program.\ncommand:execute_interpolation --video=(video path) --label=(label path) --output=(video output path)'
    prompt = '(video interpolation)'

    def do_execute_interpolation(self, argv):

        parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
        
        #default input & output (whole input & output) 
        parser.add_argument('--output', dest='output', type=str, default='test_output.mp4')
        parser.add_argument('--video', dest='video', type=str, default='test.mp4')
        parser.add_argument('--label_path', type = str, default=None)
        #RIFE settings 
        parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
        parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
        parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
        parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
        parser.add_argument('--scale', dest='scale', type=float, default=0.5, help='Try scale=0.5 for 4k video')
        parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
        parser.add_argument('--fps', dest='fps', type=int, default=None)
        parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
        parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
        parser.add_argument('--exp', dest='exp', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=1000)
        parser.add_argument('--img', dest='img', type=str, default=None)
        args = parser.parse_args(argv.split())

        print("RIFE inference")
        RIFE_inference(args)
        print("Dramatic effect module")
        dramatic_effect(args)

    def do_quit(self, argv):
        #self.close()
        #bye()
        return True
    
    def do_help():

        print("execute_interpolation --video=(video path) --label=(label path) --output=(video output path)\nfor quit, type'quit'")

        
    

    


if __name__ == "__main__":
   
    VideoShell().cmdloop()

    
    
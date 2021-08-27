#import torch
from torch.utils.data import Dataset
import torch
#import modules
import cv2 as cv
from PIL import Image
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, file_path):
        
        self.file_path = file_path
        self.cap = cv.VideoCapture(self.file_path)
        self.num_frames = self.cap.get(cv.CAP_PROP_FRAME_COUNT)

    def __len__(self):
        
   
        return int(self.num_frames)

    def __getitem__(self, index):
        
       
        frames = self.cap.set(cv.CAP_PROP_POS_FRAMES, index)
        ref, frames = self.cap.read()

        return frames
        
        
        
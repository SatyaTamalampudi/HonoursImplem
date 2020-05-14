#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
#importing opencv for feature extraction
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
from PIL import Image


print(os.getcwd()) 
# Folder which contains all the images from which video is to be generated 
path = "C:\\Users\\1609653\\AnomalyDetection\\DividedDataSet\\Training\\Test020"
os.chdir(path)

num_of_images = len(os.listdir('.')) 
print(num_of_images)


images = [img for img in os.listdir(path) 
if img.endswith(".tif") or
    img.endswith(".jpeg") or
    img.endswith("png")] 
print(images)


# ## Video making function
# This function will generate video with all the image frames provided
 
def generate_video(images,outvid=None, fps=10, size=None,is_color=True, format="XVID"): 
    video_name = 'mygeneratedvideo020.mp4'

    frame = cv2.imread(os.path.join(path, images[0])) 

    # setting the frame width, height with the width, height of first image 
    height, width, layers = frame.shape 
    fourcc = VideoWriter_fourcc(*format)
    video = VideoWriter(video_name, fourcc, float(fps), (238, 158), is_color)

    # Appending the images to the video one by one 
    for image in images: 
        video.write(cv2.imread(os.path.join(path, image))) 

    # Deallocating memories taken for window creation 
    cv2.destroyAllWindows() 
    
    video.release() # releasing the video generated 


# Calling the generate_video function 
generate_video(images) 


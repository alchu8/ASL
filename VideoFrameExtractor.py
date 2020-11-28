# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:42:23 2020

"""

import numpy as np
import cv2
import math
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default=None, help="Path to video file.", required=True)
parser.add_argument('--subsample', help="Everett's videos.", action='store_true')
args = parser.parse_args()

framedata=[]
cap = cv2.VideoCapture(args.file)
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
    sys.exit(1)
fps = cap.get(cv2.CAP_PROP_FPS)
success = 1
while(success):
    success, frame = cap.read()
    if success:
        framedata.append(frame)

# When everything done, release the video capture object

cap.release()

# Closes all the frames

cv2.destroyAllWindows()

# Remove null values from extracted data array
predata2=[]
for i in range(len(framedata)):
    if framedata[i] is None:
        print('Frame is None.') # shouldn't happen
        continue
    predata2.append(framedata[i])
predata=predata2
if args.subsample:
    # specific code to reduce frames in everett's videos
    print('Subsampling video.')
    predata=[]
    j=0
    while j < len(predata2):
        predata.append(predata2[j])
        j=j+3
    fps=10


# convert into greyscale for image processing    
predata = np.array(predata)
blackdata=[]
for i in range(len(predata)):
    greyscale= np.sum(predata[i],axis=2) #added all the RGB values
    blackdata.append(greyscale)
blackdata=np.array(blackdata)
np.shape(blackdata)

#Compute standard deviation of the difference between frames
diff=[]
for i in range(len(blackdata)-1):
    frame1=blackdata[i].astype(float)
    frame2=blackdata[i+1].astype(float)
    dif=frame2-frame1
    standard=np.std(dif) # calculated the standard deviation of the difference between frames to differentiate between static and dynamic frames 
    diff.append(standard)
diff = np.array(diff)

#find the moving average of the array of differences to smooth out data and set threshold value
n=math.ceil(0.667*fps) #smoothing based on fps to cut out the noise
z=math.floor(n/2) # shifting the phase to compensate for the delay
def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

mv= moving_average(diff,n)
mv=np.pad(mv, (z,z), 'constant') # shifting the phase to compensate for the delay
def moving_average2(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

mv2= moving_average2(diff,n*5) # heavily smoothed moving averages
threshold=np.mean(mv2) # the threshold value is the average of the heavily smoothed out moving averages



# loop to extract frames such that num frames are extracted between every supposed movement
count=0
result=[]
num = 4 # number of frames to extract between movements
resultx=[]
result_final=[]
for i in range(len(mv)-1):
    frame1= mv[i]
    frame2= mv[i+1]
    change= frame2-frame1
    if count==0:
        if frame2 < threshold:
            for j in range(1,num+1):
                resultx.append(predata[i+j])
            result_final.append(resultx) # shape will be (n,num,rows,cols,depth)
            resultx=[]
            count=count+1
    if count == 1: 
        if frame2 > threshold: 
            if change > 0:
                count=count-1
import scipy            
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd


plt.plot(diff,color='green')
plt.plot(mv)
plt.plot(mv2,color='Red')
plt.show()

# show the extracted images
#for i in range(np.shape(result_final)[0]): # number of letters
#    for j in range(np.shape(result_final)[1]): # number of frames per letter
#        plt.imshow(result_final[i][j], interpolation='nearest')
#        plt.show()

result_final=np.array(result_final) # final result storing multiple frames per alphabet

#Convert to pillow images and reconvert to numpy array to enter into albert's code

from PIL import Image
from numpy import array

result2=[] 
result2_final=[]
for i in range(np.shape(result_final)[0]): # number of letters
    for j in range(np.shape(result_final)[1]): # number of frames per letter
        img = Image.fromarray(result_final[i][j])
        img2arr = array(img)
        result2.append(img2arr)
    result2_final.append(result2)
    result2=[]

from demo import localize
result2_final = np.array(result2_final)
coordinates = localize(result2_final) # ret is np array with shape (n,num,4,2)
#print(coordinates)

def crop_hand(frame, coords):
    """
    Given coords which gives a square bounding box of the hand, crop from frame.
    frame: shape is (720,1280,3), np array
    coords: shape is (4,2) with 4-[x,y] coordinates, np array
    returns: np array with shape (l,l,3) where l is length of bounding box
    """
    upper_left = [int(c) for c in coords[0,:]] # has [x,y] for upper left
    upper_right = [int(c) for c in coords[1,:]]
    lower_right = [int(c) for c in coords[2,:]]
    lower_left = [int(c) for c in coords[3,:]]
    crop = frame[upper_left[1]:lower_left[1],upper_left[0]:upper_right[0],:]
    plt.imshow(crop, interpolation='nearest')
    plt.show()
    return crop

# for every letter, pick one valid frame; if no valid frames, the letter will be skipped
cropped = [] # list of np arrays, each array having shape (l,l,3) where l is length of bounding box
for i in range(np.shape(coordinates)[0]):
    for j in range(np.shape(coordinates)[1]):
        if np.any(coordinates[i,j,:,:]): # if coordinates not all zeros, then valid frame
            cropped.append(crop_hand(result2_final[i,j,:,:,:], coordinates[i,j,:,:]))
            break # found valid frame, go to next letter

# feed each list element in cropped to CNN to predict


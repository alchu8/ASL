# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:42:23 2020

@author: harsh
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



# loop to extract frames such that only 1 frame is extracted between every supposed movement
count=0
result=[]
#num = 4 # number of frames to extract between movements
for i in range(len(mv)-1):
    frame1= mv[i]
    frame2= mv[i+1]
    change= frame2-frame1
    if count==0:
        if frame2 < threshold:
            #for j in range(1,num+1):
            #    result.append(predata[i+j])
            result.append(predata[i+j])
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
for i in range(len(result)):
    plt.imshow(result[i], interpolation='nearest')
    plt.show()

result=np.array(result)

#Convert to pillow images and reconvert to numpy array to enter into albert's code

from PIL import Image
from numpy import array

result2=[] 
for i in range(len(result)):
    img = Image.fromarray(result[i])
    img2arr = array(img)
    result2.append(img2arr)

from demo import localize
result2 = np.array(result2)
ret = localize(result2)
print(ret)

#counter = 0 
#for i in range(np.shape(ret)[0]):
#    for 
#    counter += 1
#    if np.any(ret[i,:,:]):
        



from PIL import Image
import numpy as np
import sys
import os
sys.path.insert(0, "src/")
import multi_hand_tracker as mht

img_path = "C:/Users/alchu/Pictures/images/"

palm_model_path = "./models/palm_detection_without_custom_op.tflite"
landmark_model_path = "./models/hand_landmark.tflite"
anchors_path = "./data/anchors.csv"

# Initialise detector
# the independent flag makes the detector process each image independently, false for videos
detector = mht.MultiHandTracker(palm_model_path, landmark_model_path, anchors_path, independent = True)

def localize(frames):
    """
    Localize the hands in each of the frames.
    :param frames: Shape (n,rows,cols,depth) n being the number of letters in the sequence.
    Image files must be opened with Image module from PIL.
    :return: For each frame, a square bounding box [x,y] coordinates with shape (n,4,2).
    """
    n,r,c,_ = np.shape(frames)
    coords = np.zeros((n,4,2))
    for i in range(n):
        arr = np.array(frames[i,:,:,:])
        # get predictions
        try:
            kp_list, box_list = detector(arr)
        except TypeError:
            print("Error in running hand detector on the ith frame: " + str(i) + ". Try another frame.")
            print("Also make sure to use Image module from PIL to open images, NOT cv2 imread.")
            raise
        if kp_list[0] is None:
            print("No keypoints found in the ith frame: " + str(i) + ". Try another frame.")
            print("Also make sure to use Image module from PIL to open images, NOT cv2 imread.")
            raise RuntimeError
        # get bounding box
        x = []
        y = []
        for kp in kp_list[0]:  # first set of keypoints
            x.append(kp[0])
            y.append(kp[1])
        minx = max(min(x) - 25, 0)  # padding
        miny = max(min(y) - 25, 0)
        maxx = max(x) + 25
        maxy = max(y) + 25
        width = maxx - minx
        height = maxy - miny
        l = max(width, height)  # square bounding box
        minx = max(minx - ((l - width) / 2), 0)
        maxx = minx + l
        assert maxx < c
        miny = max(miny - ((l - height) / 2), 0)
        maxy = miny + l
        assert maxy < r
        bbox = []
        bbox.append([minx, miny])
        bbox.append([maxx, miny])
        bbox.append([maxx, maxy])
        bbox.append([minx, maxy])
        coords[i,:,:] = np.array(bbox)
    return coords

def process():
    """
    Prepares data for CNN training. Outputs csv file of bounding box coordinates for each frame.
    :return:
    """
    import csv
    #import plot_hand
    #f = open('hands.csv', 'w', newline='')
    #csvwriter = csv.writer(f, delimiter=',')

    #imglist = [x for x in os.listdir(img_path) if x.endswith(('.png', '.jpg'))]
    #imglist = ['p_Daniel.jpg','p_Daniel3.jpg','p_Everett2.jpg','p_Everett3.jpg','q_Daniel2.jpg',
    #           'q_Everett.jpg','q_Everett3.jpg','x_Daniel2.jpg','z_Everett.jpg','z_Everett2.jpg','z_Everett3.jpg']
    imglist = ['p_Daniel3.jpg']
    for idx, filename in enumerate(imglist):
        img = Image.open(img_path + filename)
        img = np.array(img)
        dims = np.shape(img)
        # Get predictions
        try:
            kp_list, box_list = detector(img)
        except TypeError:
            print(filename)
            continue
        if kp_list[0] is None:
            print(filename)
            continue
        # Get bounding box
        x = []
        y = []
        for kp in kp_list[0]: # first set of keypoints
            x.append(kp[0])
            y.append(kp[1])
        minx = max(min(x) - 25, 0) # padding
        miny = max(min(y) - 25, 0)
        maxx = max(x) + 25
        maxy = max(y) + 25
        width = maxx - minx
        height = maxy - miny
        l = max(width,height) # square bounding box
        minx = max(minx - ((l - width) / 2), 0)
        maxx = minx + l
        assert maxx < dims[1]
        miny = max(miny - ((l - height) / 2), 0)
        maxy = miny + l
        assert maxy < dims[0]
        bbox = []
        bbox.append([minx,miny])
        bbox.append([maxx,miny])
        bbox.append([maxx,maxy])
        bbox.append([minx,maxy])
        bbox = [np.array(bbox), None]
        #print(kp_list)
        #print(bbox)
        #print(np.shape(bbox[0]))
        #csvwriter.writerow([filename, bbox[0]])
        # Plot predictions
        # try:
        #     plot_hand.plot_img(img, kp_list, bbox, save="C:/Users/alchu/Pictures/out/" + filename)
        # except TypeError: # multiple sets of keypoints bug out the plot function
        #     print(filename)
        #     continue
    #f.close()

#if __name__ == "__main__":
#    process()
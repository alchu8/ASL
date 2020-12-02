from PIL import Image
import numpy as np
import sys, os
sys.path.insert(0, "src/")
import multi_hand_tracker as mht

palm_model_path = "./models/palm_detection_without_custom_op.tflite"
landmark_model_path = "./models/hand_landmark.tflite"
anchors_path = "./data/anchors.csv"

# Initialise detector
# the independent flag makes the detector process each image independently, false for videos
detector = mht.MultiHandTracker(palm_model_path, landmark_model_path, anchors_path, independent = True)
padding = 35 # in pixels

def localize(frames):
    """
    Localize the hands in each of the frames of a video.
    :param frames: Shape (n,num,rows,cols,depth) n being the number of letters in the sequence, and
    num being the number of frames per letter. Image files must be opened with Image module from PIL.
    :return: For each frame, a square bounding box [x,y] coordinates with shape (n,num,4,2).
    """
    n,num,r,c,_ = np.shape(frames)
    coords = np.zeros((n,num,4,2))
    for i in range(n):
        for j in range(num):
            arr = np.array(frames[i,j,:,:,:])
            # get predictions
            try:
                kp_list, box_list = detector(arr)
            except TypeError:
                print("Error in running hand detector on the ith frame: " + str(i) + ". Try another frame.")
                print("Also make sure to use Image module from PIL to open images, NOT cv2 imread.")
                #raise
                continue
            if kp_list[0] is None:
                print("No keypoints found in the ith frame: " + str(i) + ". Try another frame.")
                print("Also make sure to use Image module from PIL to open images, NOT cv2 imread.")
                #raise RuntimeError
                continue
            # get bounding box
            x = []
            y = []
            for kp in kp_list[0]:  # first set of keypoints
                x.append(kp[0])
                y.append(kp[1])
            minx = max(min(x) - padding, 0)
            miny = max(min(y) - padding, 0)
            maxx = max(x) + padding
            maxy = max(y) + padding
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
            coords[i,j,:,:] = np.array(bbox)
    return coords

def process():
    """
    Prepares data for CNN training.
    :return: Outputs csv file of bounding box coordinates for each frame.
    """
    import csv
    import plot_hand
    f = open('hands.csv', 'w', newline='')
    csvwriter = csv.writer(f, delimiter=',')

    img_path = "C:/Users/alchu/Pictures/Camera Roll/" # CHANGE
    imglist = [x for x in os.listdir(img_path) if x.endswith(('.png', '.jpg'))]
    #imglist = ['x_Albert.jpg','x_Albert2.jpg','x_Albert3.jpg']
    #---------------Don't edit from this point on------------------
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
            print(filename + ", keypoints is None.")
            continue
        # Get bounding box
        x = []
        y = []
        for kp in kp_list[0]: # first set of keypoints
            x.append(kp[0])
            y.append(kp[1])
        minx = max(min(x) - padding, 0)
        miny = max(min(y) - padding, 0)
        maxx = max(x) + padding
        maxy = max(y) + padding
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
        csvwriter.writerow([filename, bbox[0]])
        # Plot predictions
        try:
            plot_hand.plot_img(img, kp_list, bbox, save=None)
        except TypeError: # multiple sets of keypoints bug out the plot function
            print(filename + " has multiple sets of keypoints.")
            continue
    f.close()

if __name__ == "__main__":
    process()

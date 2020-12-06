## Requirements

- pillow
- numpy
- opencv
- tensorflow
- shapely
- scipy
- pandas

## Usage

Call VideoFrameExtractor.py and give path to mp4 file as command line argument, and use --subsample flag for Everett's videos.  

`python VideoFrameExtractor.py --file ../advance.mp4 --subsample`  

It will show the sequence of cropped images corresponding to detected hand signals.  

**Note**: image arrays must be opened with Image module from PIL. 
When passing in to localize, they can be converted to numpy arrays, but they ***must*** be opened as PIL images, and not as cv2 images.

Call demo.py to validate that the detector can localize the hand in your images. Just change the local path name in code where it says CHANGE. 
The code will plot the bounding box with your hand keypoints if detection was successful. Unsuccessful image filenames will be printed to console.  

`python demo.py`

## Acknowledgments

This work is a study of models developed by Google and distributed as a part of the [Mediapipe](https://github.com/google/mediapipe) framework.   
@metalwhale for the Python Wrapper for single hand tracking and for removing custom operations for the Palm Detections model. 
Modules and models gathered at https://github.com/JuliaPoo/MultiHand-Tracking. 
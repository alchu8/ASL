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
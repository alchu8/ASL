## Requirements

- pillow
- numpy
- opencv
- tensorflow
- shapely
- scipy

## Usage

`from demo import localize`  
`coordinates = localize(frames)`

**Note**: image arrays must be opened with Image module from PIL. 
When passing in to localize, they can be converted to numpy arrays, but they ***must*** be opened as PIL images, and not as cv2 images.
# Code for showing webcam feed is taken from Adrian Rosebrock's tutorial @
# https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
import cv2

print('OpenCV version: {}'.format(cv2.__version__))
# Read test image and show it
img = cv2.imread('test.png')
cv2.imshow('Window', img)
cv2.waitKey(0)

# Read from the video webcam
camera = cv2.VideoCapture(0)
# Loop over the frames of the video
while True:
    # Grab the current frame
    _, frame = camera.read()

    # Show the frame and record if the user presses a key
    cv2.imshow("Webcam Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    # If the `q` key is pressed, break from the loop
    if key == ord("q"):
        break

# Cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

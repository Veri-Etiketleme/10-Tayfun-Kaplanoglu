# Test code is from Adrian Rosebrock's blog post @ https://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/
import cv2

image = cv2.imread("test.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# SIFT features
sift = cv2.xfeatures2d.SIFT_create()
(kps, descs) = sift.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))

# SURF features
surf = cv2.xfeatures2d.SURF_create()
(kps, descs) = surf.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))

# Other keypoint detectors and local invariant descriptors
# KAZE
kaze = cv2.KAZE_create()
(kps, descs) = kaze.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
# AKAZE
akaze = cv2.AKAZE_create()
(kps, descs) = akaze.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
# BRISK
brisk = cv2.BRISK_create()
(kps, descs) = brisk.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))

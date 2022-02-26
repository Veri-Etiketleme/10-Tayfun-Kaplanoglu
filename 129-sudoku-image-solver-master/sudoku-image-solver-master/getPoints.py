import cv2
import argparse
import numpy as np

def selectPoint(event, x, y, flags, params):
    cpy, points, maxPoints = params
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < maxPoints:
        points.append([x-2,y-2])
        cv2.rectangle(cpy, (x-7, y-7), (x+3, y+3), color=(0, 255, 0))
    
def _getPoints(img, numPoints, wname):
    cpy = img.copy()
    points = []
    cv2.namedWindow(wname)
    cv2.setMouseCallback(
            wname,
            selectPoint, 
            [cpy, points, numPoints])
    
    while True:
        cv2.imshow(wname, cpy)
        key = cv2.waitKey(1) & 0xFF
    
        if key == 13 and len(points) == numPoints:
            break
        
        if key == ord("c"):
            points.clear()
            cpy = img.copy()
            cv2.setMouseCallback(
                    wname,
                    selectPoint, 
                    [cpy,  points, numPoints])

    return points

def getPoints(img, numPoints, wname='image'):
    return tuple(map(tuple, _getPoints(img, numPoints, wname)))

def getSquare(img, offset=0, wname='Select the 4 corners, then press Enter'):
    # Sort 4 srcPts into row, column order
    scale = max(img.shape)/500
    if scale > 1:
        img = cv2.resize(img, (int(img.shape[0]/scale), int(img.shape[1]/scale)))
    img = cv2.copyMakeBorder(img, 15, 15, 15, 15, cv2.BORDER_CONSTANT, None, (255, 255, 255)) 
    pts = _getPoints(img, 4, wname)
    pts.sort(key = lambda x: x[0]**2 + x[1]**2)
    pts2 = sorted(pts[1:3], key = lambda x: x[1])
    pts2.insert(0, pts[0]) 
    pts2.append(pts[-1]) 

    # Add offset to 4 corners
    for i, p in enumerate(pts2):
        p[0] = p[0] + (offset if i % 2 else -offset)
        p[1] = p[1] + (offset if i > 1 else -offset)
    return tuple(map(tuple, pts2)), img

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", 
            required = True, 
            help = "Path to image")
    ap.add_argument("-p", "--points", 
            required=False, default=4,  
            help="Number of points to capture")
    args = vars(ap.parse_args())

    print(getPoints(cv2.imread(args["image"]), int(args["points"])))

import cv2
import argparse
import joblib

from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np 

from coordTransform import coordTransform
from getPoints import getSquare
from puzzleSolver import solve

def createSquareMask(sz, t):
    """
    Create a square mask with size sz, and border thickness t
    """
    t = min(sz // 2, t)
    mask = [[] for _ in range(sz)] 
    for i in range(t):
        mask[i].extend([0 for _ in range(sz)])
    for j in range(t, sz-t):
        mask[j].extend([0 for _ in range(t)])
        mask[j].extend([255 for _ in range(sz - 2*t)])
        mask[j].extend([0 for _ in range(t)])
    for k in range(sz-t, sz):
        mask[k].extend([0 for _ in range(sz)])

    mask = np.array(mask, dtype=np.uint8)
    mask = mask[:, :, np.newaxis] 
    mask = np.concatenate((mask, mask.copy(), mask.copy()), axis=2)
    return mask 
    
def parseSquares(src, template, offset=0):
    sz = 34 

    squareLocs = [[] for _ in range(9)]
    curx, cury = 0, 0
    img = src.copy()
    for i in range(9):
        for j in range(9):
            if i == 0 and j == 0:
                curx, cury = 0, 0
                res = cv2.matchTemplate(src[cury:cury+sz+10, curx:curx+sz+10], template, cv2.TM_CCOEFF_NORMED)
            elif i == 0:
                curx, cury = squareLocs[i][j-1][0] + sz, squareLocs[i][j-1][1]
                res = cv2.matchTemplate(src[0:cury+sz+10, curx-10:curx+sz+10], template, cv2.TM_CCOEFF_NORMED)
            elif j == 0:
                curx, cury = squareLocs[i-1][j][0], squareLocs[i-1][j][1] + sz
                res = cv2.matchTemplate(src[cury-10:cury+sz+10, 0:curx+sz+10], template, cv2.TM_CCOEFF_NORMED)
            else:
                curx, cury = squareLocs[i-1][j][0], squareLocs[i-1][j][1] + sz
                res = cv2.matchTemplate(src[cury-10:cury+sz+10, curx-10:curx+sz+10], template, cv2.TM_CCOEFF_NORMED)

            _, curmax, _, curloc = cv2.minMaxLoc(res)
            x, y = curloc
            
            if i == 0 and j == 0:
                pass
            elif i == 0:
                x += curx - 10
            elif j == 0:
                y += cury - 10
            else:
                x += curx - 10
                y += cury - 10

            squareLocs[i].append((x, y))

    return squareLocs

def parseSums(src, squares):
    sqsz = 34 
    tot_sum = ((sqsz-6)**2)*255
    sums = [[] for _ in range(9)]
    for r in range(9):
        for c in range(9):
            rstart = squares[r][c][1]
            rend = rstart + sqsz
            cstart = squares[r][c][0] 
            cend = cstart + sqsz
            s_cropped = src[rstart+4:rend-3, cstart+4:cend-3] # crop out edges to predict empty squares
            sums[r].append(s_cropped.size * 255 - s_cropped.sum()) # Invert the sum so filled squares are higher

    return sums

def parseSumsNums(src, squares, classifier):

    # Pass each mask over each square.  Attempt to determine it's contents
    sqsz = 34 
    clf = joblib.load(classifier)
    tot_sum = ((sqsz-6)**2)*255
    parsed = [[] for _ in range(9)]
    sums = [[] for _ in range(9)]
    for r in range(9):
        for c in range(9):
            rstart = squares[r][c][1]
            rend = rstart + sqsz
            cstart = squares[r][c][0] 
            cend = cstart + sqsz
            s = src[rstart:rend, cstart:cend, 0].flatten() # feature for classifier
            s_cropped = src[rstart+4:rend-3, cstart+4:cend-3] # crop out edges to predict empty squares
            sums[r].append(s_cropped.size * 255 - s_cropped.sum()) # Invert the sum so filled squares are higher
            if sums[r][-1] > 50000:
                print(clf.predict_proba([s]))
                parsed[r].append(clf.predict([s])[0])
            else:
                parsed[r].append(0)

    return (sums, parsed)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imagePath", 
            required = True, 
            help = "Path to images")
    ap.add_argument("-c", "--classifier", 
            required = False, 
            default = "classifier.joblib",
            help = "Classifier joblib file")
    args = vars(ap.parse_args())
    offset = 5

    src = cv2.imread(args["imagePath"]) 
    classifier = args["classifier"]
    srcPts, src = getSquare(src, offset)
    warped, tformed = coordTransform(src, srcPts, 306 + 2*offset)
    tformed = tformed[:, :, np.newaxis] 
    tformed = np.concatenate((tformed, tformed.copy(), tformed.copy()), axis=2)
    parsedSquares = parseSquares(tformed, createSquareMask(34, 2), offset)
    _, parsedNums = parseSumsNums(tformed, parsedSquares, classifier)
    p = np.array(parsedNums)
    print(p)
    s = solve(p)
    print(s)

    cv2.namedWindow("Solution")

    for rowidx, row  in enumerate(parsedSquares):
        for colidx, sq  in enumerate(row):
            if p[rowidx, colidx] == 0:
                warped = cv2.putText(warped, str(s[rowidx, colidx]), (sq[0] + 5, sq[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Solution", warped)
    while True:
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

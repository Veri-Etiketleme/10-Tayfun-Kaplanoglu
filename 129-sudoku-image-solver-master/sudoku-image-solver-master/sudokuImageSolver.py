import argparse

import cv2
import numpy as np

from coordTransform import coordTransform
from getPoints import getSquare
from puzzleSolver import solve
from parsePuzzle import createSquareMask, parseSquares, parseSumsNums

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
    if isinstance(s, bool):
        print('Could not solve puzzle, check parsed output')
    else:
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

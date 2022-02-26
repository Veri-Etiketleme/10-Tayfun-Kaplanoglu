import cv2
import argparse
import os
import sys
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

p = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(p.parent))

from getPoints import getSquare
from coordTransform import coordTransform
from parsePuzzle import parseSquares, parseSums, createSquareMask

def ingest(files, outdir):
    offset  = 5

    getnum = re.compile('^\d+')
    nextFileNum = 0
    if len(os.listdir(outdir)) > 1:
        nextFileNum = 1 + max(int(re.match(getnum, f).group(0)) for f in os.listdir(outdir))

    for f in files:
        if os.path.exists(f):
            src = cv2.imread(f)
            pts, src = getSquare(src, offset)
            _, tformed = coordTransform(src, pts, 306 + 2*offset)
            tformed = tformed[:, :, np.newaxis] 
            tformed = np.concatenate((tformed, tformed.copy(), tformed.copy()), axis=2)
            parsedSquares = parseSquares(tformed, createSquareMask(34, 2), offset)
            parsedSums = parseSums(tformed, parsedSquares)
            for r in range(9):
                for c in range(9):
                    if parsedSums[r][c] > 50000:
                        x, y = parsedSquares[r][c]
                        cur = tformed[y:y+34, x:x+34]
                        print('writing:', os.path.join(outdir, str(nextFileNum) + '_n' + '.png'), end=": ")
                        if cv2.imwrite(os.path.join(outdir, str(nextFileNum) + '_n' +'.png'), cur):
                            print('Success!')
                            nextFileNum += 1
                        else:
                            print('Failure!')
            os.rename(os.path.join(os.getcwd(), f) , os.path.join(os.getcwd(), f + '_ingested'))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir', nargs=1)
    ap.add_argument('files', nargs='+')
    args = vars(ap.parse_args())
    ingest(args['files'], args['outdir'][0]) 

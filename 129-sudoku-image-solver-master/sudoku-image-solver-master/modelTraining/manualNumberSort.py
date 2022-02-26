import os
import argparse

import cv2

def moveNumbers(files, outdir):

    # Create subdirectories 0 thru 9 if they don't exist already
    for x in range(0,10):
        p = os.path.join(outdir, str(x))
        if not os.path.isdir(p):
            os.mkdir(p)

    print('Classify the displayed numbers using the keyboard')
    for f in files:
        img = cv2.imread(f) 
        img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, None, (255, 255, 255)) 
        cv2.namedWindow("image")

        while True:
            cv2.imshow("image", img)
            key = cv2.waitKey(1) & 0xFF
        
            if key >= ord('0') and key <= ord('9'):
                nf = os.path.join(outdir, chr(key), f.split('/')[-1])
                print('Moving {} to folder {}'.format(f, nf)) 
                os.rename(f, nf)
                break
                
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir', nargs=1)
    ap.add_argument('files', nargs='+')
    args = vars(ap.parse_args())
    moveNumbers(args['files'], args['outdir'][0]) 

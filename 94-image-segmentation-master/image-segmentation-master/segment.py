#!/usr/bin/python
import sys
import os
import argparse
import math
from PIL import Image, ImageDraw

class isDirectory(argparse.Action):
  def __call__(self, parser, namespace, values, optionString=None):
    if not os.path.isdir(values):
      raise argparse.ArgumentTypeError("destination:{0} is not a valid directory".format(values))
    else:
      setattr(namespace, self.dest, values)

def sum4tuple(x, y):
  return (x[0]+y[0], x[1]+y[1], x[2]+y[2], x[3]+y[3])

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Redraw images according to the mean color value of individual segments')
  parser.add_argument('source', type=str, help='Source file (or directory) specifying image(s) to mark')
  parser.add_argument('-d', '--destination', type=str, action=isDirectory, help='Destintion path to output marked files to')
  parser.add_argument('-s', '--segments', type=int, help='Number of segments to divide each axis into (overwrites x and y options when specified)')
  parser.add_argument('-x', '--xSegments', type=int, default=3, help='Number of segments to divide the x axis into')
  parser.add_argument('-y', '--ySegments', type=int, default=3, help='Number of segments to divide the y axis into')
  arguments = parser.parse_args()
  if arguments.segments:
    arguments.xSegments = arguments.segments
    arguments.ySegments = arguments.segments
  
  if os.path.isdir(arguments.source):
    dirPath, _, fileNames = os.walk(arguments.source).next()
    imageFiles = [os.path.join(dirPath, fileName) for fileName in fileNames]
  else:
    imageFiles = [arguments.source]
  
  for imageFile in imageFiles:
    im = Image.open(imageFile)
    draw = ImageDraw.Draw(im)
    xWidth = im.size[0]/arguments.xSegments
    yWidth = im.size[1]/arguments.ySegments
    for n in range(0, arguments.xSegments):
      for m in range(0, arguments.ySegments):
        segData = im.crop((n*xWidth, m*yWidth, (n+1)*xWidth, (m+1)*yWidth)).getdata()
        segLen = len(segData)
        segSum = reduce(sum4tuple, segData)
        segMean = tuple(map(lambda x: int(x/segLen), segSum))
        draw.rectangle((n*xWidth, m*yWidth, (n+1)*xWidth, (m+1)*yWidth), segMean)

    del draw
    
    if arguments.destination:
      im.save(os.path.join(arguments.destination, os.path.basename(imageFile)))
    else:
      im.show()


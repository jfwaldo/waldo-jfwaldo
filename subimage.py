#!/usr/bin/env python3

import cv2

import sys
from os import access, R_OK
from os.path import isfile
#import asyncio
#import numpy # could be useful for more precise analysis of the template matching result, e.g. to reduce false positives
#import time # for tracking timing results

## NOTE: asynchronous image loading is unused... some quick tests showed that using async instead of serial synchronous
## loads may be more performant once there are at least 10 or 20 images to load, so it's not applicable for our case here.

## a couple other notes: if load_images is used, then the resultant list will need to be checked for `None` to ensure
## no failures occurred. alternatively, asyncio.wait may give more granular control over aborting if it's important to
## fail parallel loading as soon as possible
# def load_images(*imagePaths):
#   loop = asyncio.get_event_loop()
#   images = loop.run_until_complete(asyncio.gather(*[_load_image(imagePath) for imagePath in imagePaths]))
#   return images

# async def _load_image(imagePath):
#   image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
#   return image



#
# Checks to see if the file at the given path exists, is not a directory, and is readable
#
def can_load_file(imagePath):
  return isfile(imagePath) and access(imagePath, R_OK)

#
# Reads a given image path with color mode and returns an opencv image object (or None if the read failed)
#
# NOTE: for now, just using grayscale to reduce computation time. But, not sure what the exact use case would be etc.
def load_image(imagePath, flags=cv2.IMREAD_GRAYSCALE):
   image = cv2.imread(imagePath, flags)
   return image


# Returns "x,y" if a match is found, otherwise returns -1,-1 if a match is not found.
if __name__ == "__main__":
  #TODO: add `--help` and `--version` support
  if len(sys.argv) != 3:
    sys.exit('Error: Two image paths must be provided.\n\nExample: python3 subimage.py path/to/image1.jpg path/to/image2.jpg\n')

  imagePath1 = sys.argv[1]
  imagePath2 = sys.argv[2]

  # I'm not sure what the environment looks like where this could be used, i.e. what kinds of failure modes are possible.
  # So I'm covering the basics for now: file exists, is not a directory, is readable, and then relying on imread
  # to determine whether or not the file is an image/is compatible with opencv.
  if can_load_file(imagePath1) is not True:
    sys.exit(f'Error: {imagePath1} does not exist, is a directory, or has insufficient read privileges.')
  if can_load_file(imagePath2) is not True:
    sys.exit(f'Error: {imagePath2} does not exist, is a directory, or has insufficient read privileges.')

  #TODO: maybe add file size check to make sure imagePath1 and imagePath2 can both be stored in memory--not sure about environment
  # (also check if opencv supports disk-based operations for these larger cases)

  image1 = load_image(imagePath1)
  if image1 is None:
    sys.exit(f'Error: {imagePath1} is not a compatible image.')

  image2 = load_image(imagePath2)
  if image2 is None:
    sys.exit(f'Error: {imagePath2} is not a compatible image.')

  # make sure not to do variable unpacking here--shape can sometimes have 3 elements instead of 2 depending on the given read flags
  width1 = image1.shape[0]
  height1 = image1.shape[1]
  width2 = image2.shape[0]
  height2 = image2.shape[1]

  # for now, i'm assuming rotation/resize transforms are not being taken into account, so it will simply fail if the
  # dimensions are not compatible
  if width1 <= width2 and height1 <= height2:
    fullImage = image2
    templateImage = image1
  elif width2 < width1 and height2 < height1:
    fullImage = image1
    templateImage = image2
  else: # neither image is strictly smaller than the other along both x and y, so a determination cannot be made
    sys.exit(f'Error: Bad dimensions. Neither image can be a proper subset of the other: ({width1}, {height1}), ({width2}, {height2})')

  # probably overkill depending on the reqs, but since we duplicated the objs above and they may be large...
  del image1, image2

  # adapted from: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html

  # NOTE: I'm not sure what the best approach is here for the template matching. There are six algos available, as listed here:
  # https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html#which-are-the-matching-methods-available-in-opencv

  # In the req writeup, it mentions being able to support lossy jpeg images. I'm not sure if that means that the template image
  # could have different artifacts/compression levels from the full image... in that case, maybe blurring both images slightly
  # before performing the template match would make sense. But I'm not an opencv/image processing expert.

  # For now, I'm using normalized cross-correlation, which gives an nd-array of values normalized to [-1, +1]. So I'm just looking
  # for the global max values and set a very high threshold as close to +1 as possible to eliminate false positives. Probably there
  # is a better solution here, e.g. looking at the median of the top n max values or something.
  matchTemplateResult = cv2.matchTemplate(fullImage, templateImage, cv2.TM_CCORR_NORMED)
  minValue, maxValue, minLocation, maxLocation = cv2.minMaxLoc(matchTemplateResult)

  if maxValue >= 0.98: # super rough threshold determined empirically--adjust as desired. This could be moved to a constant at the top of file or as an env var
    xTop = maxLocation[0] 
    yTop = maxLocation[1]
    print(f"{xTop},{yTop}")
  else:
    print("-1,-1")

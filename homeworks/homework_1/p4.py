#!/usr/bin/env python3
import cv2
import numpy


def normxcorr2(template, image):
  """Do normalized cross-correlation on grayscale images.

  When dealing with image boundaries, the "valid" style is used. Calculation
  is performed only at locations where the template is fully inside the search
  image.

  Args:
  - template (2D float array): Grayscale template image.
  - image (2D float array): Grayscale search image.

  Return:
  - scores (2D float array): Heat map of matching scores.
  """
  raise NotImplementedError  #TODO


def find_matches(template, image, thresh=None):
  """Find template in image using normalized cross-correlation.

  Args:
  - template (3D uint8 array): BGR template image.
  - image (3D uint8 array): BGR search image.

  Return:
  - coords (2-tuple or list of 2-tuples): When `thresh` is None, find the best
      match and return the (x, y) coordinates of the upper left corner of the
      matched region in the original image. When `thresh` is given (and valid),
      find all matches above the threshold and return a list of (x, y)
      coordinates.
  - match_image (3D uint8 array): A copy of the original search images where
      all matched regions are marked.
  """
  raise NotImplementedError  #TODO

def main(argv):
  template_img_name = argv[0]
  search_img_name = argv[1]

  template_img = cv2.imread("data/" + template_img_name + ".png", cv2.IMREAD_COLOR)
  search_img = cv2.imread("data/" + search_img_name + ".png", cv2.IMREAD_COLOR)

  _, match_image = find_matches(template_img, search_img, 0.95)

  cv2.imwrite("output/" + search_img_name + ".png", match_image)


if __name__ == "__main__":
  main(sys.argv[1:])

# example usage: python p4.py face king
# expected results can be seen here: https://hackmd.io/toS9iEujTtG2rPoxAdPk8A?view
#!/usr/bin/env python3
import cv2
from p1n2 import *
import os

def best_match(object_database, test_object):
    '''

    Args:
        object_database: a list training images and each training image is stored as dictionary with keys name and image
        test_object: test image, a 2D unit8 array

    Returns:
        object_names: a list of shapes from object_database whose patterns match the test image.
        For example, the results for test1.png should be [['oval', (x1,y1)], ['pacMan',(x2,y2)],['rhombus',(x3,y3)]], where (x,y) is the center position of each object.
        You will need to use functions from p1n2.py
    '''
  # TODO
  return object_names

def main(argv):
  img_name = argv[0]
  test_img = cv2.imread('test/' + img_name + '.png', cv2.IMREAD_COLOR)

  train_im_names = os.listdir('train/')
  object_database = []
  for train_im_name in train_im_names:
      train_im = cv2.imread('train/' + train_im_name, cv2.IMREAD_COLOR)
      object_database.append({'name': train_im_name, 'image':train_im})
  object_names = best_match(object_database, test_img)
  print(object_names)


if __name__ == '__main__':
  main(sys.argv[1:])

# example usage: python p3.py many_objects_1.png

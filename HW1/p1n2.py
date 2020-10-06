#!/usr/bin/env python3
import cv2
import numpy as np
import sys


def binarize(gray_image, thresh_val):
    """ Function to threshold grayscale image to binary
        Sets all pixels lower than threshold to 0, else 255

        Args:
        - gray_image: grayscale image as an array
        - thresh_val: threshold value to compare brightness with

        Return:
        - binary_image: the thresholded image
    """
    # TODO: 255 if intensity >= thresh_val else 0

    return binary_image

def label(binary_image):
    """ Function to labeled components in a binary image
        Uses a sequential labeling algorithm

        Args:
        - binary_image: binary image with multiple components to label

        Return:
        - lab_im: binary image with grayscale level as label of component
    """

    _, lab_im = cv2.connectedComponents(binary_image)
    return lab_im


def get_attribute(labeled_image):
    """ Function to get the attributes of each component of the image
        Calculates the position, orientation, and roundedness

        Args:
        - labeled_image: image file with labeled components

        Return:
        - attribute_list: a list of the aforementioned attributes
    """
    # TODO

    return attribute_list

def draw_attributes(image, attribute_list):
    num_row = image.shape[0]
    attributed_image = image.copy()
    for attribute in attribute_list:
        center_x = (int)(attribute["position"]["x"])
        center_y = (int)(attribute["position"]["y"])
        slope = np.tan(attribute["orientation"])

        cv2.circle(attributed_image, (center_x, num_row - center_y), 2, (0, 255, 0), 2)
        cv2.line(
            attributed_image,
            (center_x, num_row - center_y),
            (center_x + 20, int(20 * (-slope) + num_row - center_y)),
            (0, 255, 0),
            2,
        )
        cv2.line(
            attributed_image,
            (center_x, num_row - center_y),
            (center_x - 20, int(-20 * (-slope) + num_row - center_y)),
            (0, 255, 0),
            2,
        )
    return attributed_image



def detect_edges(image, sigma, threshold):
  """Find edge points in a grayscale image.

  Args:
  - image (2D uint8 array): A grayscale image.

  Return:
    - edge_image (2D binary image): each location indicates whether it belongs to an edge or not
  """
  raise NotImplementedError  #TODO

def get_edge_attribute(labeled_image, edge_image):
  '''
  Function to get the attributes of each edge of the image
        Calculates the angle, distance from the origin and length in pixels
  Args:
    labeled_image: binary image with grayscale level as label of component
    edge_image (2D binary image): each location indicates whether it belongs to an edge or not

  Returns:
     attribute_list: a list of list [[dict()]]. For example, [lines1, lines2,...],
     where lines1 is a list and it contains lines for the first object of attribute_list in part 1.
     Each item of lines 1 is a line, i.e., a dictionary containing keys with angle, distance, length.
     You should associate objects in part 1 and lines in part 2 by putting the attribute lists in same order.
     Note that votes in HoughLines opencv-python is not longer available since 2015. You will need to compute the length yourself.
  '''
  # TODO
  return attribute_list

def draw_edge_attributes(image, attribute_list):
    attributed_image = image.copy()
    for lines in attribute_list:
        for line in lines:
            angle = (float)(line["angle"])
            distance = (float)(line["distance"])

            a = np.cos(angle)
            b = np.sin(angle)
            x0 = a * distance
            y0 = b * distance
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

            cv2.line(
                attributed_image,
                pt1,
                pt2,
                (0, 255, 0),
                2,
            )

    return attributed_image

def get_circle_attribute(labeled_image, edge_image):

    # extra credits
    raise NotImplementedError  #TODO

def main(argv):
  img_name = argv[0]
  thresh_val = int(argv[1])

  img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)

  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  cv2.imwrite('output/' + img_name + "_gray.png", gray_image)

  # part 1
  binary_image = binarize(gray_image, thresh_val=thresh_val)
  cv2.imwrite('output/' + img_name + "_binary.png", binary_image)

  labeled_image = label(binary_image)
  num = np.unique(labeled_image).max()
  cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image*255//num)

  attribute_list = get_attribute(labeled_image)
  print('attribute list:')
  print(attribute_list)

  attributed_image = draw_attributes(img, attribute_list)
  cv2.imwrite("output/" + img_name + "_attributes.png", attributed_image)


  # part 2
  # feel free to tune hyperparameters or use double-threshold
  edge_image = detect_edges(gray_image, sigma=1., threshold=0.3)
  cv2.imwrite("output/" + img_name + "_edges.png", edge_image)

  edge_attribute_list = get_edge_attribute(labeled_image, edge_image)
  print('edge attribute list:')
  print(edge_attribute_list)

  attributed_edge_image = draw_edge_attributes(img, edge_attribute_list)
  cv2.imwrite("output/" + img_name + "_edge_attributes.png", attributed_edge_image)

  # extra credits for part 2: show your circle attributes and plot circles
  # circle_attribute_list = get_circle_attribute(labeled_image, edge_image)
  # attributed_circle_image = draw_circle_attributes(img, circle_attribute_list)
  # cv2.imwrite("output/" + img_name + "_circle_attributes.png", attributed_circle_image)

if __name__ == '__main__':
  main(sys.argv[1:])
# example usage: python p1n2.py two_objects 128
# expected results can be seen here: https://hackmd.io/toS9iEujTtG2rPoxAdPk8A?view

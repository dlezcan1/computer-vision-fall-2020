#!/usr/bin/env python3
import cv2
from p1n2 import *  
import os


def combine_attribute_lists( shape_attribute, edge_attribute, circle_attribute ):
    """ This is a function to combine all of the attribute lists into one big list of dicts
    
    Args:
        shape_attribute: a list of the general shape attributes (length N)
        edge_attribute: a list of edge attributes (length N)
        circle_attribute: a list of circle attributes (length N)
    
    Returns:
        attribute_list: a combined attribute list of the folllowing
        [dict()]
            - dict() of keys:
                'shape': shape_attributes
                'edge': edge_attributes
                'circle': circle_attributes
                * None for any implies that there was no match *
                
    """
    if ( len( shape_attribute ) != len( edge_attribute ) ) and ( len( shape_attribute ) != len( circle_attribute ) ):
        raise IndexError
    
    # if
    
    attribute_list = []
    for shape_attr, edge_attr, circle_attr in zip( shape_attribute, edge_attribute, circle_attribute ):
        attribute = {'shape': shape_attr, 'edge': edge_attr, 'circle': circle_attr}
        
        # append the attribute
        attribute_list.append( attribute )
        
    # for
    
    return attribute_list
    
# combine_attribute_lists


def create_feature_vector( attribute ):
    """ NOT GOING TO WORK! Function that accepts an attribute dictionary to turn it into a feature vector
    
    Args:
        attribute: a dict of keys:
                        'shape', 'edge', 'circle'
                        
    Returns:
        a numpy array of features ['shape', 'edge', 'circle']
            [com_x, com_y, orientation, roundedness, 
            edge_angle (0 if None), edge_distance (0 if None), edge_length (0 if None),
            circle_center_x (0 if None), circle_center_y (0 if None), circle_radius (0 if None)]
    """
    raise NotImplementedError
    x = np.empty( 10 )
    
    # add the general shape parameters
    x[0] = attribute['shape']['position']['x']
    x[1] = attribute['shape']['position']['y']
    x[2] = attribute['shape']['orientation']
    x[3] = attribute['shape']['roundedness']
    
    # add the edge parameters
    if isinstance( attribute['edge'], type( None ) ):
        x[4] = x[5] = x[6] = 0
        
    # if
    
    else:
        x[4] = attribute['edge']['angle']
        x[5] = attribute['edge']['distance']
        x[6] = attribute['edge']['length']
        
    # else
    
    # add the circle parameters
    if isinstance( attribute['circle'], type( None ) ):
        x[7] = x[8] = x[9] = 0
        
    # if
    
    else:
        x[7] = attribute['circle']['position']['x']
        x[8] = attribute['circle']['position']['y']
        x[9] = attribute['circle']['radius']
        
    # else
    
    return x
    
# create_feature_vector
        

def best_match( object_database, test_object ):
    '''
    
    Args:
        object_database: a list training images and each training image is stored as dictionary with keys name and image
        test_object: test image, a 2D unit8 array
    
    Returns:
        object_names: a list of filenames from object_database whose patterns match the test image
        You will need to use functions from p1n2.py
    '''
    # TODO
    raise NotImplementedError

    object_names = []
    return object_names

# best_match


def main( argv ):
    img_name = argv[0]
    test_img = cv2.imread( 'test/' + img_name + '.png', cv2.IMREAD_COLOR )
    
    train_im_names = os.listdir( 'train/' )
    object_database = []
    for train_im_name in train_im_names:
        train_im = cv2.imread( 'train/' + train_im_name, cv2.IMREAD_COLOR )
        object_database.append( {'name': train_im_name, 'image':train_im} )
    object_names = best_match( object_database, test_img )
    print( object_names )

# main


if __name__ == '__main__':
    main( sys.argv[1:] )
    
# if

# example usage: python p3.py many_objects_1.png

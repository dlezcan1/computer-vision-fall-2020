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
        if edge_attr == None:
            edge_attr = []
            
        # if
        
        if circle_attr == None:
            circle_attr = []
            
        # if
        attribute = {'shape': shape_attr, 'edge': edge_attr, 'circle': circle_attr}
        
        # append the attribute
        attribute_list.append( attribute )
        
    # for
    
    return attribute_list
    
# combine_attribute_lists


def compare_image_to_template( test_img_attr, template_attr ):
    ''' Function to score the similarity between test_img and template'''
    # grab the object orientations and roundedness
    test_orient = test_img_attr['shape']['orientation']
    template_orient = template_attr['shape']['orientation']
    
    test_round = test_img_attr['shape']['roundedness']
    template_round = template_attr['shape']['roundedness']
    
    # compute roundedness score
    round_score = 1 - np.abs( test_round - template_round )
    
    # compute edge score
    num_edge_test = len( test_img_attr['edge'] )
    num_edge_template = len( template_attr['edge'] )
    
    if num_edge_test == num_edge_template == 0:  # no edges detected
        edge_score = 1.0  # perfect score
        
    # if
    
    elif ( num_edge_test == 0 ) or ( num_edge_template == 0 ):  # xor @ this point
        edge_score = 0.0  # no edges measured
        
    # elif
        
    else:  # they are non-zero, but not necessarily the same
        # num_match_score
#         num_edge_score = max( 1 - np.abs( num_edge_test - num_edge_template ) / num_edge_template, 0.5 )  # num edges unreliable
        num_edge_score = max( 1 - np.abs( num_edge_test - num_edge_template ) / num_edge_template, 0 )  
        
        # reorient the edges with respect to their 2nd moment orientation
        edge_test_orient = np.zeros( len( test_img_attr['edge'] ) ).tolist()
        edge_template_orient = np.zeros( len( template_attr['edge'] ) ).tolist()
        for i, edge in enumerate( test_img_attr['edge'] ):
            edge_test_orient[i] = edge['angle'] - test_orient
            
        # for
        
        for i, edge in enumerate( template_attr['edge'] ):
            edge_template_orient[i] = edge['angle'] - template_orient
            
        # for
        
        # only calculate based on minimum number of matches
        n_match_edges = min( num_edge_test, num_edge_template )
        cos_sim = lambda  x, y : np.dot( x[:n_match_edges], y[:n_match_edges] ) / ( 
            np.linalg.norm( x[:n_match_edges] ) * np.linalg.norm( y[:n_match_edges] ) )  # cosine similarity

        # edge align score
        # # start with initial alignment
        align_edge_score = 1 / 2 + cos_sim( edge_test_orient, edge_template_orient ) / 2
        
        # # compute over all alignments over permutations of test score 
        for k in range( n_match_edges - num_edge_test ):
            # rotate test vector and compute score
            rot_edge_test_orient = edge_test_orient[-k:] + edge_test_orient[:-k]
            rot_edge_test_score = 1 / 2 + cos_sim( rot_edge_test_orient, edge_template_orient ) / 2
            
            # keep the best score
            align_edge_score = max( align_edge_score, rot_edge_test_score )
        
        # for
        
        # compute edge score
        edge_score = num_edge_score * 1
        
    # else
    
    # compute circle score ( no radius b/c there could be scaling effects )
    num_circle_test = len( test_img_attr['circle'] )
    num_circle_template = len( template_attr['circle'] )
    
    if num_circle_test == num_circle_template:  # same num of circles detected
        circle_score = round_score  # double weight the roundedness of the obj
        
    # if
    
    elif num_circle_template == 0:  # no template circles, but test circles
        circle_score = 0.0  # no edges measured
        
    # elif
    
    else:  # at least 1 template circle
        circle_score = round_score * ( 1 - np.abs( num_circle_test - num_circle_template ) / num_circle_template )
        
    # else
    
    # find the mean of the scores
    score = np.mean( [round_score, edge_score, circle_score] )  # typical average
    
    return score
    
# compare_images


def get_all_attributes( object_img, threshold ):
    ''' Function to get all of the attributes from the object file
    
    Args:
        object_file: an image file to get the file from
        threshold:   the threshold for the binarization 
        
    Returns:
        A list of attribues in dict form 
            'shape': shape attributes
            'edge':  edge attributes
            'circle': circle attributes
            
    '''
    # get the shape attributes
    gray_img = cv2.cvtColor( object_img, cv2.COLOR_BGR2GRAY )
    binary_img = binarize( gray_img, thresh_val = threshold )
    lbl_img = label( binary_img )
    
    shape_attribute_list = get_attribute( lbl_img )
    
    # get the edge attributes
    edge_img = detect_edges( gray_img, sigma = 1, threshold = 18, lo_thresh = 8 , nms = True )
    edge_attribute_list = get_edge_attribute( lbl_img, edge_img )
    
    # get the circle attributes
    circle_attribute_list = get_circle_attribute( lbl_img, edge_img )
    
    # combine the attribute list
    attribute_list = combine_attribute_lists( shape_attribute_list,
                                             edge_attribute_list,
                                             circle_attribute_list )
    
    return attribute_list
    
# get_all_attributes


def pad_img( img, thickness, pad_val ):
    ''' Helper function to border the image '''
    if img.ndim == 3:  # BGR image
        padding = ( ( thickness, thickness ), ( thickness, thickness ), ( 0, 0 ) )
        
    else:
        padding = thickness
        
    # else    
    pad_img = np.pad( img, padding, constant_values = pad_val )
    
    return pad_img
    
# pad_img


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
    # get the attributes of the test_object
    test_attributes = get_all_attributes( test_object, 128 ) 
    
    print( 'test object attributes' )
    for lbl, attribute in enumerate( test_attributes ):
        print( 'label:', lbl )
        print( 'shape:', attribute['shape'] )
        print( 'edge ({}):'.format( len( attribute['edge'] ) ), attribute['edge'] )
        print( 'circle ({}):'.format( len( attribute['circle'] ) ), attribute['circle'] )
        print()
        
    # for
    print( 75 * '=', end = '\n\n' )

    # gather the attributes of the training images
    for obj in  object_database :
        print( obj['name'] )
        
        attribute_list = get_all_attributes( obj['image'], 128 )
        # print out all of the attributes
        for lbl, attribute in enumerate( attribute_list ):
            print( 'label:', lbl )
            print( 'shape:', attribute['shape'] )
            print( 'edge ({}):'.format( len( attribute['edge'] ) ), attribute['edge'] )
            print( 'circle ({}):'.format( len( attribute['circle'] ) ), attribute['circle'] )
            print()
            
        # for
        
        print( 75 * '=', end = '\n\n' )

        # add the attributes to the database
        obj['attributes'] = attribute_list[0]
    
    # for
    
    # gather the scores over objects in the image
    object_names = []
    for test_obj_attr in test_attributes:
        db_score = [0] * len( object_database )  # initialize scores
        
        # compute the scores over the whole db
        for i, templ_obj in enumerate( object_database ):
            db_score[i] = compare_image_to_template( test_obj_attr, templ_obj['attributes'] )
            
        # for
        
        best_idx = np.argmax( db_score )  # location of best score
        best_obj_name = object_database[best_idx]['name'].replace( '.png', '' )
        
        # format the name
        best_obj_name = ''.join( [l for l in best_obj_name if l.isalpha()] )  # remove numerics
        
        object_names.append( best_obj_name )
        
    # for
    
    return object_names

# best_match


def main( argv ):
    img_name = argv[0]
    test_img = cv2.imread( 'test/' + img_name + '.png', cv2.IMREAD_COLOR )
    
    train_im_names = os.listdir( 'train/' )
    object_database = []
    for train_im_name in train_im_names:
        train_im = cv2.imread( 'train/' + train_im_name )
        train_im = pad_img( train_im, 5, 0 )
        object_database.append( {'name': train_im_name, 'image':train_im} )
    object_names = best_match( object_database, test_img )
    print( object_names )

# main


if __name__ == '__main__':
    main( sys.argv[1:] )
    
# if

# example usage: python p3.py many_objects_1.png

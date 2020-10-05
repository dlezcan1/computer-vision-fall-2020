#!/usr/bin/env python3
import cv2
import numpy as np
import sys
from scipy.ndimage import gaussian_filter


def binarize( gray_image, thresh_val ):
    """ Function to threshold grayscale image to binary
        Sets all pixels lower than threshold to 0, else 255

        Args:
        - gray_image: grayscale image as an array
        - thresh_val: threshold value to compare brightness with

        Return:
        - binary_image: the thresholded image
    """
    binary_image = 255 * ( gray_image >= thresh_val ).astype( np.uint8 )
    
    return binary_image

# binarize


def label( binary_image ):
    """ Function to labeled components in a binary image
        Uses a sequential labeling algorithm

        Args:
        - binary_image: binary image with multiple components to label

        Return:
        - lab_im: binary image with grayscale level as label of component
    """

    _, lab_im = cv2.connectedComponents( binary_image )
    return lab_im

# label


def measure_com( binary_image ):
    """ Helper function to get the center of mass of a binary image
    
        Args:
            - binary_image: binary image array 
            
        Return:
            - com: a dict of 'x' and 'y' for the center of mass values (int)
            
    """
    x = np.arange( binary_image.shape[1] )
    y = np.arange( binary_image.shape[0] )
    
    X, Y = np.meshgrid( x, y )
       
    com = {}
    
    mass = binary_image.sum()
    com['x'] = int( ( X * binary_image ).sum() / mass )
    com['y'] = int( ( Y * binary_image ).sum() / mass )
    
    return com

# measure_com


def measure_second_moment( binary_image, com ):
    """ Helper function to get the center of mass of a binary image
    
        a -> I_xx
        b -> I_xy
        c -> I_yy
    
        Args:
            - binary_image: binary image array 
            - com: the center of mass position dict of 'x' and 'y'
            
        Return:
            - (orientation, roundedness) of the body
            
    """
    # ensure binary_image is max value of one
    binary_img = ( binary_image > 0 ).astype( np.uint8 )
    
    # extract com
    x_com = com['x']
    y_com = com['y']
    
    # get the X and Y grid
    x_range = np.arange( binary_img.shape[1] )
    y_range = np.arange( binary_img.shape[0] )
    X, Y = np.meshgrid( x_range, y_range )
    
    # push the center of mass to the origin
    Xp = X - x_com  # x' = x - x_com
    Yp = Y - y_com  # y' = y - y_com
    
    # calculate 2nd moment parameters
    I_xx = ( ( Xp ** 2 ) * binary_img ).sum()
    I_xy = 2 * ( Xp * Yp * binary_img ).sum()
    I_yy = ( ( Yp ** 2 ) * binary_img ).sum()
    
    # calculate orientation
    theta_1 = np.arctan2( I_xy, I_xx - I_yy ) / 2  # orientation
    theta_2 = theta_1 + np.pi / 2
    
    # calculate roundedness
    E = lambda th: I_xx * np.sin( th ) ** 2 - I_xy * np.cos( th ) * np.sin( th ) + I_yy * np.cos( th ) ** 2
    
    roundedness = E( theta_1 ) / E( theta_2 )  # E_min/E_max
    
    return ( theta_1, roundedness )
    
# measure_second_moment


def get_attribute( labeled_image ):
    """ Function to get the attributes of each component of the image
        Calculates the position, orientation, and roundedness

        Args:
        - labeled_image: image file with labeled components

        Return:
        - attribute_list: a list of the aforementioned attributes
    """
    # get the object list
    objects = np.unique( labeled_image[labeled_image > 0] )  # remove the bg
    
    # iterate through the objects to get their parameters
    attribute_list = [None] * len( objects )
    for obj in objects:
        # attribute dict
        attribute_obj = {}
        
        # segment out the image
        img_mask = ( labeled_image == obj ).astype( np.uint8 )
        
        # get the COM position attribute
        com = measure_com( img_mask )
        attribute_obj['position'] = com
        
        # get the orientation of the body attribute
        orient, rounded = measure_second_moment( img_mask, com )
        attribute_obj['orientation'] = orient
        attribute_obj['roundedness'] = rounded
                
        # add the attribute to the list
        attribute_list[obj - 1] = attribute_obj
        
    # for
    return attribute_list

# get_attribute


def draw_attributes( image, attribute_list ):
    attributed_image = image.copy()
    for attribute in attribute_list:
        center_x = ( int )( attribute["position"]["x"] )
        center_y = ( int )( attribute["position"]["y"] )
        slope = np.tan( attribute["orientation"] )

        cv2.circle( attributed_image, ( center_x, center_y ), 2, ( 0, 255, 0 ), 2 )
        cv2.line( 
            attributed_image,
            ( center_x, center_y ),
            ( center_x + 20, int( 20 * ( -slope ) + center_y ) ),
            ( 0, 255, 0 ),
            2,
        )
        cv2.line( 
            attributed_image,
            ( center_x, center_y ),
            ( center_x - 20, int( -20 * ( -slope ) + center_y ) ),
            ( 0, 255, 0 ),
            2,
        )
    return attributed_image

# draw_attributes


def detect_edges( image, sigma, threshold, lo_thresh = np.inf ):
    """Find edge points in a grayscale image.
    
    Args:
        - image (2D uint8 array): A grayscale image.
        - sigma: the sigma value for a gaussian derivative convolution
        - threshold: the threshold value for determining edges
        - lo_thresh (default None): the potential edge threshold for hysteresis 
                                    thresholding
    Return:
      - edge_image (2D binary image): each location indicates whether it belongs to an edge or not
    """

    # get the derivative of the images    
    d_image_x = gaussian_filter( image, ( 0, sigma ), order = 1 )
    d_image_y = gaussian_filter( image, ( sigma, 0 ), order = 1 )
    d_image_mag = np.sqrt( d_image_x ** 2 + d_image_y ** 2 )
    
    # perform non-maximal suppression
    # # get the gradient raw directions
    d_image_dir = np.rad2deg( np.arctan2( d_image_y, d_image_x ) )
    d_image_dir_binned = np.zeros_like( d_image_dir, dtype = int )
    
    angles = list( range( 0, 180, 45 ) )
    for i, angle in enumerate( angles ):
        # bin the angles for the positive and negative directions
        mask_pos = ( angle - 22.5 <= d_image_dir ) & ( d_image_dir < angle + 22.5 )
        mask_neg = ( angle + 180 - 22.5 <= d_image_dir ) & ( d_image_dir < angle + 180 + 22.5 )
        mask = mask_pos | mask_neg
        
        # bin the angles [0 -> 0, 45 -> 1, 90 -> 2, 135 -> 3]
        d_image_dir_binned[mask] = i
        
    # for
    
    # # perform the non-maximal suppresion using for loops
    for i in range( 1, d_image_mag.shape[0] - 1 ):
        for j in range( 1, d_image_mag.shape[1] - 1 ):
            # gradient direction (binned)
            angle = angles[d_image_dir_binned[i, j]]
            
            if angle == 0:
                check_values = d_image_mag[i, j - 1:j + 2]
            
            # if    
            
            elif angle == 45:
                check_values = [d_image_mag[i - 1, j - 1], d_image_mag[i, j], d_image_mag[i + 1, j + 1]]
                
            # elif
            
            elif angle == 90:
                check_values = d_image_mag[i - 1:i + 2, j]
                
            # elif
            
            else:  # angle == 135
                check_values = [d_image_mag[i + 1, j - 1], d_image_mag[i, j], d_image_mag[i - 1, j + 1]]
                
            # else
         
            # suppress non-maximums
            if not ( np.max( check_values ) == d_image_mag[i, j] ):
                d_image_mag[i, j] = 0
                
            # if
            
        # for
    # for
            
    # hysteresis thresholding
    # # get the definitive edges
    edge_image = ( d_image_mag >= threshold ).astype( np.uint8 )
    
    if lo_thresh < threshold:  # hysteresis thresholding (only need if lo_thresh < threshold)
        # find the potential edges
        edge_image += ( d_image_mag >= lo_thresh ).astype( np.uint8 )

        # at this point, potential matches should be 1 and definitive matches should be 2
        for i in range( edge_image.shape[0] ):
            for j in range( edge_image.shape[1] ):
                # find the indices for the neighborhood array
                lo_i = i - 1 if i > 0 else 0 
                hi_i = i + 2 if i < edge_image.shape[0] - 1 else None  # on the far edge
                lo_j = j - 1 if j > 0 else 0
                hi_j = j + 2 if j < edge_image.shape[1] - 1 else None  # on the far edge
                
                neighbors = edge_image[lo_i:hi_i, lo_j:hi_j]
                
                # check if there is a definitive edge around
                if np.count_nonzero( neighbors > 1 ) == 0:  # if there are no definitive edges
                    edge_image[i, j] = 0  # set this as not an edge
                
                # if
            # for
        # for
    # if
    
    # post processing for binarization of edges
    edge_image[edge_image.nonzero()] = 255
    
    return edge_image
        
# detect_edges


def get_edge_attribute( labeled_image, edge_image ):
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
    # get the labels
    labels = np.unique( labeled_image[labeled_image.nonzero()] )
    
    # set-up
    x_range = np.arange( edge_image.shape[1] )
    y_range = np.arange( edge_image.shape[0] )
    X, Y = np.meshgrid( x_range, y_range )
    
    # get the line attributes
    attribute_list = [None] * len( labels )
    for lbl in labels:
        # segment out the edge image
        lbl_edges = 255 * np.logical_and( labeled_image == lbl, edge_image ).astype( np.uint8 )
        
        # get the Hough Transform for lines
        lines = cv2.HoughLines( lbl_edges, 2, np.pi / 36, 45 )
        if isinstance( lines, type( None ) ):
            continue
 
        # if
                    
        lines = lines[:, 0, :]  # drop unnecessary dimension
        
        # determine the length of lines 
        dist_thresh = 2  # (px) the distance threshold for counting length of lines
        
        # iterate over all of the lines associated with this label
        lbl_attr_list = []
        for rho, theta in lines:
            line_dict = {'angle': theta, 'distance': rho}
            # calculate the distance from the line
            line_dist = np.abs( np.cos( theta ) * X + np.sin( theta ) * Y - rho )
            
            # find all of the overlaps on the edges w/in a threshold
            pixels_on_line = ( lbl_edges ) & ( line_dist <= dist_thresh )
            
            # find line continuity ( not implemented )
            
            # find length of line
            line_dict['length'] = np.count_nonzero( pixels_on_line )
            
            # append the dict to the label list
            lbl_attr_list.append( line_dict )
            
        # for
        
        # Add the attribute
        attribute_list[lbl - 1] = lbl_attr_list
        
    # for
    
    return attribute_list

# get_edge_attribute


def draw_edge_attributes( image, attribute_list ):
    attributed_image = image.copy()
    for lines in attribute_list:
        if not lines:
            continue
        
        for line in lines:
            angle = ( float )( line["angle"] )
            distance = ( float )( line["distance"] )

            a = np.cos( angle )
            b = np.sin( angle )
            x0 = a * distance
            y0 = b * distance
            pt1 = ( int( x0 + 1000 * ( -b ) ), int( y0 + 1000 * ( a ) ) )
            pt2 = ( int( x0 - 1000 * ( -b ) ), int( y0 - 1000 * ( a ) ) )

            cv2.line( 
                attributed_image,
                pt1,
                pt2,
                ( 0, 255, 0 ),
                2,
            )

    return attributed_image

# draw_edge_attributes


def get_circle_attribute( labeled_image, edge_image ):
    '''
    Function to get the attributes of each circle of the image
          Calculates the center and radius in pixels (non-rounded)
    Args:
      labeled_image: binary image with grayscale level as label of component
      edge_image (2D binary image): each location indicates whether it belongs to an edge or not
    
    Returns:
       attribute_list: a list of list [[dict()]]. For example, [lines1, lines2,...],
       where lines1 is a list and it contains lines for the first object of attribute_list in part 1.
       Each item of lines 1 is a circle, i.e., a dictionary containing keys with radius, position,
           where position is a dict of keys 'x' and 'y'.
       You should associate objects in part 1 and lines in part 2 by putting the attribute lists in same order.
       Note that votes in HoughLines opencv-python is not longer available since 2015. You will need to compute the length yourself.
       Will return None for an image if no circles were detected
    '''
        
    # extra credits
    
    # get the labels
    labels = np.unique( labeled_image[labeled_image.nonzero()] )
    
    # determine all the circles in the image
    attribute_list = [None] * len( labels )
    for lbl in labels:
        # segment out the edge image
        lbl_edges = 255 * np.logical_and( labeled_image == lbl, edge_image ).astype( np.uint8 )
        
        # perform the hough transform
        circles = cv2.HoughCircles( lbl_edges, cv2.HOUGH_GRADIENT, 1, 30,
                                   param1 = 50, param2 = 30,
                                   minRadius = 0, maxRadius = 0 )
        if isinstance( circles, type( None ) ):
            continue
        
        # if
             
        circles = circles[:, 0, :]  # drop unnecessary dimenstion
        
        # determine the length of lines 
        dist_thresh = 2  # (px) the distance threshold for counting length of lines
        
        # iterate over all of the lines associated with this label
        lbl_attr_list = []
        for x0, y0, R in circles:
            circle_dict = {'radius': R}
            
            # append the position (center point
            circle_dict['position'] = {'x': x0, 'y': y0}
            
            # append the dict to the label list
            lbl_attr_list.append( circle_dict )
            
        # for
        
        # Add the attribute
        attribute_list[lbl - 1] = lbl_attr_list
        
    # for
    
    return attribute_list

# get_circle_attributes


def main( argv ):
    img_name = argv[0]
    thresh_val = int( argv[1] )
    
    img = cv2.imread( 'data/' + img_name + '.png', cv2.IMREAD_COLOR )
    
    gray_image = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    cv2.imwrite( 'output/' + img_name + "_gray.png", gray_image )
    
    # part 1
    binary_image = binarize( gray_image, thresh_val = thresh_val )
    cv2.imwrite( 'output/' + img_name + "_binary.png", binary_image )
    
    labeled_image = label( binary_image )
    cv2.imwrite( 'output/' + img_name + "_labeled.png", 255 / np.max( labeled_image ) * labeled_image )
    
    attribute_list = get_attribute( labeled_image )
    print( 'attribute list:' )
    print( attribute_list )
    
    attributed_image = draw_attributes( img, attribute_list )
    cv2.imwrite( "output/" + img_name + "_attributes.png", attributed_image )
    
    # part 2
    # feel free to tune hyperparameters or use double-threshold
    edge_image = detect_edges( gray_image, sigma = 1, threshold = 10, lo_thresh = 4 )
    cv2.imwrite( "output/" + img_name + "_edges.png", edge_image )
    
    edge_attribute_list = get_edge_attribute( labeled_image, edge_image )
    print( 'edge attribute list:' )
    print( edge_attribute_list )
    
    attributed_edge_image = draw_edge_attributes( img, edge_attribute_list )
    cv2.imwrite( "output/" + img_name + "_edge_attributes.png", attributed_edge_image )
    
    # extra credits for part 2: show your circle attributes and plot circles
    # circle_attribute_list = get_circle_attribute(labeled_image, edge_image)
    # attributed_circle_image = draw_circle_attributes(img, circle_attribute_list)
    # cv2.imwrite("output/" + img_name + "_circle_attributes.png", attributed_circle_image)
    
# main


def debug( argv ):
    img_name = argv[0]
    thresh_val = int( argv[1] )
    img = cv2.imread( 'data/' + img_name + '.png', cv2.IMREAD_COLOR )

    # debug testing
    bool_binarize = True
    bool_label = bool_binarize and True
    bool_attribute = bool_label and True
    bool_shape = True
    bool_edge = True
    bool_edge_attribute = bool_label and bool_edge and True
    
    # grayscale the image
    gray_image = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    print( 'Image Shape:', gray_image.shape )
    cv2.imshow( 'gray image', img )
    
    # test binarize
    if bool_binarize:
        binary_img = binarize( gray_image, thresh_val )
        cv2.imshow( 'binary', binary_img )
    
    # if
    
    # test the labeling
    if bool_binarize and bool_label:
        lbl_img = label( binary_img )  # this is 2-D arr of class values of integers
    
    # if
    
    # test the edge
    if bool_shape:
        # # rectangle
        box_img = np.zeros_like( gray_image )
        ( y1, x1 ) = tuple( np.array( box_img.shape ) // 6 )
        ( y2, x2 ) = tuple( 2 * np.array( box_img.shape ) // 6 )
        cv2.rectangle( box_img, ( x1, y1 ), ( x2, y2 ), [255, 255, 255], 10 )
        np.random.seed( 12 )
        salt_pepper = ( np.random.randint( 0, 100, box_img.shape ) <= 10 )
        box_img[np.logical_and( salt_pepper, box_img > 0 )] = 0
        lbl_box_img = ( box_img > 0 ).astype( int )
        cv2.imshow( 'rect', box_img )
    
        # ## test edge attributes
        print( 'box' )
        edge_attributes = get_edge_attribute( lbl_box_img, box_img )
        print()

    # if
    
    # # on The original image
    # ## test edge detection
    if bool_edge:
        edge_img = detect_edges( gray_image, sigma = 1, threshold = 8, lo_thresh = 4 )
        cv2.imshow( 'edge image', edge_img )
        
    # if
    
    # ## test edge attributes
    if bool_edge_attribute:
        print( 'Edge Attributes (OG Image)' )
        edge_attribute_list = get_edge_attribute( lbl_img, edge_img )
        print( 0, len( edge_attribute_list[0] ) )
        print( 1, len( edge_attribute_list[1] ) )
        print( edge_attribute_list )
        print()
        
        attributed_edge_image = draw_edge_attributes( img, edge_attribute_list )
        cv2.imshow( 'drawn edged img', attributed_edge_image )
        
    # if
        
    # get the attribute list
    if bool_attribute:
        attribute_list = get_attribute( lbl_img )
        print( 'attributes (OG Image)' )
        print( attribute_list )
        com_img = np.expand_dims( binary_img, axis = 2 ).repeat( 3, axis = 2 )
        for i, attr in enumerate( attribute_list ):
            x = attr['position']['x']
            y = attr['position']['y']
            if i == 0:
                cv2.circle( com_img, ( x, y ), 5, [0, 0, 255], -1 )
                
            else:
                cv2.circle( com_img, ( x, y ), 5, [0, 255, 0], -1 )
            
        # for
        
        cv2.imshow( 'com', com_img )
        attributed_image = draw_attributes( img, attribute_list )
        cv2.imshow( 'attributed image', attributed_image )
        
    # if
    
    if bool_shape:
        # test orientation and roundedness
        shape = np.array( [100, 100] )
        
        # # circle
        test_circle = np.zeros( shape, np.uint8 )
        test_circle = cv2.circle( test_circle, ( 30, 40 ), 20, [255, 255, 255], -1 )
    #     cv2.imshow( 'circle', test_circle )
        com_circle = measure_com( test_circle )
        print( 'circle' )
        print( 'hough circles', cv2.HoughCircles( test_circle, cv2.HOUGH_GRADIENT, 1, 20,
                                                param1 = 30, param2 = 15,
                                                minRadius = 0, maxRadius = 0 ) )
        print( 'com: ', ( com_circle['x'], com_circle['y'] ) )
        print( 'orient, rounded: ', measure_second_moment( test_circle, com_circle ) )
        print()
        
        # # ellipse
        test_ellipse = np.zeros( shape, np.uint8 )
        test_ellipse = cv2.ellipse( test_ellipse, ( 30, 40 ), ( 20, 15 ), 45, 0, 360, [255, 255, 255], -1 )
    #     cv2.imshow( 'ellipse', test_ellipse )
        com_ellipse = measure_com( test_ellipse )
        print( 'ellipse' )
        print( 'com: ', ( com_ellipse['x'], com_ellipse['y'] ) )
        print( 'orient actual: ', np.deg2rad( 45 ) )
        print( 'orient, rounded: ', measure_second_moment( test_ellipse, com_ellipse ) )  # differs because it takes smaller axis
        
    # if

    # show the results
    cv2.waitKey( 0 )
    
# debug

    
if __name__ == '__main__':
    main( sys.argv[1:] )
    # example usage: python p1n2.py two_objects 128
    # expected results can be seen here: https://hackmd.io/toS9iEujTtG2rPoxAdPk8A?view
    
    # module testing
#     debug( sys.argv[1:] )
    
# if
    

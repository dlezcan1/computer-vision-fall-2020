#!/usr/bin/env python3
import cv2
import numpy as np
import sys
from scipy.signal import convolve2d


def normxcorr2( template, image ):
    """Do normalized cross-correlation on grayscale images.
    
    When dealing with image boundaries, the "valid" style is used. Calculation
    is performed only at locations where the template is fully inside the search
    image.
    
    Heat maps are measured at the top left of the template.
    
    Args:
    - template (2D float array): Grayscale template image.
    - image (2D float array): Grayscale search image.
    
    Return:
    - scores (2D float array): Heat map of matching scores.
    """
    # helper variables
    t_0, t_1 = template.shape
    template_norm = template / np.linalg.norm( template )
    
    # begin norm xcorr 2-D
    retval = np.zeros( ( image.shape[0] - t_0, image.shape[1] - t_1 ) )
    for i in range( retval.shape[0] ):
        for j in range( retval.shape[1] ):
            # get the sub image for this point
            sub_img = image[i:i + t_0, j:j + t_1]
            sub_img = sub_img / np.linalg.norm( sub_img )
            
            retval[i, j] = ( sub_img * template_norm ).sum() 
            
        # for
    # for
    
    return retval
    
# normxcorr2


def find_matches( template, image, thresh = None ):
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
    # compute the correlation heat map
    template_gray = cv2.cvtColor( template, cv2.COLOR_BGR2GRAY )  # grayscale
    image_gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )  # grayscale
    heat_map = normxcorr2( template_gray, image_gray )
    
    # copy image for mark up
    match_image = image.copy()
    
    # find coordaintes and mark up match_image
    if thresh is None:
        coords = np.unravel_index( np.argmax( heat_map ), heat_map.shape )
        end_coords = tuple( np.array( coords ) + np.array( template.shape[0:2] ) )
        
        # flip the coords
        coords = tuple( reversed( coords ) )
        end_coords = tuple( reversed( end_coords ) )
        
        # draw the template matched
        cv2.rectangle( match_image, coords, end_coords, [0, 255, 0], 2 )
        
    # if
        
    else:
        I, J = np.nonzero( heat_map >= thresh )  # find threshold coordinates
        
        coords = list( zip( J, I ) )  # turn those coordinates to a list of tuples
        
        # draw the template matched
        end_coord_adjust = np.array( template.shape[1::-1] )
        for coord in coords:
            end_coord = tuple( np.array( coord ) + end_coord_adjust )
            
            cv2.rectangle( match_image, coord, end_coord, [0, 255, 0], 2 )
            
        # for
        
    # else
    
    return coords, match_image

# find_matches


def main( argv ):
    template_img_name = argv[0]
    search_img_name = argv[1]
    
    template_img = cv2.imread( "data/" + template_img_name + ".png", cv2.IMREAD_COLOR )
    search_img = cv2.imread( "data/" + search_img_name + ".png", cv2.IMREAD_COLOR )
    
    if search_img_name == 'king':  # best match for king
        _, match_image = find_matches( template_img, search_img, None )
        
    else:
        _, match_image = find_matches( template_img, search_img, 0.95 )
    
    cv2.imwrite( "output/" + search_img_name + ".png", match_image )


if __name__ == "__main__":
    main( sys.argv[1:] )

# example usage: python p4.py face king
# expected results can be seen here: https://hackmd.io/toS9iEujTtG2rPoxAdPk8A?view

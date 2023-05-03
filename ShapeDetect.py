import cv2
from ImageProcess import *

def shape_detect(img):
    thresh_image = threshold_otsu(img)
    
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        perim = cv2.arcLength(contour, True)
        
        appr = cv2.approxPolyDP(contour, 0.01 * perim, True)
        vertices =  len(appr)
        
        if vertices == 3:
            shape_name = 'Triangle'
        elif vertices == 4:
            _, _, w, h = cv2.boundingRect(appr)
            aspect = w/float(h)
            if aspect >= 0.95 and aspect <= 1.05:
                shape_name = 'Square'
            else:
                shape_name = 'Rectangle'
        elif vertices == 5:
            shape_name = 'Pentagon'
        elif vertices == 6:
            shape_name = 'Hexagon'
        elif vertices == 7:
            shape_name= 'Heptagon'
        elif vertices == 8:
            shape_name= 'Oktagon'
        elif vertices == 9:
            shape_name = 'Nonagon'
        elif vertices == 10:
            shape_name = 'Polygon'
        elif vertices >= 11:
            shape_name = 'Circle'
        else:
            shape_name= 'Unknown'
        
        
        cv2.drawContours(img, [appr], 0, (0,255,0), 2)
        cv2.putText(img, shape_name, (appr.ravel()[0], appr.ravel()[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return img
                
        
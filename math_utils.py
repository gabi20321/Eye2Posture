import math
import numpy as np
import cv2

MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),             
    (0.0, -330.0, -65.0),        
    (-225.0, 170.0, -135.0),     
    (225.0, 170.0, -135.0),      
    (-150.0, -150.0, -125.0),    
    (150.0, -150.0, -125.0)      
], dtype=np.float64)


def eyePos(eyeCoord):
    vertDist1 = math.dist(eyeCoord[2],eyeCoord[5])
    vertDist2 = math.dist(eyeCoord[3],eyeCoord[4])
    oriDist = math.dist(eyeCoord[0],eyeCoord[1])
    eyeDist = (vertDist1 + vertDist2)/(2*oriDist)
    return eyeDist
    
def headPos(faceCord,img_w,img_h):
    imgPoints = np.array(faceCord, dtype=np.float64)
    
    focal_length = img_w
    center_x = img_w / 2
    center_y = img_h / 2
    
    matrix_camera = np.array([
        [focal_length,0,center_x],
        [0,focal_length,center_y],
        [0,0,1]
    ], dtype=np.float64)
    
    dist_coef = np.zeros((4,1))
    
    succes, rotation_vector,translation_vector = cv2.solvePnP(
        MODEL_3D_POINTS,
        imgPoints,
        matrix_camera,
        dist_coef
    )
    
    rmat, _ = cv2.Rodrigues(rotation_vector)
    
    angle, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    pitch = angle[0]
    yaw = angle[1]
    
    return pitch, yaw

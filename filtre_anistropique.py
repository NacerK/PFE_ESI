import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
from lecture_image import read_image
from affichage_image import affichage

def anistropique(src, K, niters, alpha):
    dst	=	cv.ximgproc.anisotropicDiffusion(src, alpha, K, niters)
    image= cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    return dst



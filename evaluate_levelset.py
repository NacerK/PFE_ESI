import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.ndimage.filters as filters
from skimage import metrics
import FCM_fuzz as FCM
import Levelset
import Eval
import numpy as np
import cv2
from PIL import Image
import glob
import time

image_list = []
for filename in glob.glob('dataset/*flair.png'):
    im=Image.open(filename)
    im=np.asarray(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    image_list.append(im)

groundtruth_list =[]


for filename in glob.glob('dataset/*seg.png'):
    im = Image.open(filename)
    im = np.asarray(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    groundtruth_list.append(im)

results=[]
time_start=0
time_seg=0
file = open('evaluation_results.txt', 'w')

for n in range (len(image_list)):

    time_start = time.time()

    #Charger l'image et son ground truth
    print("Chargement de l'image ", n)
    image= image_list[n]
    groundtruth=groundtruth_list[n]


    # Pre-traitement de l'image
    ker_dil = 3
    iter_dil = 3
    gauss = 2

    # Reduction du bruit
    img_prep = filters.gaussian_filter(image, gauss)

    # Dilatation

    kernel = np.ones((ker_dil, ker_dil), np.uint8)
    img_prep = cv2.dilate(img_prep, kernel, iterations=iter_dil)


    # Initialisation FCM
    number_of_cluster = 3
    fuzzy_coef = 2
    seuil = 0.1
    max_iter = 100

    # Execution FCM
    image_fcm, phi_0 = FCM.fcm_execute(img_prep, number_of_cluster, fuzzy_coef, seuil, max_iter)



    #=======LEVELSET SECTION==========================================

    # Boost Contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_prep2 = clahe.apply(image)

    # Initialiation des parametres du Levelset

    timestep = -1  # time step
    mu = 0.2 / timestep  # coefficient of the distance regularization term R(phi)
    # timestep*mu must be < 1/4 (CourantFriedrichs-Lewy (CFL) condition)
    iter_inner = 4
    iter_outer = 100   #Iterations max
    lmda = 2  # coefficient of the weighted length term L(phi)
    alfa = 4  # coefficient of the weighted area term A(phi) -- SHOULD BE POSITIF IF THE CONTOUR MOVES INSIDE /
    # NEGATIF IF THE CONTOUR MOVES OUTSIDE
    # SMALL IF WEAK CONTOUR
    epsilon = 2 # parameter that specifies the width of the DiracDelta function
    errortol = 0.00002   # Stop the evolution if (phi_n)-(phi_n-1) < errortol
    window=0
    # Execution de l'algorithme Levelset
    phi_final = Levelset.levelset_execute(img_prep2, phi_0, timestep, mu, iter_inner, iter_outer, lmda, alfa,
                                          epsilon, errortol)

    #Temps de l'operation complète de segmentation (  pre-traitement + classification + segmentation )
    time_seg = time.time() - time_start


    #====== Partie evaluation du résultat ============================================
    # Binariser la segmentation: Zone en dehors du contour = 0, zone à l'interieur du contour = 1
    x_phi, y_phi = phi_final.shape
    for i in range(x_phi):
        for j in range(y_phi):
            if phi_final[i, j] < 0:
                phi_final[i, j] = 0
            else:
                phi_final[i, j] = 1

    # Evaluation avec la segmentation groundtruth (IntersectionOverUnion et DICE)

    # Binarisation de la segmentation groundtruth
    ret, groundtruth_thresh = cv2.threshold(groundtruth, 0, 1, cv2.THRESH_BINARY)
    # Calcule de l'IOU et du DICE
    iou, dice = Eval.iou_dice(phi_final, groundtruth_thresh)
    # Calcule de la Mean Squared Error
    mse = metrics.mean_squared_error(phi_final, groundtruth_thresh)
    print("IOU= ", iou, " DICE= ", dice, " MSE= ", mse)
    results.append({"Image": n , "IOU" : iou , "DICE" : dice , "MSE" : mse , "Time" : time_seg} )
    file.write(str(results[n])+"\n")



file.close()
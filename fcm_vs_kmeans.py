import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.ndimage.filters as filters
from skimage import metrics
import FCM_fuzz as FCM
import Levelset
import Kmeans
import Eval
import numpy as np
import cv2
from PIL import Image
import glob
import time


#Chargement des images du dataset
image_list = []
for filename in glob.glob('dataset/*flair.png'):
    im=Image.open(filename)
    im=np.asarray(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    image_list.append(im)

#Chargement des segmentations d'expert (groundtruth) du dataset
groundtruth_list =[]
for filename in glob.glob('dataset/*seg.png'):
    im = Image.open(filename)
    im = np.asarray(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    groundtruth_list.append(im)

#Variables de calcul des métriques
n_images=len(image_list)
results=[]
time_start= 0
time_seg= 0
sum_iou_fcm = 0
sum_dice_fcm = 0
sum_time_fcm = 0
sum_iou_kmeans = 0
sum_dice_kmeans = 0
sum_time_kmeans = 0


#Lecture du fichier qui contiendra les résultats
file = open('fcm_vs_kmeans.txt', 'w')


#Parcourer l'ensemble du dataset
for n in range(0, n_images):

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

    time_start = time.time()
    # Initialisation FCM
    number_of_cluster = 3
    fuzzy_coef = 2
    seuil = 0.1
    max_iter = 100

    # Execution FCM
    image_fcm, phi_0 = FCM.fcm_execute(img_prep, number_of_cluster, fuzzy_coef, seuil, max_iter)

    #Temps FCM
    time_seg = time.time() - time_start


    # Binariser la segmentation: Zone en dehors du contour = 0, zone à l'interieur du contour = 1
    x_phi, y_phi = phi_0.shape
    for i in range(x_phi):
        for j in range(y_phi):
            if phi_0[i, j] < 0:
                phi_0[i, j] = 0
            else:
                phi_0[i, j] = 1

    # Binarisation de la segmentation groundtruth
    ret, groundtruth_thresh = cv2.threshold(groundtruth, 0, 1, cv2.THRESH_BINARY)

    # Calcule de l'IOU et du DICE
    iou, dice = Eval.iou_dice(phi_0, groundtruth_thresh)

    #Sommes pour calculer les moyennes plutard
    sum_iou_fcm +=iou
    sum_dice_fcm += dice
    sum_time_fcm += time_seg

    #Affichage et sauvegarde des résultats
    print("FCM_Image", n, "IOU", iou, "DICE", dice, "Time", time_seg)
    results.append({"FCM_Image": n , "IOU" : iou , "DICE" : dice , "Time" : time_seg} )


    #=======Partie Kmeans =============================================================

    time_start = time.time()

    #Parametres Kmeans
    nombre_cluster = 5
    eps = 0.1
    nombre_itteration = 100
    image_kmeans, phi_0 = Kmeans.kmeans_execute(nombre_itteration, eps, nombre_cluster, img_prep)

    time_seg = time.time() - time_start

    # Binariser la segmentation: Zone en dehors du contour = 0, zone à l'interieur du contour = 1
    x_phi, y_phi = phi_0.shape
    for i in range(x_phi):
        for j in range(y_phi):
            if phi_0[i, j] < 0:
                phi_0[i, j] = 0
            else:
                phi_0[i, j] = 1

    # Calcule de l'IOU et du DICE
    iou, dice = Eval.iou_dice(phi_0, groundtruth_thresh)

    #Sommes pour calculer les moyennes plutard
    sum_iou_kmeans +=iou
    sum_dice_kmeans += dice
    sum_time_kmeans += time_seg

    # Affichage et sauvegarde des résultats
    print("K_means_Image", n, "IOU", iou, "DICE", dice, "Time", time_seg)
    results.append({"K_means_Image": n, "IOU": iou, "DICE": dice, "Time": time_seg})


#Ecriture des résultats dans le fichier
for elem in results:
    file.write(str(elem)+ "\n")

file.write("FCM: Moy IOU=" + str(sum_iou_fcm/n_images) + "Moy DICE=" + str(sum_dice_fcm/n_images) + "Moy Time=" + str(sum_time_fcm/n_images) + "\n")
file.write("Kmeans: Moy IOU=" + str(sum_iou_kmeans/n_images) + "Moy DICE=" + str(sum_dice_kmeans/n_images) + "Moy Time=" + str(sum_time_kmeans/n_images) + "\n")

file.close()

print("Evaluation terminée")

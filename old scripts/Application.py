import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.ndimage.filters as filters
from skimage import metrics
import FCM as FCM
import Levelset
import Kmeans
import Eval
import numpy as np
import cv2
import PySimpleGUI as sg
import os.path
import time
matplotlib.use('TkAgg')


# -------------- GUI---------------------------------
#Fonction d'aide pour l'affichage des images Matplotlib sur la GUI
def draw_figure(canvas_name, image):
    canvas=window[canvas_name].TKCanvas
    fig = matplotlib.figure.Figure(figsize=(4, 3), dpi=100)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.imshow(image, cmap="gray")
    figure_canvas_agg = FigureCanvasTkAgg(fig, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)

    return figure_canvas_agg

sg.theme('LightGrey1')
col_layout= [   [sg.Frame('IRM à scanner',  layout=[
              [sg.Canvas(key='-IMG_GUI-', size=(400, 310),)]])]
                ,
              [sg.Frame("Segmentation d'expert" , layout=[
              [sg.Canvas(key='-GT_GUI-', size=(400, 310),)]])]
              ]
col_layout2= [   [sg.Frame('Classification FCM',  layout=[
              [sg.Canvas(key='-FCM_GUI-', size=(400, 310),)]])]
                ,
              [sg.Frame("Segmentation Level Set" , layout=[
              [sg.Canvas(key='-LS_GUI-', size=(400, 310),)]])]
              ]
layout = [
          [sg.Text("Segmentation de tumeurs par la méthode level sets", font=("Helvetica", 12), size=(48, 1),
                   justification=('center'))],
          [sg.Frame('Initialisation' ,  layout=[

              [sg.Frame('Pre-traitement', layout=[
                  [sg.Text('CLAHE', size=(15, 1)), sg.In(default_text='2', size=(10, 1), key="-CLAHE-")],
                  [sg.Text('Filtre Gaussien', size=(15, 1)), sg.In(default_text='2', size=(10, 1), key="-Gauss-")],
                  [sg.Text('Kernel de dilatation', size=(15, 1)),
                   sg.In(default_text='3', size=(10, 1), key="-KerDil-")],
                  [sg.Text('Iterations de dilatation', size=(15, 1)),
                   sg.In(default_text='3', size=(10, 1), key="-IterDil-")],
              ])],

              [sg.Frame('Parametres FCM', layout=[

                  [sg.Text('Nombre de clusters', size=(15, 1)),
                   sg.In(default_text='3', size=(10, 1), key="-NBCLUSTER-")],
                  [sg.Text('Nombre d\'Iterations maximales', size=(15, 1)),
                   sg.In(default_text='100', size=(10, 1), key="-ITERFCM-")],
                  [sg.Text('Coefficient de flou', size=(15, 1)),
                   sg.In(default_text='2', size=(10, 1), key="-FUZZYCO-")],
                  [sg.Text('Seuil d\'arrêt', size=(15, 1)),
                   sg.In(default_text='0.1', size=(10, 1), key="-SEUILFCM-")]])],

              [sg.Frame('Parametres Levelset', layout=[
                  [sg.Text('Timestep', size=(15, 1)), sg.In(default_text='-1', size=(10, 1), key="-TIMESTEP-")],
                  [sg.Text('Iterations maximale', size=(15, 1)),
                   sg.In(default_text='100', size=(10, 1), key="-ITERLS-")],
                  [sg.Text('Alpha', size=(15, 1)), sg.In(default_text='4', size=(10, 1), key="-ALPHA-")],
                  [sg.Text('Lamda', size=(15, 1)), sg.In(default_text='2', size=(10, 1), key="-LAMDA-")],
                  [sg.Text('Epsilon', size=(15, 1)), sg.In(default_text='2', size=(10, 1), key="-EPSILON-")],
                  [sg.Text('Seuil d\'arrêt', size=(15, 1)),
                   sg.In(default_text='0.00001 ', size=(10, 1), key="-SEUILLS-")],
              ])],

              [sg.Frame('Cible', layout=[
                  [sg.Text('Image à segmenter')],
                  [sg.In(size=(25, 1), enable_events=True, key="-FILE-"),
                   sg.FileBrowse()],
                  [sg.Text('Segmentation expert (Groundtruth)')],
                  [sg.In(size=(25, 1), enable_events=True, key="-FILEGT-"),
                   sg.FileBrowse()],
                  [sg.Button('Lancer', key="-LANCER-"), sg.Button('Quitter')]
              ])],
              [sg.Text("Inserez votre image puis appuyez sur Lancer", size=(33, 2), key="-STATUS-")],
              [sg.Text("Progression", font=("Helvetica", 9)),
               sg.ProgressBar(1, orientation='h', size=(15, 8), key='progress')],

          ]),  sg.Column(col_layout, justification='up'), sg.Column(col_layout2, justification='up')
           ]]





window = sg.Window('Application', layout)
image = np.array([])
while True:  # Event Loop
    event, values = window.read()
    if event == "Quitter":
        break
    # une image a été selectionée
    if event is None or event == "-FILE-":
        filename = values["-FILE-"]

        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #Show the image on GUI

        draw_figure('-IMG_GUI-', image)

    #Un contour d'expert a été séléctioné
    if event == "-FILEGT-":
        filenamegt = values["-FILEGT-"]
        groundtruth = cv2.imread(filenamegt)
        groundtruth = cv2.cvtColor(groundtruth, cv2.COLOR_BGR2GRAY)
        # Show the GT on GUI
        draw_figure('-GT_GUI-', groundtruth)
    # Le programme a été lancé
    if event == "-LANCER-":  #
        if image.size != 0:
            window["-STATUS-"].update('FCM en cours d\'execution..')
            progress_bar = window.FindElement('progress')
            progress_bar.UpdateBar(1, 4)

            #=======PRE-TRAITEMENT===========================================

            # Pre-traitement de l'image
            ker_dil = int(values["-KerDil-"])
            iter_dil = int(values["-IterDil-"])
            gauss = float(values["-Gauss-"])

            # Reduction du bruit
            img_prep = filters.gaussian_filter(image, gauss)

            # Dilatation

            kernel = np.ones((ker_dil, ker_dil), np.uint8)
            img_prep = cv2.dilate(img_prep, kernel, iterations=iter_dil)

            #=========FCM SECTION=============================================

            # Initialisation FCM
            number_of_cluster = int(values["-NBCLUSTER-"])
            fuzzy_coef = float(values["-FUZZYCO-"])
            seuil = float(values["-SEUILFCM-"])
            max_iter = int(values["-ITERFCM-"])

            # Execution FCM
            image_fcm, phi_0 = FCM.fcm_execute(img_prep, number_of_cluster, fuzzy_coef, seuil, max_iter)
            progress_bar.UpdateBar(2, 4)

            window["-STATUS-"].update('FCM terminé, Appuyez sur Lancer pour demmarrer les Levelset')
            #Afficher FCM sur GUI
            draw_figure('-FCM_GUI-', image_fcm)

            #Afficher FCM sur une plot indépendante
            #FCM.affichage(img_prep, image_fcm, phi_0)


            #=======LEVELSET SECTION==========================================

            progress_bar.UpdateBar(3, 4)
            window["-STATUS-"].update('Levelset en cours d\'execution...')

            # Boost Contrast
            clahe = cv2.createCLAHE(clipLimit=float(values["-CLAHE-"]), tileGridSize=(8, 8))
            img_prep2 = clahe.apply(image)

            # Initialiation des parametres du Levelset

            timestep = int(values["-TIMESTEP-"])  # time step
            mu = 0.2 / timestep  # coefficient of the distance regularization term R(phi)
            # timestep*mu must be < 1/4 (CourantFriedrichs-Lewy (CFL) condition)
            iter_inner = 4
            iter_outer = int(values["-ITERLS-"])
            lmda = float(values["-LAMDA-"])  # coefficient of the weighted length term L(phi)
            alfa = float(values[
                             "-ALPHA-"])  # coefficient of the weighted area term A(phi) -- SHOULD BE POSITIF IF THE CONTOUR MOVES INSIDE /
            # NEGATIF IF THE CONTOUR MOVES OUTSIDE
            # SMALL IF WEAK CONTOUR
            epsilon = float(values["-EPSILON-"])  # parameter that specifies the width of the DiracDelta function
            errortol = float(values["-SEUILLS-"])  # Stop the evolution if (phi_n)-(phi_n-1) < errortol

            # Execution de l'algorithme Levelset
            phi_final = Levelset.levelset_execute(img_prep2, phi_0, timestep, mu, iter_inner, iter_outer, lmda, alfa,
                                                  epsilon, errortol,window)

            # Affichage du resultat sur GUI

            plt.ioff()
            draw_figure('-LS_GUI-', phi_final)


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

            progress_bar.UpdateBar(4, 4)
            window["-STATUS-"].update('Levelset terminé!')
            #Affichage final
            Levelset.show_segmentation(img_prep2, phi_final, groundtruth_thresh, iou, dice)

window.close()





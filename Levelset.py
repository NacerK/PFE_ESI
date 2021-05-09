import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage.filters as filters
from skimage import measure
import drlse_algo as drlse
import numpy as np
import cv2

#Affichage de la fonction phi 3D du Level Set
def show_3dphi():
    ax1 = fig1.add_subplot(111, projection='3d')
    y, x = phi.shape
    x = np.arange(0, x, 1)
    y = np.arange(0, y, 1)
    X, Y = np.meshgrid(x, y)
    ax1.plot_surface(X, Y, -phi, rstride=2, cstride=2, color='r', linewidth=0, alpha=0.6, antialiased=True)
    ax1.contour(X, Y, phi, 0, colors='g', linewidths=2)

#Affichage final du levelset
def show_segmentation(img,phi,groundtruth,iou,dice):
    fig = plt.figure(figsize=(8, 6), dpi=120)
    contours = measure.find_contours(phi, 0)


    if groundtruth is not None:
        ax1=fig.add_subplot(1,2,2)
        ax1.imshow(groundtruth, interpolation='nearest', cmap="Reds")
        ax1.set_title("Segmentation d'expert")
        plt.text(240, 60, "IOU= "+str(round(iou,2))+'\nF1 Score= '+str(round(dice,2)), fontdict=None)

        ax2 = fig.add_subplot(1,2,1)
        ax2.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
        ax2.set_title('Segmentation finale du Level Set')
        for n, contour in enumerate(contours):
            ax2.plot(contour[:, 1], contour[:, 0], linewidth=1)
            ax1.plot(contour[:, 1], contour[:, 0], linewidth=1)
    else:
        ax2 = fig.add_subplot(1, 1, 1)
        ax2.set_title('Segmentation finale du Level Set')
        ax2.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
        for n, contour in enumerate(contours):
            ax2.plot(contour[:, 1], contour[:, 0], linewidth=1)

    plt.show()

#Affichage de l'évolution du levelset
def show_evolution(img,phi,figure):
    contours = measure.find_contours(phi, 0)
    ax3 = figure.add_subplot(111)
    ax3.set_title('Evolution de la segmentation par Level Set')
    ax3.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(contours):
        ax3.plot(contour[:, 1], contour[:, 0], linewidth=1)

#Ancienne fonction pour afficher le Levelset sur la GUI PysimpleGUI
def show_evolution_gui(img,phi,window):
    canvas = window['-LS_GUI-'].TKCanvas
    fig = matplotlib.figure.Figure(figsize=(4, 3), dpi=120)
    contours = measure.find_contours(phi, 0)
    ax3 = fig.add_subplot(111)
    ax3.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    for n, contour in enumerate(contours):
        ax3.plot(contour[:, 1], contour[:, 0], linewidth=1)
    figure_canvas_agg = FigureCanvasTkAgg(fig, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)

#Execution du level Set
def levelset_execute(image,phi_0,timestep,mu,iter_inner,iter_outer,lmda,alfa,epsilon,errortol):

    #Transformer l'image en un tabeau numpy afin de faciliter sa manipulation
    img = np.array(image, dtype='float32')

    # edge indicator function G
    [Iy, Ix] = np.gradient(img)
    f = np.square(Ix) + np.square(Iy)
    g = 1 / (1+f)    


    phi = phi_0.copy()

    
    #Choisir la fonction potentiel (de préférence double-well)
    potential = 2
    
    if potential == 1:
        potentialFunction = 'single-well'  # use single well potential p1(s)=0.5*(s-1)^2, which is good for region-based model
    elif potential == 2:
        potentialFunction = 'double-well'  # use double-well potential in Eq. (16), which is good for both edge and region based models
    else:
        potentialFunction = 'double-well'  # default choice of potential function

    # start level set evolution
    breakloop= False
    n=0
    number_elements=phi.shape[0] * phi.shape[1]
    
    #Preparer l'affichage contenant l'évolution en temps réel du levelset
    plt.ion()
    fig_evolution = plt.figure(1)

    print("Executing Level set segmentation")
    while not breakloop:
        phi_old= phi
        phi = drlse.drlse_edge(phi, g, lmda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction)
        
        #show the levelset contour at every iteration
        fig_evolution.clf()
        show_evolution(img,phi,fig_evolution)
        plt.pause(0.001)
        

        evolution_ratio = np.linalg.norm(phi.ravel() - phi_old.ravel(), 2) / (number_elements)
        n=n+1

        print('Iteration number',n,' Evolution ratio',evolution_ratio)

        #Arreter l'évolution du levelset si on depasse le nombre d'itérations ou que le ratio d'évolution<errortol
        if evolution_ratio<errortol or n==iter_outer : breakloop=True

    #Fin de l'algorithme du levelset
    plt.ioff()
    plt.close(fig_evolution)
    print('End of Level set segmentation')
    return phi


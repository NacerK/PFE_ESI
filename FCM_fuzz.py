import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import skfuzzy as fuzz

def affichage(image, image_kmeans,phi_0):

    # affichage
    fig = plt.figure(figsize=(8, 4), dpi=120)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(image, cmap="gray")
    ax1.set_title('original')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(image_kmeans, cmap="gist_rainbow")
    ax2.set_title('Classification')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(phi_0, cmap="Reds")
    ax3.set_title('Tumeur')

    plt.show()


def fcm_execute(image,number_of_cluster,fuzzy_coef,seuil,max_iter):

        # Initialisation  de tableau des pixel 
        number_of_pixel = image.size
        #Transformer la matrice des pixels au format accepté par la fonction FCM de skfuzzy
        table_pixel = image.reshape(1,number_of_pixel)
        table_pixel= table_pixel.astype(float)


        #Execution de l'algorithme FCM
        centres, U_final, U_0, distances, fonction_objectif, iter_final, fpc = fuzz.cmeans(table_pixel,number_of_cluster,fuzzy_coef,seuil,max_iter)

        #Construction de l'image segmenté et du contour initial du levelset phi_0

        #Transformer le vecteur des centres en tableau des centres pour facilier sa manipulation
        centres = centres.flatten()

        #Center_phi_0 contiendra des centres avec des valeurs de c0 pour les clusters qui seront hors du contour initial
        # et -c0 pour les clusters qui seront à l'interieur du contour initial du level set
        center_phi_0 = np.float32(centres)


        #Table des pixels avec comme valeur pour chaque pixel l'indice du cluster pour lequel ils ont le plus grand degré d'appartenance
        seg = np.argmax(U_final, axis=0)

        #Phi_0 est une matrice ayant la meme dimension que l'image, avec pour chaque pixel la valeur de c0 si il est hors du contour initial
        # et -c0 si il est à l'interieur du contour initial du level set.
        phi_0=seg

        #On ordonne les centres afin de ne garder que les clusters ayant un grand niveau de gris comme depart du levelset
        center_ordered=centres
        center_ordered= center_ordered[ center_ordered !=-1 ]
        center_ordered.sort()

        for i in range (len(center_phi_0)):

                # On garde les clusters ayant le plus grand centres
                if center_phi_0[i].astype(int) < center_ordered[number_of_cluster - 1].astype(int):
                        center_phi_0[i] = -2
                else:
                        center_phi_0[i] = 2



        #On affecte à chaque pixel le niveau de gris du centre de son cluster
        seg = centres[seg.flatten()]  
        phi_0= center_phi_0[phi_0.flatten()]

        #reformer le tableau sous forme d'une image 
        seg = seg.reshape(image.shape).astype('int')
        phi_0=phi_0.reshape(image.shape)

        print("Fin du FCM. Nombre d'iterations: ", iter_final)

        return seg, phi_0



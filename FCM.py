import numpy as np
import cv2
import matplotlib.pyplot as plt
import random




def init_memberShip_martix(nb_pixels, nb_clusters):

    #création de la matrice d'appartenance vide
    membership_matrix = np.empty([nb_pixels, nb_clusters])

    #initialisation aléatoire
    for i in range(nb_pixels) :
        #créer un vecteur de "nb_clusters" column initialiser la matrice avec des valeur entre 0.0 et 1.0
        a = np.random.random(nb_clusters)
        n = []
        for j in a:
            n.append(j/a.sum())
        membership_matrix[i,:] = n

    return membership_matrix

def compute_centers(image_vector, memberShip_martix, fuzzy):
    #np.dot: le produit scalaire de deux matrice 

    num = np.dot(image_vector, memberShip_martix ** fuzzy)
    
    #np.sum(,axis=0): la sum de chaque column de la matrice
    den = np.sum(memberShip_martix ** fuzzy, axis=0)
    
    return num / den

def fonction_objectif(Uk1, Uk, seuil):
    

    distance = abs(Uk1-Uk)
    stop= True
   
    for i in distance:
        for j in i:
            if j < seuil :
                stop= False
                break

    return stop

def update_memberShip_martix(centre_vector, image_vector, fuzzy):
    #np.meshgrid : transformer le vecteur des centre en matrice avec n nombre de ligne (n est 
    # le nombre de pixel) telle que chaque colune de cette matrice contien la même valeur qui est la valeur 
    # de centre. et transformer le vecteur de l'image à une matrice ou chaque ligne contien la même valeur 
    # qui est la valeur de pixel)
    centre_matrix, image_matrix = np.meshgrid(centre_vector, image_vector)
    power = 2. / (fuzzy - 1)
    p1 = abs(image_matrix - centre_matrix) ** power
    p2 = np.sum((1. / abs(image_matrix - centre_matrix)) ** power, axis=1)
    return 1. / (p1 * p2[:, None])

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

   
    number_of_pixel = image.size
   
    # J: fonction objective
    J = 0.0

    # Initialisation du tableau des pixel
    #flatten(): Converti une matrice vers un tableau 
    #astype():  Converti le type des elements d'un tableau vers float 
    Table_pixel = image.flatten().astype('float')

    # Initialisation de la matrice de degré d'appartenance 
    membership_matrix = init_memberShip_martix(number_of_pixel, number_of_cluster)


    # Minimisation de la fonction objective 

    i = 0
    while True:
        

        # mise à jour de du tableau des centre 
        vector_centers = compute_centers(Table_pixel, membership_matrix, fuzzy_coef)
    
        # sauvgarder la matrice membership
        old_U = membership_matrix

        # mise à jour de la matrice des degré d'appartenance
        membership_matrix = update_memberShip_martix(vector_centers, Table_pixel, fuzzy_coef)
        
        Stop = fonction_objectif(membership_matrix, old_U, seuil)

        # Différence entre l'encienne fonction objective et la nouvelle 
    
        if i== max_iter or Stop :

            #print (" Fin de l'algorithme")
            #print (" Nombre d'itérations  = " + str(i))
        
            break
        
        i += 1
        #print('Iteration number: ', i)

    #------Construction de l'image segmenté et du contour initial du levelset phi_0------

    #Center_phi_0 contiendra des centres avec des valeurs de c0 pour les clusters qui seront hors du contour initial
    # et -c0 pour les clusters qui seront à l'interieur du contour initial du level set
    center_phi_0 = np.float32(vector_centers)

    vector_centers = np.uint8(vector_centers)

    #Table des pixels avec comme valeur pour chaque pixel l'indice du cluster pour lequel ils ont le plus grand degré d'appartenance
    seg = np.argmax(membership_matrix, axis=1)

    #Phi_0 est une matrice ayant la meme dimension que l'image, avec pour chaque pixel la valeur de c0 si il est hors du contour initial
    # et -c0 si il est à l'interieur du contour initial du level set.
    phi_0=seg

    #On ordonne les centres afin de ne garder que les clusters ayant un grand niveau de gris comme depart du levelset
    center_ordered=vector_centers
    center_ordered= center_ordered[ center_ordered !=-1 ]
    center_ordered.sort()

    for i in range (len(center_phi_0)):
            # On garde les clusters ayant les 2 plus grands centres 
            if center_phi_0[i]<center_ordered[number_of_cluster -1] : center_phi_0[i]=-2
            else: center_phi_0[i]=2


    #On affecte à chaque pixel le niveau de gris du centre de son cluster
    seg = vector_centers[seg.flatten()]  
    phi_0= center_phi_0[phi_0.flatten()]

    #reformer le tableau sous forme d'une image 
    seg = seg.reshape(image.shape).astype('int')
    phi_0=phi_0.reshape(image.shape)

    return seg, phi_0

    
    



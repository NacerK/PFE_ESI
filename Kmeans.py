import numpy as np 
import cv2
import matplotlib.pyplot as plt
 
def read_image(nom_image):

    #lire l'image 
    src = cv2.imread(nom_image)
    #convertir l'image en niveau de gris 
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    return gray

def kmeans_execute(nmbr_itteration, epsilon, nmbr_cluster,image):

    #convertir la matrice en vecteur 
    img = image.reshape((-1,1))

    #convertir les valeur de vecteur en float
    img = np.float32(img)

    #définir le critére d'arrêt qui contien le critére d'arrét à utiliser ainsi que leur valeur
    #nombre d'itération= 100 et epsilon= 0.1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, nmbr_itteration, epsilon)

    #définir le nombre de fois que l'algorithme fait la segmentation sur l'image 
    attempt=1

    #appeller l'algorithme K-means
    ret,label,center=cv2.kmeans(img,nmbr_cluster,None,criteria,attempt,cv2.KMEANS_RANDOM_CENTERS)
   
    #convertir les centre en interger 
    #center_phi_0 sera utilisé pour créer phi_0
    center_phi_0 = np.float32(center)
    center = np.uint8(center)


    #Ordonner les centres afin de garder les clusters ayant les plus hauts niveaux de gris (tumeurs) afin de les transformer en zone de depart du levelset
    center_ordered=center
    center_ordered= center_ordered[ center_ordered !=-1 ]
    center_ordered.sort()
  
    # Création de la la fonction phi_0 contenant le contour initial du level set (clusters ayant un grand niveaux de gris)

    for i in range (len(center_phi_0)):
        # On garde les clusters ayant les 2 plus grands centres, on leurs affecte -Co, et pour le reste des clusters = Co
        if center_phi_0[i]<center_ordered[nmbr_cluster-1] : center_phi_0[i]=-2
        else: center_phi_0[i]=2

    
    #faire de sort que chaque pixel aura le niveau de gris de centre de la classe à lequelle il appartient

    seg = center[label.flatten()]
    phi_0 = center_phi_0[label.flatten()]

    #convertir le vecteur en matrice pour avoir la forme de l'image gray
    seg = seg.reshape((image.shape))
    phi_0 = phi_0.reshape((image.shape))
    
    return seg, phi_0

def affichage(image,image_prep, image_kmeans,phi_0):

    # affichage
    fig = plt.figure(figsize=(8, 4), dpi=100)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(image, cmap="gray")
    ax1.set_title('original')

    ax4 = fig.add_subplot(2, 2, 2)
    ax4.imshow(image_prep, cmap="gray")
    ax4.set_title('Pre-traitement')

    ax2 = fig.add_subplot(2, 2, 3)
    ax2.imshow(image_kmeans, cmap="gray")
    ax2.set_title('segmentation')

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.imshow(phi_0, cmap="Reds")
    ax3.set_title('Phi_0')

  
    plt.show()




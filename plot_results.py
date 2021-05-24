import matplotlib.pyplot as plt
import ast
import pickle
import json
import numpy as np
  
# reading the data from the file
with open('fcm_vs_kmeans.json') as f:
    data = f.read()  

# reconstructing the data as a dictionary
js = json.loads(data)

#construction of the liste of dice, iou precision, and time excution 
Dice_kmeans, image_kmeans, Dice_fcm, image_fcm, time_kmeans, time_fcm, IOU_kmeans, IOU_fcm  = [], [], [], [], [], [], [], []
for dic in js :
    if 'K_means_Image' in dic:
        Dice_kmeans.append(dic['DICE'] *100)
        IOU_kmeans.append(dic['IOU'] *100)
        image_kmeans.append(dic['K_means_Image'])
        time_kmeans.append(dic['Time'])
    elif 'FCM_Image' in dic :
        Dice_fcm.append(dic['DICE']* 100)
        IOU_fcm.append(dic['IOU'] *100)
        image_fcm.append(dic['FCM_Image'])
        time_fcm.append(dic['Time'])
# plot graph Dice 
y_pos = np.arange(0, len(Dice_fcm), 1)
# Create bars
plt.bar(y_pos + 0.00, Dice_kmeans , color = 'b', width = 0.25)
plt.bar(y_pos + 0.25, Dice_fcm , color = 'g', width = 0.25)
# Create names on the x-axis
plt.xticks(y_pos, y_pos)
plt.legend(labels=['Dice_kmeans', 'Dice_fcm'])
plt.title('Dice coefficient pour Kmeans et FCM pour chaque image',fontsize=16)
plt.xlabel('Images',fontsize=14)
plt.ylabel('Dice coefficient %',fontsize=14)
# Show graphic
plt.show()

#plot graphic IOU

y_pos = np.arange(0, len(IOU_fcm), 1)
plt.bar(y_pos + 0.00, IOU_kmeans , color = 'b', width = 0.25)
plt.bar(y_pos + 0.25, IOU_fcm , color = 'g', width = 0.25)
plt.xticks(y_pos, y_pos)
plt.legend(labels=['IOU_kmeans', 'IOU_fcm'])
plt.title('IOU coefficient pour Kmeans et FCM pour chaque image',fontsize=16)
plt.xlabel('Images',fontsize=14)
plt.ylabel('IOU coefficient %',fontsize=14)
# Show graphic
plt.show()

#plot graphic time
print(time_fcm)
y_pos = np.arange(0, len(time_fcm), 1)
plt.bar(y_pos + 0.00, time_kmeans , color = 'b', width = 0.25)
plt.bar(y_pos + 0.25, time_fcm , color = 'g', width = 0.25)
plt.xticks(y_pos, y_pos)
plt.legend(labels=['TE_kmeans', 'TE_fcm'])
plt.title('Temps d\'exécution pour Kmeans et FCM sur chacune des images',fontsize=16)
plt.xlabel('Images',fontsize=14)
plt.ylabel('Temps en seconde',fontsize=14)
# Show graphic
plt.show()


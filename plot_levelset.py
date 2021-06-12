import matplotlib.pyplot as plt
import ast
import pickle
import json
import numpy as np

# reading the data from the file
with open('evaluation_results.json') as f:
    data = f.read()

# reconstructing the data as a dictionary
js = json.loads(data)

# construction of the liste of dice, iou precision, and time excution

Dice_kmeans, image_kmeans, Dice_fcm, image_fcm, time_kmeans, time_fcm, IOU_kmeans, IOU_fcm = [], [], [], [], [], [], [], []
Dice_levelSet, IOU_levelSet, image_levelSet, time_levelSet = [], [], [], []
for dic in js:
    if 'K_means_Image' in dic:
        Dice_kmeans.append(dic['DICE'] * 100)
        IOU_kmeans.append(dic['IOU'] * 100)
        image_kmeans.append(dic['K_means_Image'])
        time_kmeans.append(dic['Time'])
    elif 'FCM_Image' in dic:
        Dice_fcm.append(dic['DICE'] * 100)
        IOU_fcm.append(dic['IOU'] * 100)
        image_fcm.append(dic['FCM_Image'])
        time_fcm.append(dic['Time'])
    else:
        Dice_levelSet.append(dic['DICE'] * 100)
        IOU_levelSet.append(dic['IOU'] * 100)
        image_levelSet.append(dic['Image'])
        time_levelSet.append(dic['Time'])

# plot graph Dice
y_pos = np.arange(0, len(Dice_levelSet), 1)
# Create bars
plt.bar(y_pos + 0.00, Dice_levelSet, color='b', width=0.75)
# plt.bar(y_pos + 0.25, Dice_fcm , color = 'g', width = 0.25)
# Create names on the x-axis
plt.xticks(y_pos, y_pos)
# plt.legend(labels=['Dice_kmeans', 'Dice_fcm'])
plt.legend(labels=['Dice_levelSet'])
plt.title('Dice coefficient pour notre solution pour chaque image', fontsize=16)
plt.xlabel('Images', fontsize=14)
plt.ylabel('Dice coefficient %', fontsize=14)
# Show graphic
plt.show()

# plot graphic IOU

y_pos = np.arange(0, len(IOU_levelSet), 1)
plt.bar(y_pos + 0.00, IOU_levelSet, color='b', width=0.75)
# plt.bar(y_pos + 0.25, IOU_fcm , color = 'g', width = 0.25)
plt.xticks(y_pos, y_pos)
# plt.legend(labels=['IOU_kmeans', 'IOU_fcm'])
plt.legend(labels=['IOU_levelSet'])
plt.title('IOU coefficient pour notre solution pour chaque image', fontsize=16)
plt.xlabel('Images', fontsize=14)
plt.ylabel('IOU coefficient %', fontsize=14)
# Show graphic
plt.show()

# plot graphic time
y_pos = np.arange(0, len(time_levelSet), 1)
plt.bar(y_pos + 0.00, time_levelSet, color='b', width=0.75)
# plt.bar(y_pos + 0.25, time_fcm , color = 'g', width = 0.25)
plt.xticks(y_pos, y_pos)
# plt.legend(labels=['TE_kmeans', 'TE_fcm'])
plt.legend(labels=['time_levelSet'])
plt.title('Temps d\'exécution pour notre solution sur chacune des images', fontsize=16)
plt.xlabel('Images', fontsize=14)
plt.ylabel('Temps en seconde', fontsize=14)
# Show graphic
plt.show()
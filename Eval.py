
import numpy as np


def iou_dice(y_pred,y_true):
    axes = (0,1) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum  - intersection

    smooth = 0
    iou = (intersection + smooth) / (union + smooth)
    dice = 2 * (intersection + smooth)/(mask_sum + smooth)

    iou = np.mean(iou)
    dice = np.mean(dice)

    return iou,dice

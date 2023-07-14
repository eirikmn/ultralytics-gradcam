#import ultralytics_gradcam
#from ultralytics_gradcam import YOLO


#from ultralytics import YOLO


## edit model

"""
load model
set to eval mode
predict image

choose class
run backprop on pred wrt chosen class
get gradients
get activations

multiply gradients and activations
make heatmap
"""


# Path: ultralytics_gradcam.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import sys
sys.path.append('ultralytics')
from ultralytics import YOLO
model = YOLO("forams4-1280-augmentedraw-bigtest-200epochs-yolov8x.pt")
#im1 = Image.open("samplefromtest.jpg")
im1 = Image.open("sampleimg-3.jpg")
results = model.predict(source=im1, save=False, visualize = True)  # save plotted images


# loss function -> target
# send inn et args-object som inneholder info om hvilken klasse som skal brukes
# skal man bruke prediksjonsklasse og bbox
# skal man ta inn en fil?
# les på hva bbox_decode gjør

#0: sediment
#1: agglutinated
#2: calcareous
#3: planktic


#write a function that takes ¨
import numpy as np
import torch
run = 31
layer = 21

filepath = "./runs/detect/predict" + str(run) + "/"
heatmap = np.load(filepath + str(layer) + ".npy")

#plot a cv2 image
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

image = cv2.imread("sampleimg-1.jpg")
#image = Image.open("sampleimg-1.jpg")
labpath = "sampleimg-1.txt"

def labelreader(labpath=None):
        labels = []
        with open(labpath, 'r') as f:
            for line in f:
                #append each line to the labels as floats
                labels.append(line.strip())

        for i in range(len(labels)):
            labels[i] = labels[i].split(' ')
            for j in range(len(labels[i])):
                labels[i][j] = float(labels[i][j])

        #cast each element to float and store to tensor
        labels = torch.tensor(labels)
        return labels

labels = labelreader(labpath)

bboxes = labels


def make_heatmap(image, heatmap, transparancy = 0.3, plot=False):
    heatmapp = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    
    imgg = image.astype(np.uint8)
    cam_img = transparancy * heatmapp + (1-transparancy) * imgg
    cam_img = cam_img.astype(np.uint8)
    
    if plot:
        cv2.imshow("image",cam_img)
    
    return cam_img

cam_img = make_heatmap(image,heatmap)


def plot_heatmap(image, heatmap, bboxes = None, transparancy = 0.3):
    
    
    cam_img = make_heatmap(image, heatmap, transparancy, plot=False)
    
    if bboxes is not None:
        colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255,255,255]]
        boxlabels = ["sediment", "agglutinated", "calcareous", "planktic"]
        labheight = -10
        labwidth = -7
        #colors = [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255,255,255]]
        imgsz = (image.shape[0],image.shape[1])
        #imgsz = (1,1)
        for i in range(bboxes.shape[0]):
            box = bboxes[i,:]
            cls = box[0]
            cls = int(cls)
            xc = float(box[1])
            yc = float(box[2])
            w = float(box[3])
            h = float(box[4])
            x1 = round((xc-0.5*w)*imgsz[0])
            x2 = round((xc+0.5*w)*imgsz[0])
            y1 = round((yc-0.5*h)*imgsz[1])
            y2 = round((yc+0.5*h)*imgsz[1])
            
            cv2.rectangle(cam_img, (x1, y1), (x2, y2), colors[cls], 2)
            cv2.putText(cam_img, boxlabels[cls], (x1+labwidth, y1+labheight),cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[cls], 2)
    #an array of four colors
    
    #plt.imshow(cam_img)
    cv2.imshow("image",cam_img)
    
    plt.axis('off')  # Turn off axis labels
    plt.show()
    return cam_img

cam_img = plot_heatmap(image,heatmap)


def plot_bboxes(image,bboxes):
    colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255,255,255]]
    #colors = [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255,255,255]]
    imgsz = (image.shape[0],image.shape[1])
    imgsz = (1,1)
    if bboxes is not None:
        for i in range(bboxes.shape[0]):
            box = bboxes[i,:]
            cls = box[0]
            cls = int(cls)
            xc = float(box[1])
            yc = float(box[2])
            w = float(box[3])
            h = float(box[4])
            x1 = round((xc-0.5*w)*imgsz[0])
            x2 = round((xc+0.5*w)*imgsz[0])
            y1 = round((yc-0.5*h)*imgsz[1])
            y2 = round((yc+0.5*h)*imgsz[1])
            
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[cls], 2)
    

plot_heatmap(image,heatmap)


def plot_heatmap(image, heatmap, bboxes=None):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    heatmapp = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    imgg = image_rgb#[0].detach().cpu(
    
    cam_img = 0.3 * heatmapp + 0.7 * imgg
    cam_img = cam_img.astype(np.int32)
    #an array of four colors
    colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255,255,255]]
    colors = [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255,255,255]]
    imgsz = (image.shape[0],image.shape[1])
    imgsz = (1,1)
    if bboxes is not None:
        for i in range(bboxes.shape[0]):
            box = bboxes[i,:]
            cls = box[0]
            cls = int(cls)
            xc = float(box[1])
            yc = float(box[2])
            w = float(box[3])
            h = float(box[4])
            x1 = ((xc-0.5*w)*imgsz[0])
            x2 = ((xc+0.5*w)*imgsz[0])
            y1 = ((yc-0.5*h)*imgsz[1])
            y2 = ((yc+0.5*h)*imgsz[1])
            
            cv2.rectangle(cam_img, (x1, y1), (x2, y2), colors[cls], 2)
    
    #plt.imshow(cam_img)
    cv2.imshow("image",cam_img)
    
    plt.axis('off')  # Turn off axis labels
    plt.show()

plot_heatmap(image,heatmap)





def plot_heatmap_overlay(image, heatmap):
    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize the heatmap values to range [0, 1]
    normalized_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

    # Apply colormap to the normalized heatmap
    heatmap_colormap = plt.cm.jet(normalized_heatmap)

    # Resize the heatmap colormap to match the image size
    resized_heatmap = cv2.resize(heatmap_colormap, (image.shape[1], image.shape[0]))

    # Overlay the resized heatmap on the original image
    overlay = cv2.addWeighted(image_rgb, 0.6, resized_heatmap, 0.4, 0)

    # Plot the image with the heatmap overlay
    plt.imshow(overlay)
    plt.axis('off')  # Turn off axis labels
    plt.show()

plot_heatmap_overlay(image,heatmap)
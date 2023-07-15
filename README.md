# Grad-CAM for YOLO (version 8)

This repository contains a modified version of the 8th version of You Only Look Once (YOLO), created by [Ultralytics](https://github.com/ultralytics/ultralytics/). 

The implementation is similar to that of [_xiaowk5516_](https://github.com/ultralytics/yolov5/issues/5863) whom implemented Grad-CAM for YOLOv5. 

This project was done as part of the _Norwegian Artificial Intelligence Research Consortium_ (NORA) 2023 summer school _interpretability in Deep Learning_.

A demonstration of this code is available in _yolo8-gradcam.ipynb_. 

## Installation

To get started, clone this repository. Then, from the root folder load the modified ultralytics package as follows. 
```python
sys.path.append('ultralytics')
from ultralytics import YOLO
```

## Applications

First load the (pre-trained) YOLOv8 model and an image you want to apply Grad-CAM to, for example:
```python
model = YOLO("yolov8x.pt")
catimg = Image.open("cats.jpg")
```

Then, you have to tell the model where the label file, here _cats.txt_, is located. I have not yet implemented a way to pass this into the model call so you have to enter this into the source code of the package. The file you should change is:
"__./ultralytics/ultralytics/yolo/engine/predictor.py__" 

At line 382, where I have left a comment, you should place the path to the label file:
```python
labpath = "./cats.txt"
```

Then, run the prediction with _visualize = True_:
```python
results = model.predict(source=catimg, save=False, visualize = True, imgsz=1280)
```

The resulting heat maps are stored in _runs/detect/_ as _.npy_ files. 




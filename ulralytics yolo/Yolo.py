import os
HOME = os.getcwd()
print(HOME)

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO
from IPython.display import display, Image

from multiprocessing import freeze_support

if __name__ == '__main__':
    
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')

    # Train the model using the 'data.yaml' dataset for 3 epochs
    results = model.train(data='C:/Users/EaglesonLabs/object-detection/ulralytics yolo/datasets/data.yaml', epochs=1000, workers = 0, batch=54)

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Perform object detection on an image using the model
    results = model('https://ultralytics.com/images/bus.jpg')

    # Export the model to ONNX format
    success = model.export(format='onnx')
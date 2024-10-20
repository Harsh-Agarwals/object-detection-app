import torch
import cv2
import numpy as np
import random
import os

# Colors for the bounding boxes
colors = {
    # BGR format (used in openCV)
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'purple': (128, 0, 128),
    'brown': (42, 42, 165),
    'black': (0, 0, 0)
}

def load_model():
    """
    Function to load the YOLOv5 model.
    Input: None
    Output: YOLOv5 model
    """
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    return model

def detect_objects(model, image, app):
    """
    Function to detect the objects in the image.
    Input: YOLOv5 model, image, app (flask or streamlit)
    Output: Predictions
    """
    if app == "flask":
        # Getting the file from images folder, if the app is flask
        # When the app is initialized, the image is saved in the images folder
        basepath = os.getcwd()
        image_path = os.path.join(basepath, "images")
        image = os.path.join(image_path, image)
    return model(image)

def get_top_predictions_details(result, confidence_level):
    """
    Function to get the top predictions details.
    Input: YOLOv5 result, confidence level
    Output: Predictions
    """
    detections = {}

    for idx, res in enumerate(result.pred[0]):
        if res[4].item() > confidence_level:
            element = f"{result.names[res[5].item()]}_{idx}"
            detections[element] = {}
            detections[element]['confidence'] = res[4].item()
            detections[element]['class'] = result.names[res[5].item()]
            detections[element]['bbox'] = res[:4]

    return detections

def draw_bounding_boxes(image, predictions, app):
    """
    Function to draw the bounding boxes on the image.
    Input: Image, predictions, app (flask or streamlit)
    Output: Image with bounding boxes
    """
    if app == "flask":
        basepath = os.getcwd()
        image_path = os.path.join(basepath, "images")
        image = os.path.join(image_path, image)
        image = cv2.imread(image)

    # Converting the image to BGR format since openCV uses BGR format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for obj in predictions.keys():
        bbox = predictions[obj]['bbox']
        [x1, y1, x2, y2] = bbox.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_name = predictions[obj]['class']
        confidence = predictions[obj]['confidence']

        color_name = random.choice(list(colors.keys()))
        color = colors[color_name]

        # Drawing the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        label = f"{obj} {confidence:.2f}" # labels for the bounding box

        # Getting the text size
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.25, 2)

        # Drawing the rectangle for the text background (Each bounding box has a differnt color)
        cv2.rectangle(img, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)

        # Putting the text on the image
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

    # Converting image back to RGB format for display using matplotlib
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return rgb_image


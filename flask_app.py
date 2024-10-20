from flask import Flask, request, redirect, render_template, url_for
import os
import json
import cv2
import shutil
import matplotlib.pyplot as plt
from OD_functions import load_model, detect_objects, draw_bounding_boxes, get_top_predictions_details

app = Flask(__name__)

model = load_model()

# Creating the results folder to store our final results (json and image with bb)
if not os.path.exists("static/results"):
    os.makedirs("static/results")

if not os.path.exists("static/images"):
    os.makedirs("static/images")

if not os.path.exists("images"):
    os.makedirs("images")

@app.route("/object-detection-app", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        uploaded_file = request.files["image"]

        img = uploaded_file.filename

        # saving uploaded image
        uploaded_file.save(os.path.join("images", img))

        data = request.form.to_dict()
        image = uploaded_file.filename
        filename = image.split(".")[0]
        confidence = float(data["confidence"])
        
        result = detect_objects(model, image, "flask")
        predictions = get_top_predictions_details(result, confidence)
        rgb_image = draw_bounding_boxes(image, predictions, "flask")
        # saving rgb_image
        rgb_image_path = os.path.join("static/results", f"{filename}_bounding_boxes.jpg")
        cv2.imwrite(rgb_image_path, rgb_image)

        # original image path
        for pred in predictions.keys():
            predictions[pred]['bbox'] = predictions[pred]['bbox'].tolist()
        predictions_json = json.dumps(predictions, indent=4)
        json_file_path = os.path.join("static/results", f"{filename}_predictions.json")
        with open(json_file_path, "w") as json_file:
            json_file.write(predictions_json)

        filename1 = f"{filename}_bounding_boxes"
        filename2 = f"{filename}_predictions"
        # This actual_image is useless
        actual_image = f"{filename}"

        if predictions == {}:
            return redirect(url_for("results", detected_objects="No objects detected", confidence=confidence, actual_image=actual_image, filename1=filename1, filename2=filename2))

        return redirect(url_for("results", detected_objects=predictions_json, confidence=confidence, actual_image=actual_image, filename1=filename1, filename2=filename2))
    return render_template("index.html")

@app.route("/results")
def results():
    confidence = request.args.get('confidence')
    predictions = request.args.get('detected_objects')
    actual_image = request.args.get('actual_image')
    filename1 = request.args.get('filename1')
    filename2 = request.args.get('filename2')
    return render_template("detection.html", detected_objects=predictions, confidence=confidence, actual_image=actual_image, filename1=filename1, filename2=filename2)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

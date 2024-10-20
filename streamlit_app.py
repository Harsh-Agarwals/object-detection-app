from PIL import Image
import streamlit as st
import json
import io
from OD_functions import load_model, detect_objects, get_top_predictions_details, draw_bounding_boxes
import torch

model = load_model()

st.title("Object Detection App")

st.write("This is a simple object detection app built with Streamlit and Flask. Please upload an image, then click on the button to generate the object detections and click on download to download the OD image with the json file containing the details.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    filename = uploaded_file.name.split(".")[0]

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Showing this image with small size
    st.image(image, width=500, caption="Image for object detection")

    # Setting up confidence level for the output results
    st.write("Please set the confidence level for the detections")
    confidence_level =st.number_input("Confidence Level for detections", min_value=0.1, max_value=1.0)

    # Button while will trigger the object detection
    button = st.button("Detect Objects")

    if button:
        result = detect_objects(model, image, "streamlit")
        predictions = get_top_predictions_details(result, confidence_level)

        st.subheader(f"Here are the top detected objects with conficence >= {confidence_level}")

        st.write(predictions)

        # Drawing the bounding boxes on the image
        rgb_image = draw_bounding_boxes(image, predictions, "streamlit")

        # Showing the image with the bounding boxes
        st.image(rgb_image, caption="Image with bounding boxes", width=500)

        # Converting the image to PIL format to download it
        pil_image = Image.fromarray(rgb_image)

        # Save the image to a BytesIO object
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()  # Gettign the bytes of the image

        # Downloading the image with the bounding boxes
        st.download_button(label="Download OD image", data=img_bytes, file_name=f"{filename}_bounding_boxes.png", mime="image/png")

        for key, value in predictions.items():
            if isinstance(value["bbox"], torch.Tensor):
                value["bbox"] = value["bbox"].tolist()

        # Converting the predictions to JSON format
        json_data = json.dumps(predictions)

        # Downloading the json file with the details of the detections
        st.download_button(label="Download JSON", data=json_data, file_name=f"{filename}_detections.json", mime="application/json")

        # Prevent page reloading on button click
        st.stop()

        



        


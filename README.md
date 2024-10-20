Object detection app using YOLOv5

There are two versions of this app:
1. Flask app
2. Streamlit app

How to run the app:

Step-1: Cloning the repository
```
git clone https://github.com/your_username/object-detection-app.git
```

Step-2: Navigating to the app folder
```
cd object-detection-app
```

Step-3: Making a new venv environment
```
python -m venv env_name
```
env_name can be replaced with any name.

Step-4: Activating the venv environment
```
env_name\Scripts\activate
```

Step-5: Installing the dependencies
```
pip install -r requirements.txt
```

Step-6: Running the app (Running the Flask app)
```
python flask_app.py
go to http://127.0.0.1:8080/object-detection-app
Input image and confidence threshold and click on the button to generate the object detections.
``` 

Step-7: Running the app (Running the Streamlit app)
```
streamlit run streamlit_app.py
```

Main libraries used in this project:
1. Flask (for backend for flask app)
2. HTML, CSS (for frontend for flask app)
3. Streamlit (For frontend and backend for streamlit app)
4. YOLOv5 using torch
4. OpenCV

# Dockerfile for the object detection app

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "flask_app.py"]
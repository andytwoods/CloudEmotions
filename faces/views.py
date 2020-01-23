import base64

from django.http import JsonResponse
from django.shortcuts import render

import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

tf.config.set_visible_devices([], 'GPU')

detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(detection_model_path)


emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
emotion_classifier = load_model(emotion_model_path)

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]


def __data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def __extract_faces(b64_string):
    img = __data_uri_to_cv2_img(b64_string)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    simplified_faces = []
    for face in faces:
        simplified_faces.append(face.tolist())
    return simplified_faces


def __extract_emotion(b64_face):
    img = __data_uri_to_cv2_img(b64_face)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    return {label: float(emotion_probability)}


def frame(request):
    frame = request.GET.get('frame', None)
    if frame:
        faces = __extract_faces(frame)
        return JsonResponse({'faces': faces}, status=200)

    frame = request.GET.get('emotion', None)

    if frame:
        emotion = __extract_emotion(frame)
        return JsonResponse({'emotion': emotion}, status=200)

    context = {}
    return render(request, 'faces/webcam.html', context=context)

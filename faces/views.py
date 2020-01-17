import base64
import io

from django.http import JsonResponse
from django.shortcuts import render
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(detection_model_path)


def __data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def crunch_frame(b64_string):
    img = __data_uri_to_cv2_img(b64_string)
    cv2.imshow("test", img)
    cv2.waitKey(50)

def frame(request):

    frame = request.GET.get('frame', None)
    if frame:
        crunch_frame(frame)
        return JsonResponse({}, status=200)

    context = {}
    return render(request, 'faces/webcam.html', context=context)

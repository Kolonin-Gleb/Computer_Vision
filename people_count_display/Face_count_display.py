import sys
import numpy as np
import cv2
import math
from PIL import Image

# Класс, позволяющий читать кадры из видеопотока
frame = cv2.VideoCapture(0)

# Создаем объект для обнаружения лица
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while True:
    # Получаем кадр
    status, image = frame.read()
    image_done = image.copy()  # делаем копию кадра

    faces = face.detectMultiScale(image, scaleFactor=1.2, minNeighbors=6, minSize=(110,110))
    # scaleFactor - параметр, определяющий "тщательность" поиска лица на изображении. Чем он ближе к 1,
    # тем больше времени уйдет на поиск лица, тем дольше будет происходить этот поиск. При больших
    # значениях есть риск пропустить лица. По умолчанию - 1.1
    # Объяснение всех параметров: https://russianblogs.com/article/76161527227/

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_done, f"Faces on the screen = {len(faces)}", (20, 28), font, 0.5, color=(250, 0, 0), thickness=1)
    for (x, y, w, h) in faces:
        cv2.rectangle(image_done, (x,y), (x+w,y+h), (0,255,64), 2)

    cv2.imshow("Face", image_done)

    k = cv2.waitKey(30)
    if k == 27:
        break

frame.release()
cv2.destroyAllWindows()

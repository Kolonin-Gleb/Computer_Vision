from PIL import Image, ImageEnhance
import sys
import numpy as np
import cv2
import math
import os
import shutil  # Библиотека для работы с файлами
import glob    # Расширение для использования Unix обозначений при задании пути к файлу
import tensorflow as tf
from tensorflow.keras.models import Model  # Импортируем модели keras: Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, concatenate, Activation, MaxPooling2D, Conv2D, BatchNormalization, Flatten, Dense, Dropout, Conv2DTranspose, Concatenate, Reshape

# Новые библиотеки
import face_recognition


# Загрузка модели и весов
mf = open("faces.json", 'r')
j_data = mf.read()
mf.close()
model = tf.keras.models.model_from_json(j_data)
model.load_weights('faces.hdf5')

persons_list = ['gleb', 'sofia', 'alex', 'arkady', 'nikita']

# Функция умной обрезки
def smart_crop(img, target_size):
    if img.width < target_size[0]:
        new_height = int(target_size[0] * img.height / img.width)
        img = img.resize((target_size[0], new_height), Image.ANTIALIAS)

    if img.height < target_size[1]:
        new_width = int(img.width * target_size[1] / img.height)
        img = img.resize((new_width, target_size[1]), Image.ANTIALIAS)

    new_img = Image.new("RGB", target_size, (255, 255, 255)).convert('L')

    left = (target_size[0] - img.width) // 2
    top = (target_size[1] - img.height) // 2

    new_img.paste(img, (left, top))

    return new_img


# Накладываем имя
def add_person(img, coord, text):
    #color = (0,255,64)
    #clock_img = cv2.rectangle(img, coord, (coord[0]+20, coorde[1]+30), color, -1)
    face_img = img.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(face_img, text, coord, font, 1, color=(255,255,255), thickness=2)

    return face_img

# Класс для работы с видеопотоком с видеокамеры
frame = cv2.VideoCapture(0)

#
# Обрабатываем кадры в цикле
#

while True:
    # Получаем кадр
    status, image = frame.read()
    image_done = image.copy()
    
    # Находим лица с помощью Face_recognition
    face_locations = face_recognition.face_locations(image, model='hog') #cnn ? # Указать название переменной, что соответствует модели импортируемой из json?
    # face_locations is now an array listing the co-ordinates of each face!

    print(face_locations)
    # Правый нижний, Левый верхний
    for (top, right, bottom, left) in face_locations:
        # Получить из кадра лицо и сохраняем в отдельную картинку
        
        # Изменить формирование квадратика по координатам
        image_face_frame = image[top:bottom, left:right]
        image_face_frame = cv2.cvtColor(image_face_frame, cv2.COLOR_BGR2GRAY)
        img_obj = Image.fromarray(image_face_frame)
        img_obj = smart_crop(img_obj, (140, 140))
        img = np.array(img_obj).reshape(140,140,1)
        img = img.reshape(1,140,140,1)
        # print("!!!")
        # print(img.shape)
        
        # Подаем лицо на распознавание в нейросеть
        prediction = model.predict(img)
        person = np.argmax(prediction[0])
        
        # Обводим квадрат и выводим надпись, чьё лицо
        cv2.rectangle(image_done, (top, right), (bottom, left), (0,255,64), 2)
        image_done = add_person(image_done, (top, right-4), persons_list[person])

    cv2.imshow("Face", image_done)

    k = cv2.waitKey(30)

    # Обрабатываем нажатие клавиши ESC
    if k == 27:
        break

frame.release()
cv2.destroyAllWindows()

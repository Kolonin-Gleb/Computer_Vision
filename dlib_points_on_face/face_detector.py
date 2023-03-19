# TODO: Выполняю задание 
# Поиск ключевых точке, выравнивание лица на базе dlib
# https://ithub.bulgakov.app/lessons/49419

# Ход работы:
# 1) Получить из видеопотока кадр
# 2) Найти на нём лицо
# 3) Отрисовать 3 точки лица прямо на кадре видеопотока

import dlib # Мощнейшая библиотека, содержащая функции для распознавания лиц
import cv2 # Для работы с видеопотоком


'''
from PIL import Image
import numpy as np
import cv2
import os

# Новые библиотеки
import face_recognition
'''

def get_rect_coordinates(r):
  x1 = r.left() # left point
  y1 = r.top() # top point
  x2 = r.right() # right point
  y2 = r.bottom() # bottom point

  return (x1,y1),(x2,y2)

# Предобученная модель определения ключевых точек лица
dlib_predictor_path = 'shape_predictor_68_face_landmarks.dat'

# Класс для работы с видеопотоком с видеокамеры
frame = cv2.VideoCapture(0)

# Обрабатываем кадры в цикле
while True:
    # Получаем кадр
    status, image = frame.read()
    # Перевод в RGB
    image_face_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ============= Находим лицо на изображении. Лучше, чем OpenCV =============

    detector = dlib.get_frontal_face_detector() # Создаем детектор для поиска лиц
    face_rects = detector(image_face_frame, 1) # Находим лица, в данном случае одно

    # TODO: Костыль, чтобы всегда находилось лицо
    try:
        face_rect = face_rects[0] # Так как вернется массив, берем единственный элемент
    except:
        continue

    # Получаем координаты в виде двух точек
    rect_start, rect_end = get_rect_coordinates(face_rect)
    print(rect_start, rect_end) # Для отладки

    # ============= Находим ключевые точки исп. предобуенную модель =============

    predictor = dlib.shape_predictor(dlib_predictor_path)
    points = predictor(image_face_frame, face_rect)

    # Для распознавания лица в библиотеке dlib используются 68 точек
    image_points = image_face_frame.copy()
    for n in range(0, 68):
        x = points.part(n).x
        y = points.part(n).y
        # Ноносим точки на наше изображение
        cv2.circle(img=image_points, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

    # Отобразить image_points пользователю!

    cv2.imshow("Face", image_points)

    # Возможность нажатия клавиши
    key = cv2.waitKey(30)

    # Обрабатываем нажатие клавиши ESC
    if key == 27:
        break

frame.release()
cv2.destroyAllWindows()
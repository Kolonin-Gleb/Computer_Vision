import os
from imutils import paths # нам нужна для получения списка файлов
import pickle # для сохранения данных в файл

import face_recognition # распознавание лиц на базе dlib
import cv2

# ВАЖНО: ВСЕ ЛИЦА ИЗ БД будут записаны в один файл.
# Этот файл будет явл. БД векторизованных лиц.

imagePaths = list(paths.list_images('faces'))
knownEncodings = []
knownNames = []

# перебираем все папки с изображениями
for imagePath in imagePaths:
    # Извлекаем имя человека из названия папки
    name = imagePath.split(os.path.sep)[-2]

    # загрузка ч\б изображения
    image = cv2.imread(imagePath)

    # Находим лица с помощью Face_recognition
    # Т.к. всё изображение это и есть лицо - беру его целиком по координатам
    boxes = (0, 140, 140, 0) #TODO: (top, right, bottom, left) - должно быть верно...

    # вычисляем вектор для каждого лица
    encodings = face_recognition.face_encodings(image, [boxes])

    # добавляем каждый вектор
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

# сохраним векторы вместе с их именами в формате словаря
data = {"encodings": knownEncodings, "names": knownNames}

f = open(f"faces.enc", "wb")
f.write(pickle.dumps(data)) # для сохранения данных в файл используем метод pickle
f.close()

import os
from imutils import paths # нам нужна для получения списка файлов
import pickle # для сохранения данных в файл

import face_recognition # распознавание лиц на базе dlib
import cv2

persons_list = ['gleb', 'sofia', 'alex', 'arkady'] # доступные имена
curent_person = 'gleb' # Перед запуском съёмки человека указать его имя

imagePaths = list(paths.list_images(curent_person))
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
    boxes = (0, 140, 140, 0)
    
    # вычисляем вектор для каждого лица
    encodings = face_recognition.face_encodings(image, [boxes])
    # добавляем каждый вектор
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

# сохраним векторы вместе с их именами в формате словаря
data = {"encodings": knownEncodings, "names": knownNames}

f = open(f"{curent_person}.enc", "wb")
f.write(pickle.dumps(data)) # для сохранения данных в файл используем метод pickle
f.close()

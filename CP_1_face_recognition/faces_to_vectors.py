import os
from imutils import paths # нам нужна для получения списка файлов
import pickle # для сохранения данных в файл

import face_recognition # распознавание лиц на базе dlib
import cv2

# =============== Установка имени человека для добавления в БД ===============
with open(r'faces\faces.txt', 'r') as f:
    persons = f.readlines()

persons = [person.strip() for person in persons]
curent_person = ''

print("Введите имя человека, фотографии которого будут переведены в вектор: ")
curent_person = input()

while curent_person == '' or curent_person not in persons:
    print("Человек с этим именем отсутствует в БД лиц!\n\n")
    print("Введите имя человека, фотографии которого будут переведены в вектор: ")
    curent_person = input()


imagePaths = list(paths.list_images(f'faces/{curent_person}'))
knownEncodings = []
knownNames = []

# перебираем все папки с изображениями
for imagePath in imagePaths:
    # Извлекаем имя человека из названия папки
    name = curent_person

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

with open(f"vectors/{curent_person}.enc", "wb") as f:
    f.write(pickle.dumps(data)) # для сохранения данных в файл используем метод pickle
    f.close()

from PIL import Image
import numpy as np
import cv2
import os

# Новые библиотеки
import face_recognition

# =============== Установка имени человека для добавления в БД ===============
with open(r'faces\faces.txt', 'r') as f:
    persons = f.readlines()

persons = [person.strip() for person in persons]
curent_person = ''

print("Введите имя человека, которого будете снимать: ")
curent_person = input()

while curent_person == '' or curent_person in persons:
    print("Человек с этим именем уже есть в БД лиц!\n\n")
    print("Введите имя человека, которого будете снимать: ")
    curent_person = input()

# Перед запуском съёмки человека указать его имя
with open(r'faces\faces.txt', 'a') as f:
    f.write(curent_person)

# Создание папки под пользователя
os.mkdir(f'faces/{curent_person}')



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

# Класс для работы с видеопотоком с видеокамеры
frame = cv2.VideoCapture(0)

faces_collected = 0 # Количество собранных лиц
faces_needed = 10

# Обрабатываем кадры в цикле
while True:
    # Получаем кадр
    status, image = frame.read()
    image_done = image.copy()

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_done, f"Faces collected = {faces_collected} out of {faces_needed}", (20, 28), font, 0.5, color=(250, 0, 0), thickness=1)
    
    # Находим лица с помощью Face_recognition
    face_locations = face_recognition.face_locations(image, model='hog')
    # face_locations is now an array listing the co-ordinates of each face!
    print(face_locations)

    # Возможность нажатия клавиши
    key = cv2.waitKey(30) # Клавиша "r или ESC"

    # Получить из кадра лицо и сохранить в отдельную картинку
    for (top, right, bottom, left) in face_locations:
        # Обводим в квадрат лицо
        cv2.rectangle(image_done, (top, right), (bottom, left), (0,255,64), 2)
        
        image_face_frame = image[top:bottom, left:right]
        image_face_frame = cv2.cvtColor(image_face_frame, cv2.COLOR_BGR2GRAY) # ПЕРЕВОД В НУЖНОЕ ЦВЕТОВОЕ ПРЕДСТАВЛЕНИЕ
        img_obj = Image.fromarray(image_face_frame)
        img_obj = smart_crop(img_obj, (140, 140)) # Изображения для сохранения и будущей векторизации

        if faces_collected <= faces_needed and key == 114: # Обработка клавиши "r" (114) для сохранения фото
            img_obj.save(f"faces/{curent_person}/{faces_collected}.jpg")
            faces_collected += 1
        else:
            break
    
    cv2.imshow("Face", image_done)

    # Обрабатываем нажатие клавиши ESC
    if key == 27:
        break

frame.release()
cv2.destroyAllWindows()
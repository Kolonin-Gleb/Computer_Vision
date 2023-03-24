# Задание 2
# С помощью видеопотока распознавайте лица проходящих.
# Фиксируйте дату и время прохода

# План решения:
# Взять лицо проходящего человека из видеопотока.
# Определить кому оно принадлежит. 
# TODO: В каком колабе есть код для этого?
# Сохранить лицо в файл с именем вида кто_дата_время.jpg

from PIL import Image # Для обработки фото
import cv2 # Для работы с видео
import face_recognition # Для распознавания лиц

# Для сохранения файла с нужной датой и временем
import datetime as dt


persons_list = ['gleb', 'sofia', 'alex', 'arkady'] # доступные имена
curent_person = None # Перед запуском съёмки человека указать его имя

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

# Обрабатываем кадры в цикле
while True:
    # Получаем кадр
    status, image = frame.read()
    image_done = image.copy()
    
    # Находим лица с помощью Face_recognition
    face_locations = face_recognition.face_locations(image, model='hog')
    # face_locations is now an array listing the co-ordinates of each face!

    print(face_locations)
    # Получить из кадра лицо и сохранить в отдельную картинку
    for (top, right, bottom, left) in face_locations:
        # Обводим в квадрат лицо
        cv2.rectangle(image_done, (top, right), (bottom, left), (0,255,64), 2) # В правильном ли порядке расставил слова?
        
        image_face_frame = image[top:bottom, left:right]
        # ПЕРЕВОД В НУЖНОЕ ЦВЕТОВОЕ ПРЕДСТАВЛЕНИЕ
        image_face_frame = cv2.cvtColor(image_face_frame, cv2.COLOR_BGR2GRAY)
        # image_face_frame = cv2.cvtColor(image_face_frame, cv2.COLOR_GRAY2RGB) # Из GRAY в RGB нужно ли?
        img_obj = Image.fromarray(image_face_frame)
        img_obj = smart_crop(img_obj, (140, 140)) # Изображение для сохранения в БД

        # Получение вектора текущего лица
        


        # ============= Сравнение полученного вектора с имеющимися в БД векторов лиц =============



        # curent_person = None # Нужно распознать пойманное лицо исп. БД векторов.

        current_time = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{curent_person}_{current_time}.jpg"
        img_obj.save(f"face_control/{file_name}")

    cv2.imshow("Face_Control", image_done)

    # Возможность нажатия клавиши
    key = cv2.waitKey(30)
    # Обрабатываем нажатие клавиши ESC
    if key == 27:
        break

frame.release()
cv2.destroyAllWindows()
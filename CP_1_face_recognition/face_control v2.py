# ПРИМЕЧАНИЕ v2 - делаю распознавание лиц другим способом!

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
import pickle # Для БД лиц

# Для сохранения файла с нужной датой и временем
import datetime as dt


persons_list = ['gleb', 'sofia', 'alex', 'arkady'] # доступные имена
# Перед запуском съёмки человека указать его имя
curent_person = None

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

# ============= Загрузка БД векторов лиц =============
data = pickle.loads(open('faces.enc', "rb").read())

# Обрабатываем кадры в цикле
while True:
    # Получаем кадр
    status, image = frame.read()
    image_done = image.copy()

    # ============= Поиск векторов всех лиц на изображении =============
    encodings = face_recognition.face_encodings(image)
    names = []

    # Перебираем все найденные лица
    for encoding in encodings:
        # Сравниваем поступившую фотографию с теми, что в базе
        matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.6)
        # tolerance=0.6 - порог срабатывания
        
        # По умолчанию выставим имя - неизвестен
        name = "Unknown"
        
        if True in matches:
            # Сравниваем изображение лица с данными в нашей базе (при совпадении мы получаем True)
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # Пробегаемся по всем найденным лицам
            for i in matchedIdxs:
                # Находим в базе имя человека
                name = data["names"][i]
                # Увеличиваем счетчик пололжительных срабатываний для данного имени
                counts[name] = counts.get(name, 0) + 1
            
            # Берем то имя, которое набрало наибольшее количество срабатываний
            name = max(counts, key=counts.get)
        
        # Добавляем найденное имя в массив
        names.append(name)

    # ============= Обнаружение лиц с помощью Face_recognition =============
    face_locations = face_recognition.face_locations(image, model='hog')
    # face_locations is now an array listing the co-ordinates of each face!
    # (top, right, bottom, left) order
    # (y1    x1       y2     x2)
    
    for ((top, right, bottom, left), name) in zip(face_locations, names):
        top_left = (left, top)
        bottom_right = (right, bottom)
        # Обводим в квадрат лицо
        cv2.rectangle(image_done, top_left, bottom_right, (0,255,0), 2)
        # Наложение текста
        cv2.putText(image_done, name, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        image_face_frame = image[top:bottom, left:right]
        # ПЕРЕВОД В НУЖНОЕ ЦВЕТОВОЕ ПРЕДСТАВЛЕНИЕ
        image_face_frame = cv2.cvtColor(image_face_frame, cv2.COLOR_BGR2GRAY)
        # image_face_frame = cv2.cvtColor(image_face_frame, cv2.COLOR_GRAY2RGB) # Из GRAY в RGB нужно ли?
        img_obj = Image.fromarray(image_face_frame)
        img_obj = smart_crop(img_obj, (140, 140)) # Изображение для сохранения в БД

        curent_person = name # Лицо, что было обработано на этой итерации цикла

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
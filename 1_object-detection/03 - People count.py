# Подключаем нужные библиотеки
import numpy as np
import cv2

# Класс, позволяющий читать кадры из видеопотока
cap = cv2.VideoCapture(1)
 
# Инициализируем детектор обнаружения людей в кадре
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()


# Определяем файл, куда будем записывать получившийся видеопоток
out = cv2.VideoWriter(
    'persons.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))


while True:
    # Получаем очередной кадр из потока
    ret, frame = cap.read()
    
    # Изменяем размер изображения
    frame = cv2.resize(frame, (640, 480))
    # Переводим изображение из цветного в черно-белое
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Находим в кадре людей и возвращаем квадраты с ними
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))
    
    # Переводим полученные координаты квардартов в вид, пригодный для использования
    # в фукнции рисования прямоугольников
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    
    for (xA, yA, xB, yB) in boxes:
        # Отображаем наши прямоугольники синим цветом и шириной 2 пикселя
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    # Записываем кадр в поток выходного файла
    out.write(frame.astype('uint8'))
    
    # Отрисовываем на экране кадр
    cv2.imshow('Person detection', frame)
    
    # При нажатии на ESC выйдем из программы
    k = cv2.waitKey(30)
    if k == 27:
        break

# Завершаем запись видео
out.release()

# В заверешнии всего закрываем все окна
cap.release()
cv2.destroyAllWindows()
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import datetime

# Класс, позволяющий читать кадры из видеопотока
# Если используем встроенную камеру - указываем 0
# Если подключаете камеру по USB - указываем 1
# Если будут еще камеры, то 2, 3 и т.д.
frame = cv2.VideoCapture(1)


# Статусы того, показываем мы часы на изображении или нет
isClock = False


# Накладываем время
def add_clock(img):
	color = (16, 16, 16)
	clock_img = cv2.rectangle(img, (11, 11), (100, 35), color, -1)

	font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
	time = datetime.datetime.today().strftime("%H:%M:%S")
	cv2.putText(clock_img, time, (20, 28), font, 0.5, color=(120, 120, 120), thickness=1)
	return clock_img


# Обрабатываем кадры в цикле
while True:
	status, image = frame.read()

	if isClock:
		image = add_clock(image)

	# Отображаем фрэйм с видео-трансляцией
	cv2.imshow("TV show", image)

	k = cv2.waitKey(30)

	# Обрабатываем нажатие клавиши Esc для выхода
	if k == 27:
		break

	# Обрабатываем нажатие клавиши t, которая включает или отключает часы
	if k == 116:
		if isClock:
			isClock = False
		else:
			isClock = True


frame.release()
cv2.destroyAllWindows()

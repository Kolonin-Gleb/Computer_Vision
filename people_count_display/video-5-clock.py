import cv2
from PIL import Image, ImageEnhance
import numpy as np
import datetime

frame = cv2.VideoCapture(0)

# Стутусы отображения
isColor = True
isLogo = False
isCaption = False
isClock = False

# Счетчик времени для подписи
captionTimer = 0

# Уровни прозрачности
opacity_logo = 1
opacity_caption = 1

# Загружаем картинки
caption_img = cv2.imread('caption.png', cv2.IMREAD_UNCHANGED)
logo_img = cv2.imread('logo-rgba.png', cv2.IMREAD_UNCHANGED)

# Переводим массивы в объекты Pillow
logo = Image.fromarray(logo_img)
caption = Image.fromarray(caption_img)


# Делаем изображение черно-белым
def add_mono(img):
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
	return img


# Накладываем логотип
def add_logo(img):
	# Переводим массив изображения в объект Pillow и преобразуем в тип RGBA с маской прозрачности
	img_out = Image.fromarray(img)
	l = logo.copy()
	alpha = ImageEnhance.Brightness(l.split()[3]).enhance(opacity_logo)
	l.putalpha(alpha)
	img_out.paste(logo, (540, 15), l)

	return np.asarray(img_out, dtype='uint8')


# Накладываем подпись
def add_caption(img):
	# Переводим массив изображения в объект Pillow и преобразуем в тип RGBA с маской прозрачности
	img_out = Image.fromarray(img)

	global opacity_caption
	global isCaption
	global captionTimer

	# print(captionTimer)

	if (captionTimer < 50) and (opacity_caption < 1):
		opacity_caption += 0.04

	if captionTimer > 80:
		if opacity_caption > 0:
			opacity_caption -= 0.04
		else:
			isCaption = False

	cap = caption.copy()
	alpha = ImageEnhance.Brightness(cap.split()[3]).enhance(opacity_caption)
	cap.putalpha(alpha)

	img_out.paste(caption, (0, 380), cap)

	captionTimer += 1

	return np.asarray(img_out, dtype='uint8')


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

	if not isColor:
		image = add_mono(image)

	if isLogo:
		image = add_logo(image)

	if isCaption:
		image = add_caption(image)

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

	# Обрабатываем нажатие клавиши l, которая включает или отключает логотип
	if k == 108:
		if isLogo:
			isLogo = False
		else:
			isLogo = True

	# Обрабатываем нажатие клавиши c, которая включает или отключает подпись
	if k == 99:
		if not isCaption:
			isCaption = True
			captionTimer = 0
			opacity_caption = 0

	# Обрабатываем нажатие клавиши b, которая включает или отключает цветность
	if k == 98:
		if isColor:
			isColor = False
		else:
			isColor = True

	# Обрабатываем нажатие клавиши +, которая уменьшает прозрачность лого
	if k == 45:
		if opacity_logo > 0:
			opacity_logo -= 0.1

	# Обрабатываем нажатие клавиши +, которая уменьшает прозрачность лого
	if k == 43:
		if opacity_logo < 1:
			opacity_logo += 0.1

frame.release()
cv2.destroyAllWindows()

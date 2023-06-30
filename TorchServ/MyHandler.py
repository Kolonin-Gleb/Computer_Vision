import logging
import torch
import torch.nn.functional as F
import io
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler

class MyHandler(BaseHandler):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms. CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    # Обработка одного изображения
    def preprocess_one_image(self, req):
        # Получаем бинарные данные графического файла
        image = req.get("data")
        if image is None:
            image = req.get("body")       

        # Загружаем бинарные данные в виде файла
        image = Image.open(io.BytesIO(image))
        image = self.transform(image)
        image = image.unsqueeze(0)
        return image

    # Обработка всех изображений из REST-запроса
    def preprocess(self, requests):
        images = [self.preprocess_one_image(req) for req in requests]
        images = torch.cat(images)    
        return images


    # Предикт всех изображений
    def inference(self, x):
        outs = self.model.forward(x)
        probs = F.softmax(outs, dim=1) 
        preds = torch.argmax(probs, dim=1)
        return preds


    # Обработка данных для вывода
    def postprocess(self, preds):
        res = []
        preds = preds.cpu().tolist()
        for pred in preds:
            label = self.mapping[str(pred)][1]
            res.append({'label' : label, 'index': pred })
        return res
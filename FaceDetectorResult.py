from super_image import EdsrModel, ImageLoader
from PIL import Image
import requests, os

image = Image.open('firts1.jpg')

model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
inputs = ImageLoader.load_image(image)
preds = model(inputs)

ImageLoader.save_image(preds, './scaled_2x.png')
ImageLoader.save_compare(inputs, preds, './scaled_2x_compare.png')


import cv2


img = cv2.imread('./scaled_2x_compare.png') 
cv2.imshow('1',img)

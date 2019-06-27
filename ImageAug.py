# -*- coding: utf-8 -*-
# *****图像的扩充*****
import os
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


Source_Path = 'Raw_Img/类别'
dst_path = 'OutputDir/种类'
datagen = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
)

gen = datagen.flow_from_directory(
    Source_Path,
    batch_size = 15,
    save_to_dir=dst_path,
    save_prefix='种类名称',
    save_format='jpg'
)
for i in range(15):
    gen.next()
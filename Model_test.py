#coding:utf-8
#coding: utf-8
#*****Main Trainning*****
from keras.models import load_model #加载模型
import matplotlib.image as  processimage #预处理图片库
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras import backend as K
K.clear_session()

class ModelTest(object):
    def __init__(self,ModelFile,TestFile,ImageType,Width=100,Height=100):
        self.modelfile = ModelFile
        self.test_file = TestFile
        self.Width = Width
        self.Height = Height
        self.ImageType = ImageType

    #预测
    def Test(self):
        # 预测
        model = load_model(self.modelfile)
        # 处理照片格式和尺寸
        imag_open = Image.open(self.test_file)
        conv_RGB = imag_open.convert('RGB')
        new_img = conv_RGB.resize((self.Width,self.Height),Image.BILINEAR)
        new_img.save(self.test_file)
                                                                                   
        image = processimage.imread(self.test_file)
        image_to_array = np.array(image)/255.0                                                      
        image_to_array = image_to_array.reshape(-1,100,100,3)
        #测试照片                                                                   
        test = model.predict(image_to_array)
        Final_Test = [result.argmax() for result in test][0]

        #Display
        count = 0
        ImageCount = [0.0,0.0,0.0,0.0,0.0,0.0]
        ImageTypeCount = ['type1', 'type2','type3', 'type4', 'type5', 'type6']

        ImageCountmid = 0.0
        ImageTypeCountmid = 'type1'
        for i in test[0]:
            percent2 = '%.2f%%' % (i * 100)
            percent0 = '%.2f'%(i*100)
            percent = float(percent0)
            ImageCount[count] = percent
            ImageTypeCount[count] = self.ImageType[count]
            count +=1
        for i in range(0, 3):
            for j in range(i, 5):
                if (ImageCount[j] > ImageCount[j + 1]):
                    ImageCountmid = ImageCount[j]
                    ImageCount[j] = ImageCount[j + 1]
                    ImageCount[j + 1] = ImageCountmid

                    ImageTypeCountmid = ImageTypeCount[j]
                    ImageTypeCount[j] = ImageTypeCount[j + 1]
                    ImageTypeCount[j + 1] = ImageTypeCountmid
        for i in range(3, 6):
            print(ImageTypeCount[i], '概率:',ImageCount[i],'%')


ImageType = ['type1', 'type2','type3', 'type4', 'type5', 'type6']
#实例化类
Test = ModelTest(TestFile='Test_img/1.jpg',ModelFile='modelname.h5',ImageType=ImageType)
Test.Test()

# from keras.preprocessing.image import ImageDataGenerator
# training_datagen=ImageDataGenerator(rescale=1./255)
# training_set=training_datagen.flow_from_directory('./origin_data/train',target_size=(50,50),batch_size=32,class_mode='binary')
# from keras.models import Sequential
# from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
#
# model = Sequential()
# # model.add(Conv2D(32,(3,3),input_shape=(50,50,3),activation='relu'))
# # model.add(MaxPool2D(pool_size=(2,2)))
# # model.add(Conv2D(32,(3,3),input_shape=(50,50,3),activation='relu'))
# # model.add(MaxPool2D(pool_size=(2,2)))
# # model.add(Flatten())
# # model.add(Dense(units=128,activation='relu'))
# # model.add(Dense(units=1,activation='sigmoid'))
# # model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# # model.summary()
# # model.fit_generator(training_set,epochs=25)
# # accuracy_train=model.evaluate_generator(training_set)
# # print(accuracy_train)
# # test_set=training_datagen.flow_from_directory('./origin_data/test',target_size=(50,50),batch_size=32,class_mode='binary')
# # accuracy_train=model.evaluate_generator(test_set)
# # print(accuracy_train)
# from keras.models import load_model
# model=load_model('catordog_model.h5')
# import matplotlib as mlp
# mlp.rcParams['font.family']='Arial'
# mlp.rcParams['axes.unicode_minus']=False
# from matplotlib import pyplot as plt
# from keras.preprocessing.image import load_img,img_to_array
# a=[i for i in range(1,10)]
# fig=plt.figure(figsize=(10,10))
# for i in a:
#     img_name='origin_data/test/'+str(i)+'.jpg'
#     img_ori=load_img(img_name,target_size=(50,50))
#     img=img_to_array(img_ori)
#     img=img.astype('float32')/255
#     img=img.reshape(1,50,50,3)
#     predict=model.predict(img)
#     img_ori=load_img(img_name,target_size=(250,250))
#     plt.subplot(3,3,i)
#     plt.imshow(img_ori)
#     plt.title('predict dog' if predict>0.5 else 'predict cat')
# plt.show()
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
img_name='origin_data/test/'+'1.jpg'
img_ori=load_img(img_name,target_size=(50,50))
img=img_to_array(img_ori)
model_vgg=VGG16(weights='imagenet',include_top=False)
x1=np.expand_dims(img,axis=0)
x1=preprocess_input(x1)


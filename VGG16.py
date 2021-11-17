from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
model_vgg=VGG16(weights='imagenet',include_top=False)
def modelProcess(img_path,model):
    img_ori=load_img(img_path,target_size=(224,224))
    img=img_to_array(img_ori)
    x1=np.expand_dims(img,axis=0)
    x1=preprocess_input(x1)
    x1_vgg=model.predict(x1)
    x1_vgg=x1_vgg.reshape(1,25088)
    return x1_vgg
import os
folder ="origin_data/data_vgg/cat"
dirs=os.listdir(folder)
img_path=[]
for i in dirs:
    if os.path.splitext(i)[1]=='.jpg':
        img_path.append(i)
img_path=[folder+"//"+i for i in img_path]
feature1=np.zeros([len(img_path),25088])
for i in range(len(img_path)):
    feature_i=modelProcess(img_path[i],model_vgg)
    print('preprocessed:',img_path[i])
    feature1[i]=feature_i
folder ="origin_data/data_vgg/dog"
dirs=os.listdir(folder)
img_path=[]
for i in dirs:
    if os.path.splitext(i)[1]=='.jpg':
        img_path.append(i)
img_path=[folder+"//"+i for i in img_path]
feature2=np.zeros([len(img_path),25088])
for i in range(len(img_path)):
    feature_i=modelProcess(img_path[i],model_vgg)
    print('preprocessed:',img_path[i])
    feature2[i]=feature_i
y1=np.zeros(300)
y2=np.ones(300)
X=np.concatenate((feature1,feature2),axis=0)
y=np.concatenate((y1,y2),axis=0)
y=y.reshape(-1,1)
print(X.shape,y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=50)
print(X_train.shape,X_test.shape,X.shape)
from keras.models import Sequential
from keras.layers import Dense
#卷积神经网络
model = Sequential()
model.add(Dense(units=10,activation='relu',input_dim=25088))
model.add(Dense(units=1,activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=50)
model.save('catordog_vgg_model.h5')
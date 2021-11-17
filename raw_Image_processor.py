import glob
import shutil
import os

#数据集目录
path = "train"
#训练集目录
train_path ='origin_data/train'
#测试集目录
test_path = path+'/test'

# 将某类图片移动到该类的文件夹下
def img_to_file(path):
    print("=========开始移动图片============")
    #如果没有dog类和cat类文件夹，则新建
    if not os.path.exists(path+"/dog"):
            os.makedirs(path+"/dog")
    if not os.path.exists(path+"/cat"):
            os.makedirs(path+"/cat")
    print("共：{}张图片".format(len(glob.glob('origin_data/train/*.jpg'))))
    #通过glob遍历到所有的.jpg文件
    for imgPath in glob.glob(path+"/*.jpg"):
        #print(imgPath)
        #使用/划分
        img=imgPath.strip("\n").replace("\\","/").split("/")
        #print(img)
        #将图片移动到指定的文件夹中
        if img[-1].split(".")[0] == "cat":
            shutil.move(imgPath,path+"/cat")
        if img[-1].split(".")[0] == "dog":
            shutil.move(imgPath,path+"/dog")
img_to_file(train_path)
print("训练集猫共：{}张图片".format(len(glob.glob('./data_vgg'+"/cat/*.jpg"))))
print("训练集狗共：{}张图片".format(len(glob.glob(train_path+"/dog/*.jpg"))))

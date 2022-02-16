# TIGER DETECTION MODEL


## AIM AND OBJECTIVES

## Aim

  To create a Tiger detection system which will detect Tigers and inform the authorities about its current location.
 
## Objectives

• The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting using the camera module on the device.
    
• Using appropriate datasets for recognizing and interpreting data using machine learning.
    
• To show on the optical viewfinder of the camera module whether the animal is Tiger or not and then inform about it’s location in realtime.

## Abstract

• An animal is classified based on wheteher it is Tiger or not and is detected by the live feed from the system’s camera.
    
•  We have completed this project on jetson nano which is a very small computational device.
    
• A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected.
    
• One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier.
    
• Tiger tracking has become of great importance so as to avoid the tragedy that occured which resulted in the death of 5 people after the rewilding of the Tigress Avni .
    
• Tracking Tigers in real time not only Protects Human Beings and live stock from the attack of Tiger it also protects Tigers from becoming Man Eaters and then getting killed because of that.

## Introduction

• This project is based on a Tiger detection model with modifications. We are going to implement this project with Machine Learning and this project can be         even run on jetson nano which we have done.
    
• This project can also be used to gather information about location of Tigers in a given area.
    
• The Tigers can even be further classified into Baby tiger, Adult Tiger based on the image annotation we give in roboflow. 
    
• Tiger detection sometimes becomes difficult as Tigers are hidden in forest and gets even harder in summer as Tigers Hide and the ground as well as trees become of same color for the model to detect. However, training in Roboflow has allowed us to crop images and also change the contrast of certain images to match the time of day for better recognition by the model.
    
• Neural networks and machine learning have been used for these tasks and have obtained good results.
    
• Machine learning algorithms have proven to be very useful in pattern recognition and classification, and hence can be used for Tiger detection as well.


## Literature Review

    • More than 100,000 tigers ranged across Asia a century ago, from the Indian subcontinent to the Russian Far East. Today they are endangered, with only about 4,000 tigers left in the wild..
   
    • In India, one study estimated that widening highways, along with unplanned development, would increase tiger extinction risks within protected areas by 56% over 100 years. The growing network of transportation infrastructure in Asia could therefore be disastrous for tigers.
   
    • Wildlife Conservation Society researcher Ullas Karanth says all the methods used to count tigers, historically and now, have been poor. And it's not a surprise, because tigers are extremely rare. They exist at very low population densities on the ground. They are wide-ranging. They are very secretive. They avoid people. So, they're not an easy species to count.

    • He further adds people have used naive methods of different kinds, using just tracks in different manners in India, Russia, and Nepal, and subsequent to 2005, some attempts to use camera traps, which I had actually developed in the '90s. It took about 20 years before it came into more widespread practice in the tiger world. Even now, the camera traps, used properly, can give good, really good  counts and estimates for small areas where tigers are found in high densities – the so-called source populations. But when you're looking at large regions, whole states, entire landscapes or countries, you can't put camera traps out on that scale. It's not practical. .

    • The best way to recognize Tigers is from photographs. Each individual is unique. And this is not special to tigers: This is true of leopards, zebras, many, many naturally-marked animals. The coat patterns are so characteristic, almost like human fingerprints, and you can identify an animal without any doubt. .
 
    • Automated cameras that are scattered across the park, or a reserve, and which are tripped by the tigers themselves and which our model will use to detect them forms the basis of this project.

## Jetson Nano Compatibility

   • The power of modern AI is now available for makers, learners, and embedded developers everywhere.
   
   • NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts.
   
   • Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training.
   
   • NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK.
  
   • In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.


## Jetson Nano 2gb

![Jetson Nano](https://github.com/f-ate/Helmet-Detection/blob/main/IMG_20220125_115121.jpg)



 ## Proposed System
 
    1. Study basics of machine learning and image recognition.
   
   2. Start with implementation
   
         • Front-end development
   
         • Back-end development
   
   3. Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine learning to identify whether the object on viewfinder is Tiger or not.
   
   4. use datasets to interpret the object and suggest whether the object is Tiger or not.



## Methodology

The Tiger detection system is a program that focuses on implementing real time Tiger detection.

It is a prototype of a new product that comprises of the main module:

Tiger detection and then showing on viewfinder whether the object is Tiger or not.

### Tiger Detection Module

## This Module is divided into two parts:

1] Tiger detection

   • Ability to detect the location of object in any input image or frame. The output is the bounding box coordinates on the detected object.
   
   • For this task, initially the Dataset library Kaggle was considered. But integrating it was a complex task so then we just downloaded the images from gettyimages.ae and google images and made our own dataset.
   
   • This Datasets identifies object in a Bitmap graphic object and returns the bounding box image with annotation of object present in a given image.
   
2] Classification Detection

   • Classification of the object based on whether it is Tiger or not.
   
   • Hence YOLOv5 which is a model library from roboflow for image classification and vision was used.
   
   • There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward.
   
   • YOLOv5 was used to train and test our model for 149 epochs and achieved an accuracy of approximately 91%.    

 

## Installation
'''
sudo apt-get remove --purge libreoffice*

sudo apt-get remove --purge thunderbird*

sudo fallocate -l 10.0G /swapfile1

sudo chmod 600 /swapfile1

sudo mkswap /swapfile1

sudo vim /etc/fstab

#################add line###########

/swapfile1 swap defaults 0 0
'''
'''
vim ~/.bashrc
#############add line #############

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

sudo apt-get update

sudo apt-get upgrade
'''
'''
################pip-21.3.1 setuptools-59.6.0 wheel-0.37.1#############################

sudo apt install curl

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

sudo python3 get-pip.py

sudo apt-get install libopenblas-base libopenmpi-dev

sudo apt-get install python3-dev build-essential autoconf libtool pkg-config python-opengl python-pil python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-

tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev libssl-dev 

libpq-dev python-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev libfreetype6-dev python3-dev
'''
'''

vim ~/.bashrc

####################### add line #################### 

export OPENBLAS_CORETYPE=ARMV8

source ~/.bashrc

sudo pip3 install pillow

curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl

mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl

sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

sudo python3 -c "import torch; print(torch.cuda.is_available())"

git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
'''
'''
cd torchvision/

sudo python3 setup.py install

cd

git clone https://github.com/ultralytics/yolov5.git

cd yolov5/

sudo pip3 install numpy==1.19.4

history 
'''
'''
#####################comment torch,PyYAML and torchvision in requirement.txt##################################

sudo pip3 install --ignore-installed PyYAML>=5.3.1

sudo pip3 install -r requirements.txt

sudo python3 detect.py

sudo python3 detect.py --weights yolov5s.pt --source 0
'''
'''
#############################################Tensorflow######################################################

sudo apt-get install python3.6-dev libmysqlclient-dev

sudo apt install -y python3-pip libjpeg-dev libcanberra-gtk-module libcanberra-gtk3-module

pip3 install tqdm cython pycocotools
'''
############# https://developer.download.nvidia.com/compute/redist/jp/v46/tensorflow/tensorflow-2.5.0%2Bnv21.8-cp36-cp36m-linux_aarch64.whl ######
'''
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

sudo apt-get install python3-pip

sudo pip3 install -U pip testresources setuptools==49.6.0

sudo pip3 install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython 

pkgconfig

sudo env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==3.1.0

sudo pip3 install -U cython

sudo apt install python3-h5py

sudo pip3 install #install downloaded tensorflow(sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow)

python3

import tensorflow as tf

tf.config.list_physical_devices("GPU")

print(tf.reduce_sum(tf.random.normal([1000,1000])))

#######################################mediapipe##########################################################

git clone https://github.com/PINTO0309/mediapipe-bin

ls

cd mediapipe-bin/

ls

./v0.8.5/numpy119x/mediapipe-0.8.5_cuda102-cp36-cp36m-linux_aarch64_numpy119x_jetsonnano_L4T32.5.1_download.sh

ls
sudo pip3 install mediapipe-0.8.5_cuda102-cp36-none-linux_aarch64.whl 
'''

Images






Images show the various classes of images with annotations
## Demo




https://user-images.githubusercontent.com/97509895/154219435-16ba1225-ef7d-408d-b778-04fc646f2aef.mp4






## Advantages

   • The Tiger Detection model can can be used instead of simply using cameras on trees as it takes 12 to 24 hours for reviewing of the footage manually while the Tiger Detection system will do the detection in seconds.
   
   • The Tiger Detection model is even better than collar system which sometimes can be fatal to the Tigers as it may lead to infection and hence become counter productive to the protection of Tigers.
   
   • The collar doesn’t come cheap as each collar is fitted with GPS and VHF which cost around 3.50 lakhs and when the Tigers fight or hunt the collars may get damaged also the battery might need replacing from time to time hence our model of Tiger detection is a cost effective way of determining the location of Tigers.
   
   • It can convey to the Authorities in real time about the detection of Tiger in a particular area and thus will lead to a rapid response of authorities.
   
   • As it is completely automated and low maintenance the cost drops significantly.
   
   • It can work around the clock and therefore leads to increase in the number of instances of Tiger spotting.

## Application

    • Detects Tigers in a given image frame or viewfinder using a camera module and informs about it’s current location to the authorities.
   
   • Can be used in places where cameras are already installed by sending a live feed from cameras toward the server where the model will detect and then notify the authorities.
   
   • Can be used as a refrence for other ai models based on Animal detection.

## Future Scope

  • As we know technology is marching towards automation, so this project is one of the step towards automation.
  
  • Thus, for more accurate results it needs to be trained for more images, and for a greater number of epochs.
  
  • Tigers can even be classified based on their stripes as each stripe of individual Tiger is different just as figerprints of individuals and thus help us to maintain a healthy population of Tigers by stopping inbreeding.
  
  • As more Tigers are born they can be individually tracked and thus can be protected with ease and when it seems that the Tiger is moving towards a human settlement the authorities can then take appropriate action.
  
  • It can be used to even classify various endangered species apart from Tigers and help conserve them.




## Conclusion

    • In this project our model is trying to detect Tigers and then informing about their location in real time to authorities as we have specified in Roboflow.
    
    • The model solves the problem of protection of endangered species like Tigers by tracking them and then informing of their whereabouts to the respective authorities.
    
    • Better tracking leads to better conservation of species by tracking the number of Tigers in the wild and thus taking appropriate action by the authorities based on their current location that is whether they are close to human settlement or not.






## Reference

1] Roboflow: - https://roboflow.com/
2] Datasets or images used: https://www.gettyimages.ae/search/2/image?family=creative&phrase=tiger
3] Google images


## Articles: -

    1. https://www.hindustantimes.com/india/tigress-dies-in-madhya-pradesh-after-radio-collar-infection/story-Cgt2UCfTSAWRmBIlgU7IjP.html
    2. https://www.scientificamerican.com/podcast/episode/tiger-tiger-being-tracked/
    3. https://www.hindustantimes.com/india-news/missing-tigers-trigger-debate-on-usability-of-tracking-devices/story-dP3aN782qoLrRRozF8qhNJ.html
    4. https://theworld.org/stories/2021-05-03/gps-tracking-could-help-tigers-and-traffic-coexist-across-asia

# Signature-verification-using-deep-learning
Using SigComp'11 dataset for signature verification

## Getting Started 
Before starting with this tutorial, you should already have latest version of tensorflow and Keras installed(with Python 3). I have done all the work in Google Colab, which provides GPU for limited time. This repository contains 3 files final_code.ipynb, model.ipynb, trial.ipynb. model and trial ipynb files contains the detailed work, which are sequentially and neatly covered again in final_code.ipynb. Should only refer to this file.

## Getting Started with Colab
Colab uses your Google drive for loading and storing data/model. I have covered how to mount drive to Google Compute Engine in the file. During execution of the 1st block(for mounting the drive), you need to give access to Colab.

## Data Specifaction
I have used SigComp'11 dataset. The details of dataset are provided on http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011) During unzipping, you need to provide the pass-key provided on the site.

## Data Pre-processing
I have shown have to generate dataset arrays from the folder structure.

## Models
I have used a novel architecture and InceptionV3 model provided by Keras for transfer learning. In this model, I am not training the weights of pre-trained model. I am using multi-class classification to distinguish between the 10 authors.
```
for layer in base_model.layers:
    layer.trainable = False
```
This part of code makes the weights un-trainable for subsequent usage.

Novel architecture:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_95 (Conv2D)           (None, 544, 1110, 16)     3088      
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 272, 555, 16)      0         
_________________________________________________________________
conv2d_96 (Conv2D)           (None, 257, 540, 16)      65552     
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 128, 270, 16)      0         
_________________________________________________________________
conv2d_97 (Conv2D)           (None, 113, 255, 16)      65552     
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 56, 127, 16)       0         
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 28, 63, 16)        0         
_________________________________________________________________
conv2d_98 (Conv2D)           (None, 13, 48, 16)        65552     
_________________________________________________________________
max_pooling2d_9 (MaxPooling2 (None, 6, 24, 16)         0         
_________________________________________________________________
max_pooling2d_10 (MaxPooling (None, 3, 12, 16)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 576)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 100)               57700     
_________________________________________________________________
dense_4 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_5 (Dense)              (None, 10)                510       
=================================================================
Total params: 263,004
Trainable params: 263,004
Non-trainable params: 0
```

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
conv_activation = 'relu'
deep_activation = 'relu'

input_shape = (551, 1117, 3)
num_classes = 10

model = Sequential()

model.add(Conv2D(16,kernel_size=(8, 8),strides=(1, 1),activation=conv_activation,input_shape=input_shape,data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (16, 16), activation=conv_activation))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (16, 16), activation=conv_activation))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (16, 16), activation=conv_activation))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# model.add(Dense(100, activation=deep_activation))
model.add(Dense(100, activation=deep_activation))
model.add(Dense(50, activation=deep_activation))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])  

model.summary()
```

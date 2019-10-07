# Signature-verification-using-deep-learning
Using SigComp'11 dataset for signature verification (With Siamese network and triplet loss) 

## Getting Started 
Before starting with this tutorial, you should already have latest version of tensorflow and Keras installed(with Python 3). I have done all the work in Google Colab, which provides GPU for limited time. This repository contains 5 files- mycode.ipynb,siamese_net.h5, final_code.ipynb, model.ipynb, trial.ipynb. model and trial.ipynb files contains the detailed work, which are sequentially and neatly covered again in final_code.ipynb. The data and model parameter files are omitted, you have to train it yourselves. Should only refer to this file. For siamese network and triplet loss refer to mycode.ipynb. The model is saved in siamese_net.h5.

## Resources
I am compiling all the resources here-
#### http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011)
#### https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d
#### http://cs231n.stanford.edu/reports/2016/pdfs/276_Report.pdf
#### https://www.coursera.org/lecture/convolutional-neural-networks/siamese-network-bjhmj.
#### https://arxiv.org/abs/1503.03832
#### https://github.com/adambielski/siamese-triplet
#### https://thelonenutblog.wordpress.com/2017/12/18/what-siamese-dreams-are-made-of/
#### https://keras.io/applications/#nasnet
#### https://keras.io/applications/#documentation-for-individual-models  

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
## Siamese networks and Triplet loss
I have used siamese training dor generating embeddings from images and triplet loss as loss function. For, more information about siamese network and triplet loss, please refer to -
https://www.coursera.org/lecture/convolutional-neural-networks/siamese-network-bjhmj. 
For paper-
https://arxiv.org/abs/1503.03832
##### Here, I have used offline siamese data generation as the the dataset I am using is offline dataset.
##### PS: This was a joke and offline siamese data generation and offline dataset provided have no relation whatsoever. If you didn't get this one, you should consider revisiting the dataset page and siamese learning tutorial! I checked again, and my strategy does not classify as offline or online siamese triplet selection.
### Model Architecture
As siamese training with triplet loss requires quite a lot of memory, I tried searching for architectures which would have lesser parameters. I have chosen one of the lesser parameter model- NASnetMobile. For details of model architecture and comparison of models, plese refer to this Keras documentation - 
https://keras.io/applications/#nasnet and 
https://keras.io/applications/#documentation-for-individual-models  

### Training
I have added few Conv and max-pooling layers for dimentionality accordance. In the end, I am generating 1024 length vector.
```
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_10 (InputLayer)           (None, 551, 1117, 1) 0                                            
__________________________________________________________________________________________________
input_11 (InputLayer)           (None, 551, 1117, 1) 0                                            
__________________________________________________________________________________________________
input_12 (InputLayer)           (None, 551, 1117, 1) 0                                            
__________________________________________________________________________________________________
model_5 (Model)                 (None, 1, 1024)      5352642     input_10[0][0]                   
                                                                 input_11[0][0]                   
                                                                 input_12[0][0]                   
__________________________________________________________________________________________________
concatenate_7 (Concatenate)     (None, 3, 1024)      0           model_5[1][0]                    
                                                                 model_5[2][0]                    
                                                                 model_5[3][0]                    
==================================================================================================
Total params: 5,352,642
Trainable params: 1,082,926
Non-trainable params: 4,269,716
```
As you might have guessed, only storing the 'Model' is required from above architecture. Please refer to code for details. The 3 input layers are for anchor, positive and negative image respectively. I am compiling the model with triplet loss
```
mod.compile(optimizer='adam',loss=triplet_loss,metrics=['accuracy'])
```
which I have deviced as per Convolutional neural networks course, deeplearning.ai. Just for the demonstration and due to being rather computationally extensive, I have only trained the model for 5 epochs. As per the defination of the model, more the loss, better the features learned.

## Comparison
Posing verification task as multi-class classification, I have trained my small baseline model and InceptionV3 with transfer learning. InceptionV3 model gives much better performance than basline, as expected. But without regularization, it overfits, which is evident from validation. I haven't tested of my siamese network with other approches yet, will release it as soon as I get onto to it.

## Few last words...
Thank you for staying with me till the end. If you liked my repo and the work I have done, feel free to star this repo and follow me. I will make sure to bring out awesome deep learning projects like this in the future. Until the next time, **サヨナラ!** 
###### PS: I am an anime fan ;)

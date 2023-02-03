# PerGAN
 
This code was implemented based on DCGAN(https://github.com/carpedm20/DCGAN-tensorflow) which proposed in ''Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks''. 

# Prerequisites

Python 2.7 
Tensorflow 1.12.0
Numpy
SciPy
pillow

# Dataset

Offical Dataset
As many existing work has done, we used the public MSRA-10K dataset to train the proposed PerGAN. Please put the training data into the datasets/train folder before train the model.

Preprocessed dataset
We have done the data augment including flip,noisy and translate to enlarge the training dataset. We totally got 80K images (can be downloaded from google driver) to train the model. We provided the preprocessed datasets in []. After dowloading the dataset it should be proprocessed including crop,flip and rotation put into the folder path as specified in the datasets

# Training
Download the dataset and pre-processed it，then put the pre-processed images into the folder datasets/train.
Download the VGG16.npy (https://mega.nz/file/YU1FWJrA#O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) and put it into the PerGAN folder.
Then train the model with the following commands:
$ python main.py  (default settings)
$ python main.py --dataset dataset_name --input_height= XX --output_height=XX --train 

# Testing
Download the pre-trained model(), put it into the folder checkpoints/Sal-1-256.
Download the VGG16.npy (https://mega.nz/file/YU1FWJrA#O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM) and put it into the PerGAN folder.
Put the testing samples into the folder datasets/test.
Then test the model with the following commands:
$ python main.py --phase test (default settings)
Please refer to  https://github.com/carpedm20/DCGAN-tensorflow for more information about the usage of the code.


Acknoledgement
We thank for the open-sourced DCGAN (https://github.com/carpedm20/DCGAN-tensorflow) and VGG （https://github.com/machrisaa/tensorflow-vgg） .

Author
caixiaoxu90@163.com

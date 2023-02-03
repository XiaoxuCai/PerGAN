"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import tensorflow as tf
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
import scipy.io as scio
import h5py
import vgg16
import imageio

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix


def load_cube(image_path, flip=True, is_test=False):
    #img_A, img_B = load_image(image_path)
    #img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

    try:
        input_img = h5py.File(image_path)
        input_img = np.transpose(input_img['yy'])
        img_AB = input_img[:,:,0:4]
    except:
        input_img = scio.loadmat(image_path)
        img= input_img['yy']
        img_AB = img[:,:,0:4]
    return img_AB


def save_images(images, size, image_path):
    #return imsave(inverse_transform(images), size, image_path)
    return imsave(images, size, image_path)

def save_cube(images,image_path):
    return scio.savemat(image_path,{"face":images[0,:,:,:]})

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    #return scipy.misc.imsave(path, merge(images, size))
    return imageio.imsave(path, merge(images, size))

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


def map_cat(images):
    mask = tf.concat([images[:,:,:,3:4], images[:,:,:,3:4]], 3)
    mask = tf.concat([mask, images[:,:,:,3:4]], 3)
    map = tf.multiply(images[:,:,:,:3], mask)
    return map

def gram(feature_maps):
    fea_shape  = tf.shape(feature_maps)
    #filter_size  = fea_shape[1]*fea_shape[2]*fea_shape[3]
    feature_set = tf.reshape(feature_maps, [fea_shape[0], fea_shape[1]*fea_shape[2], fea_shape[3]])
    gram_matrix = tf.matmul(feature_set, feature_set, transpose_a = True)
    return gram_matrix
	
	
def semantic_loss(sal_target, sal_output, filter_size):
    gram_target1 = gram(get_feature_conv1_2(sal_target))
    gram_output1 = gram(get_feature_conv1_2(sal_output))
    gram_target2 = gram(get_feature_conv2_2(sal_target))
    gram_output2 = gram(get_feature_conv2_2(sal_output))
    gram_target3 = gram(get_feature_conv3_3(sal_target))
    gram_output3 = gram(get_feature_conv3_3(sal_output))
    gram_target4 = gram(get_feature_conv4_3(sal_target))
    gram_output4 = gram(get_feature_conv4_3(sal_output))

    sum_squared_difference1 = tf.reduce_sum(tf.square(gram_output1 - gram_target1))
    sum_squared_difference2 = tf.reduce_sum(tf.square(gram_output2 - gram_target2))
    sum_squared_difference3 = tf.reduce_sum(tf.square(gram_output3 - gram_target3))
    sum_squared_difference4 = tf.reduce_sum(tf.square(gram_output4 - gram_target4))


    loss_contribution1 = 10*sum_squared_difference1/(64*224*224)/(64*224*224)
    loss_contribution2 = 10*sum_squared_difference2/(128*112*112)/(128*112*112)
    loss_contribution3 = 10*sum_squared_difference3/(256*56*56)/(256*56*56)
    loss_contribution4 = 10*sum_squared_difference4/(512*28*28)/(512*28*28)
    #loss_contribution1 = tf.losses.mean_squared_error(labels = gram_target1, predictions = gram_output1)
    #loss_contribution2 = tf.losses.mean_squared_error(labels = gram_target2, predictions = gram_output2)
    #loss_contribution3 = tf.losses.mean_squared_error(labels = gram_target3, predictions = gram_output3)
    #loss_contribution4 = tf.losses.mean_squared_error(labels = gram_target4, predictions = gram_output4)

    return (loss_contribution1 + loss_contribution2 + loss_contribution3 + loss_contribution4)/4

def get_feature_conv1_2(images):
    img = map_cat(images)
    vgg = vgg16.Vgg16()
    vgg.build(img)
    feature_maps = tf.contrib.layers.batch_norm(vgg.conv1_2)
    return feature_maps

def get_feature_conv2_2(images):
    img = map_cat(images)
    vgg = vgg16.Vgg16()
    vgg.build(img)
    feature_maps = tf.contrib.layers.batch_norm(vgg.conv2_2)
    return feature_maps

def get_feature_conv3_3(images):
    img = map_cat(images)
    vgg = vgg16.Vgg16()
    vgg.build(img)
    feature_maps = tf.contrib.layers.batch_norm(vgg.conv3_3)
    return feature_maps

def get_feature_conv4_3(images):
    img = map_cat(images)
    vgg = vgg16.Vgg16()
    vgg.build(img)
    feature_maps = tf.contrib.layers.batch_norm(vgg.conv4_3)
    return feature_maps



def content_loss(saliency_target, saliency_output):
    target = get_feature_conv1_2(saliency_target)
    output = get_feature_conv1_2(saliency_output)

    loss_contribution = tf.losses.mean_squared_error(labels = target, predictions = output)

    return loss_contribution

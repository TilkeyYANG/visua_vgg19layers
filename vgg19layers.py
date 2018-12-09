# -*- coding: utf-8 -*-
'''
Created on Mon Oct 15 22:30:51 2018

@author: TilkeyYang
'''

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import scipy.io



# =============================================================================
# Fonctions
# =============================================================================

def conv_layer(input, weights, bias):
    # Because we're using VGG19, we use strides(w=1, h=1)
    # Use tf.constant to change format
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
                        padding='SAME')
    return tf.nn.bias_add(conv, bias)

def pool_layer(input):
    # Ksize - Our pooling window is batch=1, w=h=2, channel=1
    # Strides - Our pooling step is batch=1, w=h=2, channel=1
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                          padding='SAME')

# "Minus-Mean" pre process
def preprocess(image, mean_pixel):
  return image - mean_pixel

# To cancel the preprocess
def unprocess(image, mean_pixel):
  return image + mean_pixel

print('Functions OK')



# =============================================================================
# Directory and import
# =============================================================================

# Enter directory to change dir
os.chdir('.') # Enter your directory
#os.chdir('D:/Github/Practicing/visua_vgg19layers') #This one is my dir
cwd = os.getcwd()
print('Working Directory:', cwd)
os.makedirs(cwd + '/output', exist_ok=True)

# Import model and image
vgg_path = cwd + '/model/imagenet-vgg-verydeep-19.mat' 
img_path = cwd + '/data/CatTest1.jpg' 
input_image = scipy.misc.imread(img_path).astype(np.float)



# =============================================================================
# NetWork VGG19
# =============================================================================

def net(data_path, input_image):
  
    # Given model layers
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',      
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
    
    # Load the model mat
    data = scipy.io.loadmat(data_path)
    
    # Get mean value of input_image
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0,1))
    print('mean_pixel = ', mean_pixel)
    
    # Get weights of each layer from model
    weights = data['layers'][0]
    print('weights = ', weights)
    
#   Analysing Model:
#   After tring lots of weights[0]...[0].shape, we found out that 
#   weights[0][0][0][0][0].shape = (2,)
#   Which means weights[i_layer][0][0][0][0] pointed our w and b of the layer
# 
#   # Example:
#  
#   # conv1_1 weight
#   print(weights[5][0][0][0][0][0].shape)
#   (3, 3, 3, 64): width = 3; height = 3; in_channels = 3; out_channels = 64; 
#   will give us 64 feature figures
#
#   # conv1_2 bias
#   print(weights[2][0][0][0][0][1].shape)
#   (1, 64)
    
    
    # Define a net dict to memorize w and b for each layer
    net = {}
    
    current = input_image
    
    for i, name in enumerate(layers):  
      # Just reading the 4 first chars to find out the layer type:
      kind = name[:4]
      
      if kind == 'conv':
        # Read directly to get Kernel(weight) and bias of layer[i]
        kernels, bias = weights[i][0][0][0][0]
        # Transfert from .mat format (0 1 2 3) into tensorflow format (1 0 2 3)
        # Matconvnet: weights = [width, height, in_channels, our_channels]
        # Tensorflow: weights = [height, width, in_channels, our_channels]
        kernels = np.transpose(kernels, (1, 0, 2, 3))     
        # Reshape bias (1, 64) to be compatible with the original shape (64)
        bias = bias.reshape(-1)
        # Forward Propagation: calculate conv by using w and b
        current = conv_layer(current, kernels, bias)
      
      elif kind == 'relu':
        current = tf.nn.relu(current)
        
      elif kind == 'pool':
        current = pool_layer(current)  
      
      # Net[name] is in the dict, using name of layers
      # Current is the result of the current forward propagation
      net[name] = current
      print('Current net[name]', net[name])
      
    assert len(net) == len(layers)
    
    # Finally, we return the dictionnary net
    # -- in order to use enum all layers and figures in the net
    return net, mean_pixel, layers
  
print('Network OK')



# =============================================================================
# Session
# =============================================================================

# Reshape input_img
shape = (1, input_image.shape[0], input_image.shape[1], input_image.shape[2]) 

with tf.Session() as sess: 
  # Waiting for enum(layers) features' feed_dicts
  image = tf.placeholder('float', shape=shape) 
  
  # Get nets(dictionnary); mean_pixel for the preprocess; all layers for 
  nets, mean_pixel, all_layers = net(vgg_path, image) 
  layers = all_layers
  
  # Preprocessing our input_img using "Minus-Mean"
  input_image_pre = np.array([preprocess(input_image, mean_pixel)]) 
   
  # To visualize the Propagation result of each layer
  for i, layer in enumerate(layers): 
    print('[%d/%d] %s' % (i+1, len(layers), layer))
    # Send image to placeholder, then goto net()
    features = nets[layer].eval(feed_dict={image: input_image_pre}) 
    print('Type of ‘features’ is ', type(features)) 
    print('Shape of ‘features’ is ', (features.shape,)) 

    # Plot response
    if 1: plt.figure(i+1, figsize=(5, 6)) 
    # Show heatmap of figure
    plt.matshow(features[0, :, :, 0], cmap=plt.cm.magma, fignum=i+1) 
    plt.title(''+layer) 
    plt.colorbar() 
    # Auto save figures
    plt.savefig(cwd + '/output/cat%d.jpg'%(i+1)) 
    plt.show()



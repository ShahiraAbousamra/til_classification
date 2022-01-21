# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

weight_decay_val = 0.0;

def vgg_arg_scope(weight_decay=0.0005):  
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  global weight_decay_val;
  weight_decay_val = weight_decay
  #print('weight_decay_val =', weight_decay_val)
  #print('weight_decay =', weight_decay)
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l1_regularizer(weight_decay_val),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def vgg_a(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='vgg_a',
          fc_conv_padding='VALID',
          global_pool=False,
          weight_decay = 0.0005):
  """Oxford Net VGG 11-Layers version A Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  global weight_decay_val;
  weight_decay_val = weight_decay
  #print('weight_decay_val =', weight_decay_val)
  #print('weight_decay =', weight_decay)
  with tf.variable_scope(scope, 'vgg_a', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8', weights_regularizer=slim.l1_regularizer(weight_decay_val))
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_a.default_image_size = 224


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           dropout_keep_prob_inner=1.0,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False,
           weight_decay = 0.0005,
           input_ctypes=(None,None,None),
           version = '0' 
           ):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  global weight_decay_val;
  weight_decay_val = weight_decay
  input_ctypes_fc6 = input_ctypes[0];
  input_ctypes_fc7 = input_ctypes[1];
  input_ctypes_fc8 = input_ctypes[2];
  #print('weight_decay_val =', weight_decay_val)
  #print('weight_decay =', weight_decay)
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      #if(dropout_keep_prob_inner < 1):
      #  net = slim.dropout(net, dropout_keep_prob_inner, is_training=is_training, scope='dropout1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      if(dropout_keep_prob_inner < 1):
        net = slim.dropout(net, dropout_keep_prob_inner, is_training=is_training, scope='dropout2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      final_1 = net;        
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      if(dropout_keep_prob_inner < 1):
        net = slim.dropout(net, dropout_keep_prob_inner, is_training=is_training, scope='dropout3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      if(dropout_keep_prob_inner < 1):
        net = slim.dropout(net, dropout_keep_prob_inner, is_training=is_training, scope='dropout4')
      final_2 = net;        
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      if(dropout_keep_prob_inner < 1):
        net = slim.dropout(net, dropout_keep_prob_inner, is_training=is_training, scope='dropout5')

      if(not input_ctypes_fc6 is None):  
        #input_ctypes2 = tf.zeros((net.shape[0], net.shape[1], net.shape[2], input_ctypes.shape[1]));
        #input_ctypes2 = tf.add(input_ctypes2, input_ctypes);
        net = tf.concat([net, input_ctypes_fc6], axis=-1);
      # Use conv2d instead of fully_connected layers.
      if(version == 'add_prev_feat_w1x1'):
        final_1 = slim.avg_pool2d(final_1, [9, 9], stride=7, padding='VALID', scope='pool3_9x9')
        final_2 = slim.avg_pool2d(final_2, [2, 2], stride=2, padding='VALID', scope='pool4_2x2')
        net = tf.concat([net, final_1, final_2], axis=-1);
        net = slim.conv2d(net, 1024, [1, 1], scope='Conv2d_pre_1x1', weights_regularizer=slim.l1_regularizer(weight_decay_val))
              
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      if(not input_ctypes_fc7 is None):  
        net = tf.concat([net, input_ctypes_fc7], axis=-1);
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      end_points[sc.name + '/fc7'] = net
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        if(not input_ctypes_fc8 is None):  
            net = tf.concat([net, input_ctypes_fc8], axis=-1);
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_16.default_image_size = 224


def vgg_19(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           fc_conv_padding='VALID',
           global_pool=False,
           weight_decay = 0.0005):
  """Oxford Net VGG 19-Layers version E Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0 or
      None).
    end_points: a dict of tensors with intermediate activations.
  """
  global weight_decay_val;
  weight_decay_val = weight_decay
  #print('weight_decay =', weight_decay)
  with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5', weights_regularizer=slim.l1_regularizer(weight_decay_val))
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_19.default_image_size = 224

# Alias
vgg_d = vgg_16
vgg_e = vgg_19

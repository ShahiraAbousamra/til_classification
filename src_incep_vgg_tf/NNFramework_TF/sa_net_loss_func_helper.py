# author: Shahira Abousamra <shahira.abousamra@stonybrook.edu>
# created: 12.23.2018 
# ==============================================================================
import tensorflow as tf;
from enum import Enum;
import numpy as np;

class CostFuncTypes(Enum):
    CROSS_ENTROPY = 1


class CNNLossFuncHelper:

    @staticmethod
    def cost_cross_entropy(logits, labels, class_weights, n_classes):
        flat_logits = tf.reshape(logits, [-1, n_classes]);
        flat_labels = tf.reshape(labels, [-1, n_classes]);
        print(class_weights)
        print(flat_labels)
        print(flat_logits)
        return tf.abs(tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits( \
            logits=flat_logits \
            , targets=flat_labels \
            , pos_weight=class_weights \
        )));

    @staticmethod
    def cost_mse(logits, labels, class_weights, n_classes):
        flat_logits = tf.reshape(logits, [-1, n_classes]);
        flat_labels = tf.reshape(labels, [-1, n_classes]);
        print(class_weights)
        print(flat_labels)
        print(flat_logits)
        return tf.losses.mean_squared_error(\
            labels=flat_labels \
            , predictions=flat_logits \
            , weights=class_weights \
        );


# author: Shahira Abousamra <shahira.abousamra@stonybrook.edu>
# created: 12.23.2018 
# ==============================================================================
import tensorflow as tf;
from enum import Enum;


class OptimizerTypes(Enum):
    ADAM = 1
    SGD = 2


class CNNOptimizer:

    @staticmethod
    def adam_optimizer(learning_rate, cost, global_step):
        return tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08,) \
            .minimize(cost, global_step=global_step);

    @staticmethod
    def sgd_optimizer(learning_rate, cost, global_step):
        return tf.train.GradientDescentOptimizer(learning_rate) \
            .minimize(cost, global_step=global_step);


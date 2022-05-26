import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import init_ops


def build_expert(input, config, scope_name, is_train):
    l2_norm = config["l2_norm"]
    drop_rate = config["drop_rate"]
    random_seed = config["random_seed"]
    hidden_units = config["expert_dnn_hidden_units"]
    act_fn = config["expert_act"]
    with tf.variable_scope(name_or_scope=scope_name, reuse=tf.AUTO_REUSE):
        output = input
        with arg_scope(model_arg_scope(l2_weight_decay=l2_norm)):
            for layer_id, hidden_unit in enumerate(hidden_units):
                with tf.variable_scope("scope_name_{}_{}".format(scope_name, layer_id)) as layer_scope:
                    output = layers.fully_connected(
                        output, hidden_unit, get_act_fn(act_fn),
                        scope=layer_scope,
                        variables_collections=[scope_name],
                        outputs_collections=[scope_name + '_output'],
                        normalizer_fn=layers.batch_norm,
                        normalizer_params={"scale": True, "is_training": is_train}
                    )
                    output = tf.layers.dropout(output, drop_rate, training=is_train, seed=random_seed)
    return output


def build_gate(input, config, scope_name, is_train):
    l2_norm = config["l2_norm"]
    hidden_units = [config["num_expert"]]
    with tf.variable_scope(name_or_scope=scope_name, reuse=tf.AUTO_REUSE):
        output = input
        with arg_scope(model_arg_scope(l2_weight_decay=l2_norm)):
            for layer_id, hidden_unit in enumerate(hidden_units):
                with tf.variable_scope("scope_name_{}_{}".format(scope_name, layer_id)) as layer_scope:
                    output = layers.fully_connected(
                        output, hidden_unit, None,
                        scope=layer_scope,
                        variables_collections=[scope_name],
                        outputs_collections=[scope_name + '_output'],
                        normalizer_fn=layers.batch_norm,
                        normalizer_params={"scale": True, "is_training": is_train}
                    )
    return output


def cal_KLdis(p, q):
    """
        calculate KL distance
    """
    return tf.reduce_sum(tf.multiply(p, tf.log(tf.div(p, q))), 2)


def model_arg_scope(l1_weight_decay=0.0000,
                    l2_weight_decay=0.0001,
                    weights_initializer=initializers.xavier_initializer(),
                    weights_regularizer=regularizers.l1_l2_regularizer,
                    biases_initializer=init_ops.zeros_initializer()):
    with arg_scope(
            [layers.fully_connected, layers.conv2d],
            weights_initializer=weights_initializer,
            weights_regularizer=weights_regularizer(l1_weight_decay, l2_weight_decay),
            biases_initializer=biases_initializer) as arg_sc:
        return arg_sc


def get_act_fn(act_name="relu"):
    if act_name.lower() == 'relu':
        return tf.nn.relu
    elif act_name.lower() == 'tanh':
        return tf.nn.tanh
    elif act_name.lower() == 'lrelu':
        return lambda x: tf.nn.leaky_relu(x, alpha=0.01)
    else:
        return tf.nn.relu

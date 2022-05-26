import os
import sys
import numpy as np
import tensorflow as tf
from utils import build_expert, build_gate, cal_KLdis


def AEMS2(feature_input, scenario_idxs, scenario_indicators, task_idxs, task_indicators, config, is_train=False):
    """
        The core part of AEMS2 model.
    :param feature_input: tensor, outputs from embedding layer or more bottom part of your model.
    :param scenario_idxs: list of tensor,  one hot scenario ids of one data instance, if one data instance
                        follows a scenario tree like "domain 𝛼2 ➡️ channel 𝛽1, then scenario ids will be
                        [[[0, 1]], [[1, 0]]].
    :param scenario_indicators: list of tensor, if one data instance follow a scenario tree like
                        "domain 𝛼2 ➡️ channel 𝛽1, then scenario indicators will be [embedding of 𝛼2,  embedding of 𝛽1].
    :param task_idxs: list of tensor, all task idxs, the "idxs" goes same with "idxs" in "scenario idxs".
    :param task_indicators: list of tensor, all task embeddings. the "indicators" goes same with
                            "indicators" in "scenario indicators".
    :param config: string, see "config.json".
    :param is_train: bool, indicate training phase(True) or others(False).
    :return: tensor, outputs of all tasks, then with MLP tower you can get different predictions for different tasks.
             list, "specific" KL distance loss of each layer.
             list, "shared" KL distance loss of each layer.
    """

    base_config = config["base"]
    scenario_configs = config["scenario"]
    task_config = config["task"]

    # sp - specific, sh - shared
    sp_kl_loss = []
    sh_kl_loss = []

    # depth of scenario tree
    tree_depth = scenario_configs["tree_depth"]
    scenario_output = feature_input
    for i in range(tree_depth):
        scenario_config = scenario_configs[str(i + 1)]
        scenario_idx = scenario_idxs[i]
        if i == 0:
            scenario_indicator = scenario_indicators[i]
        else:
            scenario_indicator = tf.concat([scenario_indicators[:i + 1]], axis=1)
        scenario_output, layer_sp_kl_loss, layer_sh_kl_loss \
            = scenario_layers(scenario_output, scenario_config, scenario_idx, scenario_indicator, tree_depth, is_train)
        sp_kl_loss += [layer_sp_kl_loss]
        sh_kl_loss += [layer_sh_kl_loss]

    outputs, task_sp_kl_loss, task_sh_kl_loss \
        = task_layers(scenario_output, task_config, task_idxs, scenario_indicator, task_indicators, is_train)
    sp_kl_loss += [task_sp_kl_loss]
    sh_kl_loss += [task_sh_kl_loss]
    return outputs, sp_kl_loss, sh_kl_loss


def scenario_layers(scenario_input, scenario_config, scenario_idx, scenario_indicator, tree_depth, is_train):
    num_layer = scenario_config["num_layer"]
    output = scenario_input
    scenario_sp_kl_loss = []
    scenario_sh_kl_loss = []
    for i in range(num_layer):
        num_expert = scenario_config["num_expert"]
        num_gate = scenario_config["num_gate"]
        experts = tf.concat([tf.expand_dims(build_expert(output, scenario_config,
                                                         f"expert_{j}_of_layer{i}_in_depth{tree_depth}", is_train)
                                            , 1) for j in range(num_expert)], axis=1)
        gate_input = tf.concat([output, scenario_indicator], axis=1)
        gates = tf.concat([tf.expand_dims(build_gate(gate_input, scenario_config,
                                                     f"gate_{j}_of_layer{i}_in_depth{tree_depth}", is_train)
                                          , 1) for j in range(num_gate)], axis=1)

        expert_sp_idxs, experts_sh_idxs, layer_kl_loss = select_experts(scenario_idx, gates, scenario_config, is_train)
        gate_masks = create_gate_masks(expert_sp_idxs, experts_sh_idxs, num_expert)
        gates = gates * gate_masks
        gates = tf.nn.softmax(gates, axis=2)
        output = tf.squeeze((scenario_idx @ gates) @ experts, 1)
        scenario_sp_kl_loss += [layer_kl_loss[0]]
        scenario_sh_kl_loss += [layer_kl_loss[1]]

    return output, scenario_sp_kl_loss, scenario_sh_kl_loss


def task_layers(task_input, task_config, task_idxs, scenario_indicator, task_indicators, is_train):
    num_layer = task_config["num_layer"]
    num_expert = task_config["num_expert"]
    num_task = task_config["num_gate"]
    num_gate = num_task
    outputs = [task_input for _ in range(num_task)]
    task_sp_kl_loss = []
    task_sh_kl_loss = []
    for i in range(num_layer):
        for t in range(num_task):
            experts = tf.concat([tf.expand_dims(build_expert(outputs[t], task_config,
                                                             f"expert_{j}_of_layer{i}_of_task", is_train)
                                                , 1) for j in range(num_expert)], axis=1)
            gate_input = tf.concat([outputs[t], scenario_indicator, task_indicators[t]], axis=1)
            gates = tf.concat([tf.expand_dims(build_gate(gate_input, task_config,
                                                         f"gate_{j}_of_layer{i}_of_task", is_train)
                                              , 1) for j in range(num_gate)], axis=1)
            expert_sp_idxs, experts_sh_idxs, layer_kl_loss = select_experts(task_idxs[t], gates, task_config, is_train)
            gate_masks = create_gate_masks(expert_sp_idxs, experts_sh_idxs, num_expert)
            gates = gates * gate_masks
            gates = tf.nn.softmax(gates, axis=2)
            outputs[t] = tf.squeeze((task_idxs[t] @ gates) @ experts, 1)
            task_sp_kl_loss += [layer_kl_loss[0]]
            task_sh_kl_loss += [layer_kl_loss[1]]

    return outputs, task_sp_kl_loss, task_sh_kl_loss


def select_experts(idx, gates, config, is_train):
    """
    :param idx: tensor, one hot tensor indicates the scenario of one instance or optimized task
    :param gates: tensor, gate matrix before softmax, [batch_size, num_gate, num_expert]
    :param config: string
    :param is_train: boolean
    :return:
    """
    num_expert = config["num_expert"]
    num_gate = config["num_gate"]
    topK_sp = config["topK_sp"]
    topK_sh = config["topK_sh"]
    noise_scale = config["noise_scale"]

    if is_train:
        noisy_scale = noise_scale * num_expert
        gates += noisy_scale * tf.random_normal(shape=tf.shape(gates), mean=0.0, dtype=tf.float32)

    gates_matrix = tf.nn.softmax(gates, axis=1)

    gates_matrix = tf.transpose(gates_matrix, [0, 2, 1])
    special_p = tf.squeeze(idx, 1)
    shared_p = tf.ones_like(special_p) / num_gate

    special_p = tf.concat([tf.expand_dims(special_p, 1) for _ in range(num_expert)], axis=1)
    special_kldis = cal_KLdis(special_p + 1e-9, gates_matrix + 1e-9)
    special_idx = tf.math.top_k(-special_kldis, topK_sp).indices

    special_kl_loss = tf.reduce_mean(tf.batch_gather(special_kldis, special_idx))

    shared_p = tf.concat([tf.expand_dims(shared_p, 1) for _ in range(num_expert)], axis=1)
    shared_kldis = cal_KLdis(shared_p, gates_matrix)
    shared_idx = tf.math.top_k(-shared_kldis, topK_sh).indices

    shared_kl_loss = tf.reduce_mean(tf.batch_gather(shared_kldis, shared_idx))

    return special_idx, shared_idx, [special_kl_loss, shared_kl_loss]


def create_gate_masks(sp_idx, sh_idx, num_expert):
    """
        selected idx of expert will set to 1 and unselected idx of expert will set to -1e9(-∞),
        then with softmax operation, its gate scale will get closer to zero.
    """
    masks = tf.one_hot(sp_idx, num_expert) + tf.one_hot(sh_idx, num_expert)
    masks = tf.reduce_sum(masks, 1, keepdims=True)
    m = tf.ones_like(masks) * -1e9
    return tf.where(masks > 0, masks, m)

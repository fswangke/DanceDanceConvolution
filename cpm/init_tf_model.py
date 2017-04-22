from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import cpm_tf
import cPickle as pickle
import skimage
import skimage.io
import skimage.transform
import sys


def init_tf_weights(root_scope, weights):
    names_to_values = {}
    for scope, weights in weights.iteritems():
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, '%s/%s' % (root_scope, scope))
        assert len(weights) == len(variables)
        for v, w in zip(variables, weights):
            names_to_values[v.name] = w
    print(names_to_values.keys())
    return tf.contrib.framework.assign_from_values(names_to_values)


if __name__ == '__main__':
    paf_net_tf_dir = os.path.join('tf_model', 'paf_net')
    if os.path.exists(paf_net_tf_dir) is False:
        os.makedirs(paf_net_tf_dir)

    paf_net_params_file = os.path.join('paf_model', 'paf_params.pickle')
    paf_net_tf_cpt_file = os.path.join(paf_net_tf_dir, 'paf_net_tf_checkpoint')
    with open(paf_net_params_file, 'rb') as f:
        paf_net_params = pickle.load(f)

    PAF_SIZE = 224
    paf_graph = tf.Graph()
    with paf_graph.as_default():
        image_in = tf.placeholder(tf.float32, [1, PAF_SIZE, PAF_SIZE, 3], name='image')
        two_branch_output = cpm_tf.build_paf_net(image_in, paf_net_params)
        paf_sess = tf.Session()
        init = tf.global_variables_initializer()
        paf_sess.run(init)
        tf.train.write_graph(paf_graph.as_graph_def(), paf_net_tf_dir, "paf_net.pb", False)
        tf.train.write_graph(paf_graph.as_graph_def(), paf_net_tf_dir, "paf_net.pbtxt", True)

    paf_graph = tf.Graph()
    with paf_graph.as_default():
        image_in = tf.placeholder(tf.float32, [1, PAF_SIZE, PAF_SIZE, 3], name='image')
        two_branch_output = cpm_tf.build_paf_net(image_in, paf_net_params)
        heatmap_output = cpm_tf.build_simple_paf_net(image_in, paf_net_params)
        paf_sess = tf.Session()
        init = tf.global_variables_initializer()
        paf_sess.run(init)
        tf.train.write_graph(paf_graph.as_graph_def(), paf_net_tf_dir, "paf_net_simple.pb", False)
        tf.train.write_graph(paf_graph.as_graph_def(), paf_net_tf_dir, "paf_net_simple.pbtxt", True)

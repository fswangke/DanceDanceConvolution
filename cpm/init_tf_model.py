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
        heatmap_person_output = cpm_tf.build_paf_net(image_in, paf_net_params)
        paf_sess = tf.Session()
        init = tf.global_variables_initializer()
        paf_sess.run(init)
        # paf_sess.run(heatmap_person_output, {image_in: image[np.newaxis] / 255.0 - 0.5})
        # saver = tf.train.Saver()
        # saver.save(person_sess, person_net_tf_cpt_file)
        tf.train.write_graph(paf_graph.as_graph_def(), paf_net_tf_dir, "paf_net.pb", False)
        tf.train.write_graph(paf_graph.as_graph_def(), paf_net_tf_dir, "paf_net.pbtxt", True)

    sys.exit(0)
    pose_net_tf_dir = os.path.join('tf_model', 'pose_net')
    if os.path.exists(pose_net_tf_dir) is False:
        os.makedirs(pose_net_tf_dir)
    person_net_tf_dir = os.path.join('tf_model', 'person_net')
    if os.path.exists(person_net_tf_dir) is False:
        os.makedirs(person_net_tf_dir)

    pose_net_params_file = os.path.join('caffe_model', 'pose_net.pickle')
    pose_net_tf_cpt_file = os.path.join(pose_net_tf_dir, 'pose_net.cpk')
    with open(pose_net_params_file, 'rb') as f:
        pose_net_params = pickle.load(f)

    person_net_params_file = os.path.join('caffe_model', 'person_net.pickle')
    person_net_tf_cpt_file = os.path.join(person_net_tf_dir, 'person_net.cpk')
    with open(person_net_params_file, 'rb') as f:
        person_net_params = pickle.load(f)

    PH, PW = 656, 376
    image = skimage.io.imread('nadal.png')
    image = skimage.transform.resize(image, (PH, PW)) * 255.0

    person_graph = tf.Graph()
    with person_graph.as_default():
        image_in = tf.placeholder(tf.float32, [1, PH, PW, 3], name='image')
        heatmap_person_output = cpm_tf.build_person_net(image_in, person_net_params)
        person_sess = tf.Session()
        init = tf.global_variables_initializer()
        person_sess.run(init)
        person_sess.run(heatmap_person_output, {image_in: image[np.newaxis] / 255.0 - 0.5})
        # saver = tf.train.Saver()
        # saver.save(person_sess, person_net_tf_cpt_file)
        tf.train.write_graph(person_graph.as_graph_def(), person_net_tf_dir, "person_net.pb", False)

        # PW, PH = 656, 376
        # person_graph = tf.Graph()
        # with person_graph.as_default():
        #     image = tf.placeholder(tf.float32, [1, PW, PH, 3])
        #     heatmap_person_output = cpm.inference_person(image)
        #     heatmap_person_large = tf.image.resize_images(heatmap_person_output, [PW, PH])
        #     print('Convert Caffe weights [WxHxCxN] to TF weights [NxCxWxH]')
        #     init_person_op, init_person_feed = init_tf_weights('PersonNet', person_net_params)
        #     print('Initialize TF model from converted weights')
        #     person_sess = tf.Session()
        #     person_sess.run(init_person_op, init_person_feed)
        #     init = tf.global_variables_initializer()
        #     person_sess.run(init)
        #     saver = tf.train.Saver()
        #     saver.save(person_sess, person_net_tf_cpt_file)
        #     tf.train.write_graph(person_graph.as_graph_def(), person_net_tf_dir, "person_net.pbtxt", True)

'''
    N, W, H = 16, 376, 376
    with tf.Session() as pose_sess:
        pose_image_in = tf.placeholder(tf.float32, [N, H, W, 3])
        pose_centermap_in = tf.placeholder(tf.float32, [N, H, W, 1])
        heatmap_pose = cpm.inference_pose(pose_image_in, pose_centermap_in)
        print('Convert Caffe weights [WxHxCxN] to TF weights [NxCxWxH]')
        init_pose_op, init_pose_feed = init_tf_weights('PoseNet', pose_net_params)
        print('Initialize TF model from converted weights')
        pose_sess.run(init_pose_op, init_pose_feed)

        saver = tf.train.Saver()
        saver.save(pose_sess, pose_net_tf_cpt_file)
        tf.train.write_graph(pose_sess.graph_def, pose_net_tf_dir, "pose_net.pbtxt", True)

'''

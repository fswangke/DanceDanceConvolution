import tensorflow as tf


def conv2dhelper(x, params, name, strides=1, padding='SAME'):
    params_list = params[name]
    W = tf.constant(params_list[0])
    b = tf.constant(params_list[1])

    def conv2d(x, W, b, strides=1, padding='SAME', name=None):
        return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding) + b, name=name)

    return conv2d(x, W, b, strides=strides, padding=padding, name=name)


def conv2d_no_relu(x, params, name, strides=1, padding='SAME'):
    params_list = params[name]
    W = tf.constant(params_list[0])
    b = tf.constant(params_list[1])

    def conv2d(x, W, b, strides=1, padding='SAME', name=None):
        return tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding), b, name=name)

    return conv2d(x, W, b, strides=strides, padding=padding, name=name)


def maxpool(x, k=2, padding='SAME', name=None):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding, name=name)


def build_person_net(image, params_dict):
    conv1_1 = conv2dhelper(image, params_dict, 'conv1_1')
    conv1_2 = conv2dhelper(conv1_1, params_dict, 'conv1_2')
    pool1_stage1 = maxpool(conv1_2, name='pool1_stage1')

    conv2_1 = conv2dhelper(pool1_stage1, params_dict, 'conv2_1')
    conv2_2 = conv2dhelper(conv2_1, params_dict, 'conv2_2')
    pool2_stage1 = maxpool(conv2_2, name='pool2_stage1')

    conv3_1 = conv2dhelper(pool2_stage1, params_dict, 'conv3_1')
    conv3_2 = conv2dhelper(conv3_1, params_dict, 'conv3_2')
    conv3_3 = conv2dhelper(conv3_2, params_dict, 'conv3_3')
    conv3_4 = conv2dhelper(conv3_3, params_dict, 'conv3_4')
    pool3_stage1 = maxpool(conv3_4, name='pool3_stage1')

    conv4_1 = conv2dhelper(pool3_stage1, params_dict, 'conv4_1')
    conv4_2 = conv2dhelper(conv4_1, params_dict, 'conv4_2')
    conv4_3 = conv2dhelper(conv4_2, params_dict, 'conv4_3')
    conv4_4 = conv2dhelper(conv4_3, params_dict, 'conv4_4')

    conv5_1 = conv2dhelper(conv4_4, params_dict, 'conv5_1')
    conv5_2_CPM = conv2dhelper(conv5_1, params_dict, 'conv5_2_CPM')

    conv6_1_CPM = conv2dhelper(conv5_2_CPM, params_dict, 'conv6_1_CPM')
    conv6_2_CPM = conv2d_no_relu(conv6_1_CPM, params_dict, 'conv6_2_CPM')

    concat_stage2 = tf.concat([conv6_2_CPM, conv5_2_CPM], 3, name='concat_stage2')
    Mconv1_stage2 = conv2dhelper(concat_stage2, params_dict, 'Mconv1_stage2')
    Mconv2_stage2 = conv2dhelper(Mconv1_stage2, params_dict, 'Mconv2_stage2')
    Mconv3_stage2 = conv2dhelper(Mconv2_stage2, params_dict, 'Mconv3_stage2')
    Mconv4_stage2 = conv2dhelper(Mconv3_stage2, params_dict, 'Mconv4_stage2')
    Mconv5_stage2 = conv2dhelper(Mconv4_stage2, params_dict, 'Mconv5_stage2')
    Mconv6_stage2 = conv2dhelper(Mconv5_stage2, params_dict, 'Mconv6_stage2')
    Mconv7_stage2 = conv2d_no_relu(Mconv6_stage2, params_dict, 'Mconv7_stage2')

    concat_stage3 = tf.concat([Mconv7_stage2, conv5_2_CPM], 3, name='concat_stage3')
    Mconv1_stage3 = conv2dhelper(concat_stage3, params_dict, 'Mconv1_stage3')
    Mconv2_stage3 = conv2dhelper(Mconv1_stage3, params_dict, 'Mconv2_stage3')
    Mconv3_stage3 = conv2dhelper(Mconv2_stage3, params_dict, 'Mconv3_stage3')
    Mconv4_stage3 = conv2dhelper(Mconv3_stage3, params_dict, 'Mconv4_stage3')
    Mconv5_stage3 = conv2dhelper(Mconv4_stage3, params_dict, 'Mconv5_stage3')
    Mconv6_stage3 = conv2dhelper(Mconv5_stage3, params_dict, 'Mconv6_stage3')
    Mconv7_stage3 = conv2d_no_relu(Mconv6_stage3, params_dict, 'Mconv7_stage3')

    concat_stage4 = tf.concat([Mconv7_stage3, conv5_2_CPM], 3, name='concat_stage4')
    Mconv1_stage4 = conv2dhelper(concat_stage4, params_dict, 'Mconv1_stage4')
    Mconv2_stage4 = conv2dhelper(Mconv1_stage4, params_dict, 'Mconv2_stage4')
    Mconv3_stage4 = conv2dhelper(Mconv2_stage4, params_dict, 'Mconv3_stage4')
    Mconv4_stage4 = conv2dhelper(Mconv3_stage4, params_dict, 'Mconv4_stage4')
    Mconv5_stage4 = conv2dhelper(Mconv4_stage4, params_dict, 'Mconv5_stage4')
    Mconv6_stage4 = conv2dhelper(Mconv5_stage4, params_dict, 'Mconv6_stage4')
    Mconv7_stage4 = conv2d_no_relu(Mconv6_stage4, params_dict, 'Mconv7_stage4')
    return Mconv7_stage4


def build_paf_net(image, params_dict):
    conv1_1 = conv2dhelper(image, params_dict, 'conv1_1')
    conv1_2 = conv2dhelper(conv1_1, params_dict, 'conv1_2')
    pool1_stage1 = maxpool(conv1_2, name='pool1_stage1')

    conv2_1 = conv2dhelper(pool1_stage1, params_dict, 'conv2_1')
    conv2_2 = conv2dhelper(conv2_1, params_dict, 'conv2_2')
    pool2_stage1 = maxpool(conv2_2, name='pool2_stage1')

    conv3_1 = conv2dhelper(pool2_stage1, params_dict, name='conv3_1')
    conv3_2 = conv2dhelper(conv3_1, params_dict, name='conv3_2')
    conv3_3 = conv2dhelper(conv3_2, params_dict, name='conv3_3')
    conv3_4 = conv2dhelper(conv3_3, params_dict, name='conv3_4')
    pool3_stage1 = maxpool(conv3_4, name='pool3_stage1')

    conv4_1 = conv2dhelper(pool3_stage1, params_dict, name='conv4_1')
    conv4_2 = conv2dhelper(conv4_1, params_dict, name='conv4_2')
    conv4_3_CPM = conv2dhelper(conv4_2, params_dict, name='conv4_3_CPM')
    conv4_4_CPM = conv2dhelper(conv4_3_CPM, params_dict, name='conv4_4_CPM')

    conv5_1_CPM_L1 = conv2dhelper(conv4_4_CPM, params_dict, name='conv5_1_CPM_L1')
    conv5_2_CPM_L1 = conv2dhelper(conv5_1_CPM_L1, params_dict, name='conv5_2_CPM_L1')
    conv5_3_CPM_L1 = conv2dhelper(conv5_2_CPM_L1, params_dict, name='conv5_3_CPM_L1')
    conv5_4_CPM_L1 = conv2dhelper(conv5_3_CPM_L1, params_dict, name='conv5_4_CPM_L1')
    conv5_5_CPM_L1 = conv2d_no_relu(conv5_4_CPM_L1, params_dict, name='conv5_5_CPM_L1')

    conv5_1_CPM_L2 = conv2dhelper(conv4_4_CPM, params_dict, name='conv5_1_CPM_L2')
    conv5_2_CPM_L2 = conv2dhelper(conv5_1_CPM_L2, params_dict, name='conv5_2_CPM_L2')
    conv5_3_CPM_L2 = conv2dhelper(conv5_2_CPM_L2, params_dict, name='conv5_3_CPM_L2')
    conv5_4_CPM_L2 = conv2dhelper(conv5_3_CPM_L2, params_dict, name='conv5_4_CPM_L2')
    conv5_5_CPM_L2 = conv2d_no_relu(conv5_4_CPM_L2, params_dict, name='conv5_5_CPM_L2')

    return conv5_5_CPM_L1, conv5_5_CPM_L2

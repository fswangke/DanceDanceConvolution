from __future__ import print_function
from __future__ import division

import cPickle as pickle
import caffe
import os
import requests
import shutil
import sys


def convert_caffe_weights_to_tf(W):
    if len(W.shape) == 4:
        # TF order: N, C, W, H
        # CF order: W, H, C, N
        return W.transpose((2, 3, 1, 0))
    elif len(W.shape) == 1:
        return W
    else:
        raise ValueError('Unsupported weights.')


def convert_caffe_net(net):
    return {name: [convert_caffe_weights_to_tf(blob.data) for blob in blobs]
            for name, blobs in net.params.iteritems()}


def download_file(url, dst_filename):
    r = requests.get(url, stream=True)
    print('Downloading from', url)
    with open(dst_filename, 'wb') as f:
        shutil.copyfileobj(r.raw, f)
    print('Saved to', dst_filename)
    return


if __name__ == '__main__':
    paf_proto_file = os.path.join('paf_model', 'pose_deploy.prototxt')
    paf_model_file = os.path.join('paf_model', 'pose_iter_440000.caffemodel')
    print('Initalize PAF network')
    paf_net = caffe.Net(paf_proto_file, caffe.TEST, weights=paf_model_file)
    print('Convert PAF weight to TF-compatible order')
    paf_params = convert_caffe_net(paf_net)
    print('Saving PAF weights')
    with open('paf_model/paf_params.pickle', 'wb') as f:
        pickle.dump(paf_params, f)

    sys.exit(0)
    person_deploy_file = ''
    person_model_file = ''

    # Load the pose net model and weight
    posenet_weight_url = r'http://pearl.vasc.ri.cmu.edu/caffe_model_github/model/_trained_MPI/pose_iter_320000.caffemodel'
    posenet_caffe_weight_file = os.path.join('caffe_model', 'pose_net.caffemodel')
    posenet_caffe_proto_file = os.path.join('caffe_model', 'pose_net.prototxt')
    posenet_pickle_file = os.path.join('caffe_model', 'pose_net.pickle')

    if os.path.exists(posenet_caffe_weight_file) is False:
        download_file(posenet_weight_url, posenet_caffe_weight_file)
    posenet_caffe_net = caffe.Net(posenet_caffe_proto_file, caffe.TEST, weights=posenet_caffe_weight_file)
    posenet_params = convert_caffe_net(posenet_caffe_net)
    with open(posenet_pickle_file, 'wb') as f:
        pickle.dump(posenet_params, f)

    personnet_weight_url = r'http://pearl.vasc.ri.cmu.edu/caffe_model_github/model/_trained_person_MPI/pose_iter_70000.caffemodel'
    personnet_caffe_weight_file = os.path.join('caffe_model', 'person_net.caffemodel')
    personnet_caffe_proto_file = os.path.join('caffe_model', 'person_net.prototxt')
    personnet_pickle_file = os.path.join('caffe_model', 'person_net.pickle')
    if os.path.exists(personnet_caffe_weight_file) is False:
        download_file(personnet_weight_url, personnet_caffe_weight_file)
    personnet_caffe_net = caffe.Net(personnet_caffe_proto_file, caffe.TEST, weights=personnet_caffe_weight_file)
    personnet_params = convert_caffe_net(personnet_caffe_net)
    with open(personnet_pickle_file, 'wb') as f:
        pickle.dump(personnet_params, f)

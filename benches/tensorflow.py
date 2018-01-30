#!/usr/bin/env python3

import time

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import config
from models.tensorflow import resnet_model


def run(name, net, images):
    print(name)

    sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)

    for j in range(config.NUM_BATCHES):
        sess.run(net, feed_dict={'images:0': images})

    end = time.time()
    for i in range(config.NUM_RUNS):
        for j in range(config.NUM_BATCHES):
            sess.run(net, feed_dict={'images:0': images})
        print('fw\t', f'{(time.time() - end) / 32:0.4f}')
        end = time.time()


def run_classification(name, model):
    shape = config.CLASSIFICATION_SHAPE
    shape = [shape[0], shape[2], shape[3], shape[1]]
    images = tf.placeholder(tf.float32, shape, name='images')
    logits = model(inputs=images, is_training=False)
    images = np.random.uniform(0, 1, shape)

    run(name, logits, images)


def run_all():
    with tf.Graph().as_default():
        resnet18 = resnet_model.resnet_v2(resnet_size=18, num_classes=1001)
        run_classification('resnet18', resnet18)
    with tf.Graph().as_default():
        resnet50 = resnet_model.resnet_v2(resnet_size=50, num_classes=1001)
        run_classification('resnet50', resnet50)


if __name__ == '__main__':
    run_all()

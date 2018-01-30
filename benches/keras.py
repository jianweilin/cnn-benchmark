#!/usr/bin/env python3

import time

import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input

import config


def run(name, model, images):
    print(name)

    net = model()
    for j in range(config.NUM_BATCHES):
        net.predict(images)

    end = time.time()
    for i in range(config.NUM_RUNS):
        for j in range(config.NUM_BATCHES):
            net.predict(images)
        print('fw\t', f'{(time.time() - end) / 32:0.4f}')
        end = time.time()


def run_classification(name, model):
    shape = config.CLASSIFICATION_SHAPE
    shape = [shape[0], shape[2], shape[3], shape[1]]
    images = np.random.uniform(0, 255, shape)
    images = preprocess_input(images)

    run(name, model, images)


def run_all():
    run_classification('resnet50', ResNet50)


if __name__ == '__main__':
    run_all()

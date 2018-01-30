#!/usr/bin/env python3

import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet18, resnet50

import config
from models.pytorch.linknet import linknet18


def run(name, model, images):
    print(name)

    net = model()
    net = nn.DataParallel(net).cuda()
    for j in range(config.NUM_BATCHES):
        net.forward(images)

    end = time.time()
    for i in range(config.NUM_RUNS):
        for j in range(config.NUM_BATCHES):
            net.forward(images)
        print('fw\t', f'{(time.time() - end) / 32:0.4f}')
        end = time.time()


def run_segmentation(name, model):
    images = Variable(torch.randn(config.SEGMENTATION_SHAPE), volatile=True)

    run(name, model, images)


def run_classification(name, model):
    images = Variable(torch.randn(config.CLASSIFICATION_SHAPE), volatile=True)

    run(name, model, images)


def run_all():
    run_classification('resnet18', resnet18)
    run_classification('resnet50', resnet50)
    run_segmentation('linknet18', linknet18)


if __name__ == '__main__':
    cudnn.benchmark = True

    run_all()

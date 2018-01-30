#!/usr/bin/env python3

import time

import torch
torch.backends.cudnn.benchmark = True
from torch.autograd import Variable
from torchvision.models import resnet18, resnet50

import config
from models.pytorch.linknet import linknet18


def run(name, model, images):
    print(name)

    net = model()
    net.cuda()

    # warm-up
    for j in range(config.NUM_BATCHES):
        net.forward(images)

    timings = []
    for i in range(config.NUM_RUNS):
        torch.cuda.synchronize()
        t1 = time.time()
        for j in range(config.NUM_BATCHES):
            net.forward(images)
        torch.cuda.synchronize()
        t2 = time.time()
        batch_time = (t2 - t1) / config.NUM_BATCHES
        timings.append(batch_time)
        print('fw\t', f'{batch_time:0.6f}')
    mean_time = sum(timings) / config.NUM_RUNS
    print('mean\t', f'{mean_time:0.6f}')


def run_segmentation(name, model):
    images = Variable(torch.randn(config.SEGMENTATION_SHAPE), volatile=True).cuda()

    run(name, model, images)


def run_classification(name, model):
    images = Variable(torch.randn(config.CLASSIFICATION_SHAPE), volatile=True).cuda()

    run(name, model, images)


def run_all():
    run_classification('resnet18', resnet18)
    print()
    run_classification('resnet50', resnet50)
    print()
    run_segmentation('linknet18', linknet18)


if __name__ == '__main__':
    run_all()

from __future__ import print_function, absolute_import
import argparse
import sys
import os.path as osp

import torch
from torch.utils.data import DataLoader


import numpy as np

from reid.datasets import get_dataset
from reid.dist_metric import DistanceMetric
from reid.evaluators import FeatureEvaluate
from reid.utils.data import transforms
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger

def main(args):
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    #create dataset and loader
    root = args.data_dir
    dataset = get_dataset('singleperson41', root, split_id=args.split, num_val=10, download=False)

    #only need test loader
    test_loader = DataLoader(Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                                          root=dataset.images_dir,
                                          transform=transforms.Compose([
                                              transforms.RandomSizedRectCrop(256, 128),
                                              transforms.ToTensor(),
                                          ])),
                             batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

    #create evaluator for extracted features
    metric = DistanceMetric(algorithm=args.dist_metric)
    evaluator = FeatureEvaluate(args.feature_dir)
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of PartNet')
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--dist-metric', type=str, default='euclidean', choices=['euclidean', 'kissme'])

    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data/singleperson41_acf'))
    parser.add_argument('--feature_dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data/singleperson41_acf/feats_partnet'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/eval_singleperson41/partnet'))

    main(parser.parse_args())


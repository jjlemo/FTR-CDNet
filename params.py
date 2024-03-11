import glob
from argparse import ArgumentParser
from thop import profile
import utils
import torch

from models.FTRNet import FTRNet_V1, FTRNet_V2
from models.basic_model import MITCDEvaluator
from pytorch_model_summary import summary
import models.trainer
import numpy as np
import osgeo as gdal

import os

"""
quick start

sample files in ./samples

save prediction files in the ./samples/predict

"""


def get_args():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='TIPCD_20221129', type=str)
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--output_folder', default='predict/MITCD_LEVIR_v3', type=str)

    # data
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dataset', default='TIPCDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default="test", type=str)
    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='TIPCD_V1', type=str,
                        help='base_resnet18 | base_transformer_pos_s4_dd8 | base_transformer_pos_s4_dd8_dedim8|')
    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    utils.get_device(args)
    device = torch.device("cuda:%s" % args.gpu_ids[0]
                          if torch.cuda.is_available() and len(args.gpu_ids) > 0
                          else "cpu")
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    # os.makedirs(args.output_folder, exist_ok=True)
    #
    # log_path = os.path.join(args.output_folder, 'log_vis.txt')

    # data_loader = dataloaders = models.TIP_trainer.utils.get_loaders(args)

    model = FTRNet_V2()
    # model.load_checkpoint(args.checkpoint_name)
    input1 = torch.randn(1, 3, 256, 256)
    input2 = torch.randn(1, 3, 256, 256)
    input3 = torch.randn(1, 3, 256, 256)
    flops, params = profile(model, (input1, input2, input3))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))

    #
    # for i, batch in enumerate(data_loader):
    #     name = batch['name']
    #     print('process: %s' % name)
    #     score_map = model._forward_pass(batch)
    #     model._save_predictions()

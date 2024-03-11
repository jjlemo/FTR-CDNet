from argparse import ArgumentParser
import torch
import models.trainer

# print(models.MultiInput_trainer.torch.cuda.is_available())
print(models.trainer.torch.cuda.is_available())

print(torch.cuda.is_available())

"""
the main function for training the CD networks
"""


def train(args):
    dataloaders = models.trainer.utils.get_loaders(args)
    model = models.trainer.CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()


def test(args):
    from models.evaluator import CDEvaluator
    dataloader = models.trainer.utils.get_loader(args.data_name, img_size=args.img_size,
                                                 batch_size=args.batch_size, is_train=False,
                                                 split='test', dataset="CDDataset")
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models()


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='2', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--local_rank', default='0', type=str)
    parser.add_argument('--project_name', default='FTRNet_LEVIR_V4_20230320', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    parser.add_argument('--vis_root', default='vis', type=str)
    # data
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--shuffle_AB', default=False, type=str)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--pretrain', default=r"./checkpoints/FTRNet_LEVIR_V4_20230306/best_ckpt.pt", type=str)
    parser.add_argument('--multi_scale_train', default=False, type=str)
    parser.add_argument('--multi_scale_infer', default=False, type=str)
    parser.add_argument('--multi_pred_weights', nargs='+', type=float, default=[0.5, 0.5, 0.5, 0.8, 1.0])

    parser.add_argument('--net_G', default='FTRNet_V4', type=str,
                        help='base_resnet18 | base_transformer_pos_s4 | base_transformer_pos_s4_dd8 | '
                             'base_transformer_pos_s4_dd8_dedim8|FTRNet_V1')
    parser.add_argument('--loss', default='ce', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=0.0006, type=float)
    parser.add_argument('--max_epochs', default=600, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)

    args = parser.parse_args()
    models.trainer.utils.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = models.trainer.os.path.join(args.checkpoint_root, args.project_name)
    models.trainer.os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = models.trainer.os.path.join(args.vis_root, args.project_name)
    models.trainer.os.makedirs(args.vis_dir, exist_ok=True)

    train(args)

    test(args)

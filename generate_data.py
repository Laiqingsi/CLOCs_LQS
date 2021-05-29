from torch.utils.data import DataLoader
from pathlib import Path
from tool.dataset import clocs_data
import argparse
import torch
import datetime
from tool import fusion,nms
from tool.pcdet.config import cfg, cfg_from_yaml_file
from tool.pcdet.datasets import build_dataloader
from tool.pcdet.utils import common_utils
from tqdm import tqdm
from tool.Focaloss import SigmoidFocalClassificationLoss





def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    parser.add_argument('--cfg_file', type=str, default='./tool/cfgs/kitti_models/second/second_car.yaml', help='specify the config for training')
    parser.add_argument('--d2path', type=str, default='../data/clocs_data/2D',
                        help='2d prediction path')
    parser.add_argument('--d3path', type=str, default='../data/clocs_data/3D',
                        help='3d prediction path')
    parser.add_argument('--inputpath', type=str, default='../data/clocs_data/input_data',
                        help='input data save path')
    parser.add_argument('--train-indexpath', type=str, default='../data/clocs_data/index/train.txt',
                        help='index data path')
    parser.add_argument('--val-indexpath', type=str, default='../data/clocs_data/index/val.txt',
                        help='index data path')
    parser.add_argument('--epochs', type=int, default=50,
                        help='training epochs')
    parser.add_argument('--infopath', type=str, default='../data/clocs_data/info/kitti_infos_trainval.pkl',
                        help='index data path')
    parser.add_argument('--d2method', type=str, default='faster',
                        help='2d prediction method')
    parser.add_argument('--d3method', type=str, default='pointpillar',
                        help='3d prediction method')
    parser.add_argument('--log-path', type=str, default='./log/pointpillar/faster',
                        help='log path')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    return args, cfg

if __name__ == "__main__":
    args, cfg = parse_args()
    _2d_path = args.d2path
    _3d_path = args.d3path
    input_data = args.inputpath
    train_ind_path = args.train_indexpath
    val_ind_path = args.val_indexpath


    train_dataset = clocs_data(_2d_path, _3d_path,train_ind_path, input_data, d2method , d3method)

    train_dataset.generate_input()

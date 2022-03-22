from torch.utils.data import DataLoader
from pathlib import Path
from tool.dataset import clocs_data
import argparse
import torch
import datetime
from tool import fusion,nms
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from tqdm import tqdm
from tool.Focaloss import SigmoidFocalClassificationLoss

Focal = SigmoidFocalClassificationLoss()



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
    parser.add_argument('--d2method', type=str, default='cascade_o',
                        help='2d prediction method')
    parser.add_argument('--d3method', type=str, default='pointpillar',
                        help='3d prediction method')
    parser.add_argument('--log-path', type=str, default='./log/pointpillar',
                        help='log path')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    return args, cfg


def eval(val_data, logf, log_path, cfg, eval_set, logger):
    det_annos = []
    logger.info("#################################")
    print("#################################", file=logf)
    logger.info("# EVAL" + 'baseline')
    print("# EVAL"+'baseline', file=logf)
    logger.info("#################################")
    print("#################################", file=logf)
    logger.info("Generate output labels...")
    print("Generate output labels...", file=logf)
    for fusion_input,tensor_index,path in tqdm(val_data):
        fusion_input = fusion_input.cuda()
        tensor_index = tensor_index.reshape(-1,2)
        tensor_index = tensor_index.cuda()
        _3d_result = torch.load(path[0])[0]
        cls_preds = torch.tensor(_3d_result['score'])
        box_preds = torch.tensor(_3d_result['boxes_lidar']).cuda()
        selected = nms.nms(cls_preds, box_preds, cfg.MODEL.POST_PROCESSING)
        selected = selected.numpy()
        for key in _3d_result.keys():
            if key == 'frame_id':
                continue
            _3d_result[key] = _3d_result[key][selected]
        det_annos.append(_3d_result)
    
    logger.info("Generate output done")
    print("Generate output done", file=logf)
    torch.save(det_annos,log_path+'/'+'baseline'+'.pt')
    result_str, result_dict = eval_set.evaluation(
        det_annos, cfg.CLASS_NAMES,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC
    )
    print(result_str, file=logf)
    logger.info(result_str)




if __name__ == "__main__":
    args, cfg = parse_args()
    _2d_path = args.d2path
    _3d_path = args.d3path
    d2method = args.d2method
    d3method = args.d3method
    input_data = args.inputpath
    train_ind_path = args.train_indexpath
    val_ind_path = args.val_indexpath
    log_path = args.log_path
    logf = open(log_path+'/log_baseline.txt', 'a')
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    infopath = args.infopath

    val_dataset = clocs_data(_2d_path, _3d_path,val_ind_path, input_data, d2method , d3method, infopath, val=True)

    val_data = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True
    )
    
    eval_set, _, __ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=8, logger=logger, training=False
    )


    eval(val_data, logf, log_path,  cfg, eval_set, logger)
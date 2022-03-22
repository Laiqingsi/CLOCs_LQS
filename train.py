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
from pathlib import Path

Focal = SigmoidFocalClassificationLoss()



def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    parser.add_argument('--cfg_file', type=str, default='./tool/cfgs/kitti_models/second/second_car.yaml', help='specify the config for training')
    parser.add_argument('--rootpath', type=str, default='./data/clocs_data',
                        help='data root path')
    parser.add_argument('--d2path', type=str, default='./data/clocs_data/2D',
                        help='2d prediction path')
    parser.add_argument('--d3path', type=str, default='./data/clocs_data/3D',
                        help='3d prediction path')
    parser.add_argument('--inputpath', type=str, default='./data/clocs_data/input_data',
                        help='input data save path')
    parser.add_argument('--train-indexpath', type=str, default='./data/clocs_data/index/train.txt',
                        help='index data path')
    parser.add_argument('--val-indexpath', type=str, default='./data/clocs_data/index/val.txt',
                        help='index data path')
    parser.add_argument('--epochs', type=int, default=50,
                        help='training epochs')
    parser.add_argument('--infopath', type=str, default='./data/clocs_data/info/kitti_infos_trainval.pkl',
                        help='index data path')
    parser.add_argument('--d2method', type=str, default='faster',
                        help='2d prediction method')
    parser.add_argument('--d3method', type=str, default='second',
                        help='3d prediction method')
    parser.add_argument('--log-path', type=str, default='./log/second/faster',
                        help='log path')
    parser.add_argument('--generate', type=int, default=0,
                        help='whether generate data')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    return args, cfg

def train(net, train_data, optimizer, epoch, logf):
    cls_loss_sum = 0
    optimizer.zero_grad()
    step = 1
    display_step = 500
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for fusion_input,tensor_index,positives,negatives,one_hot_targets,label_n,idx in tqdm(train_data):
        fusion_input = fusion_input.cuda()
        tensor_index = tensor_index.reshape(-1,2)
        tensor_index = tensor_index.cuda()
        positives = positives.cuda()
        negatives = negatives.cuda()
        one_hot_targets = one_hot_targets.cuda()
        cls_preds,flag = fusion_layer(fusion_input,tensor_index)

        negative_cls_weights = negatives.type(torch.float32) * 1.0
        cls_weights = negative_cls_weights + 1.0 * positives.type(torch.float32)
        pos_normalizer = positives.sum(1, keepdim=True).type(torch.float32)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        if flag==1:
            cls_preds = cls_preds[:,:one_hot_targets.shape[1],:]
            cls_losses = Focal._compute_loss(cls_preds, one_hot_targets, cls_weights.cuda())  # [N, M]
            cls_losses_reduced = cls_losses.sum()/(label_n.item()+1)
            # cls_losses_reduced = cls_losses.sum()
            cls_loss_sum = cls_loss_sum + cls_losses.sum()
            cls_losses_reduced.backward()
            optimizer.step()
            optimizer.zero_grad()
        step = step + 1
        if step%display_step == 0:
            print("epoch:",epoch, " step:", step, " and the cls_loss is :",cls_loss_sum.item()/display_step, file=logf)
            print("epoch:",epoch, " step:", step, " and the cls_loss is :",cls_loss_sum.item()/display_step)
            cls_loss_sum = 0


def eval(net, val_data, logf, log_path, epoch, cfg, eval_set, logger):
    net.eval()
    det_annos = []
    
    logger.info("#################################")
    print("#################################", file=logf)
    logger.info("# EVAL" + str(epoch))
    print("# EVAL"+ str(epoch), file=logf)
    logger.info("#################################")
    print("#################################", file=logf)
    logger.info("Generate output labels...")
    print("Generate output labels...", file=logf)
    for fusion_input,tensor_index,path in tqdm(val_data):
        fusion_input = fusion_input.cuda()
        tensor_index = tensor_index.reshape(-1,2)
        tensor_index = tensor_index.cuda()
        _3d_result = torch.load(path[0])[0]
        fusion_cls_preds,flag = net(fusion_input,tensor_index)
        cls_preds = fusion_cls_preds.reshape(-1).cpu()
        cls_preds = torch.sigmoid(cls_preds)
        cls_preds = cls_preds[:len(_3d_result['score'])]
        _3d_result['score'] = cls_preds.detach().cpu().numpy()
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
    torch.save(det_annos,log_path+'/'+'result'+'/'+str(epoch)+'.pt')
    result_str, result_dict = eval_set.evaluation(
        det_annos, cfg.CLASS_NAMES,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC
    )
    print(result_str, file=logf)
    logger.info(result_str)
    net.train()




if __name__ == "__main__":
    args, cfg = parse_args()
    root_path = args.rootpath
    _2d_path = args.d2path
    _3d_path = args.d3path
    input_data = args.inputpath
    train_ind_path = args.train_indexpath
    val_ind_path = args.val_indexpath
    log_path = args.log_path
    log_Path = Path(log_path)
    if not log_Path.exists():
        log_Path.mkdir(parents=True)
    result_Path = log_Path / 'result'
    if not result_Path.exists():
        result_Path.mkdir(parents=True)
    infopath = args.infopath
    logf = open(log_path+'/log.txt', 'a')
    log_file = log_Path / 'log.txt'
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    if args.generate :
        train_dataset = clocs_data(_2d_path, _3d_path,'./data/clocs_data/index/trainval.txt', input_data, infopath)
        train_dataset.generate_input()

    train_dataset = clocs_data(_2d_path, _3d_path,train_ind_path, input_data, infopath)
    val_dataset = clocs_data(_2d_path, _3d_path,val_ind_path, input_data, infopath, val=True)

    train_data = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True
    )

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
        root_path=root_path,
        dist=False, workers=8, logger=logger, training=False
    )

    fusion_layer = fusion.fusion()
    fusion_layer.cuda()

    optimizer = torch.optim.Adam(fusion_layer.parameters(),lr = 3e-3, betas=(0.9, 0.99),weight_decay=0.01)

    for epoch in range(args.epochs):
        train(fusion_layer, train_data, optimizer, epoch, logf)
        torch.save(fusion_layer, log_path+'/'+str(epoch)+'.pt')
        eval(fusion_layer, val_data, logf, log_path, epoch, cfg, eval_set, logger)


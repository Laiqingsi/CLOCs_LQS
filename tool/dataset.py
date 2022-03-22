import numpy as np
import torch
import time
import numba
import pickle
from torch.utils.data import Dataset
from pcdet.datasets.kitti.kitti_object_eval_python.eval import d3_box_overlap
from tqdm import tqdm
from pathlib import Path


class clocs_data(Dataset):

    def __init__(self, _2d_path, _3d_path, index_path, input_data, infopath = '../data/clocs_data/info/kitti_infos_trainval.pkl', val = False):

        self._2d_path = _2d_path 
        self._3d_path = _3d_path 
        f = open(index_path, "r")
        self.ind = f.read().splitlines()
        f.close()
        self.val = val
        self.input_data = input_data
        self.anno = pickle.load(open(infopath,'rb'))
        self.id2ind = np.zeros(len(self.anno))
        # self.id2ind[img_id] = number in self.anno
        for i in range(len(self.anno)):
            # print(self.anno[i])
            self.id2ind[int(self.anno[i]['image_idx'])] = i

    def generate_input(self):
        input_path = Path(self.input_data)
        if not input_path.exists():
            input_path.mkdir(parents=True)
        for i in tqdm(range(len(self.ind))):
            
            # find 2d detection
            detection_2d_file_name = self._2d_path+"/"+self.ind[i]+".txt"
            with open(detection_2d_file_name, 'r') as f:
                lines = f.readlines()
            content = [line.strip().split(' ') for line in lines]
            predicted_class = np.array([x[0] for x in content],dtype='object')
            predicted_class_index = np.where(predicted_class=='Car')

            # get bbox in 2d 
            detection_result = np.array([[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
            score = np.array([float(x[15]) for x in content])  # 1000 is the score scale!!!
            f_detection_result=np.append(detection_result,score.reshape(-1,1),1)
            middle_predictions=f_detection_result[predicted_class_index,:].reshape(-1,5)
            top_predictions=middle_predictions[np.where(middle_predictions[:,4]>=-100)]

            # get 3d result
            _3d_result = torch.load(self._3d_path+"/"+self.ind[i]+".pt")[0]

            # get input data
            res, iou_test, tensor_index = self.train_stage_2(_3d_result, top_predictions)
            all_3d_output_camera_dict, fusion_input,tensor_index = res, iou_test, tensor_index

            # get 3d anno
            int_ind = int(self.id2ind[i])
            gt_anno = self.anno[int_ind]['annos']
            d3_gt_boxes = self.process_anno(gt_anno)

            # get training label
            d3_gt_boxes_camera=d3_gt_boxes
            if d3_gt_boxes.shape[0] == 0:
                target_for_fusion = np.zeros((1,20000,1))
                positive_index = np.zeros((1,20000),dtype=np.float32)
                negative_index = np.zeros((1,20000),dtype=np.float32)
                negative_index[:,:] = 1
            else:
                ###### predicted bev boxes
                pred_3d_box = all_3d_output_camera_dict[0]["box3d_camera"]
                iou_bev = d3_box_overlap(d3_gt_boxes_camera, pred_3d_box, criterion=-1)
                iou_bev_max = np.amax(iou_bev,axis=0)
                #print(np.max(iou_bev_max))
                target_for_fusion = ((iou_bev_max >= 0.7)*1).reshape(1,-1,1)

                positive_index = ((iou_bev_max >= 0.7)*1).reshape(1,-1)
                negative_index = ((iou_bev_max <= 0.5)*1).reshape(1,-1)
            
            # save data and label
            all_data = {}
            all_data['input_data'] = {'fusion_input':fusion_input.numpy(), 'tensor_index':tensor_index.numpy()}
            all_data['label'] = {'target_for_fusion':torch.tensor(target_for_fusion),
            'positive_index':torch.tensor(positive_index), 'negative_index':torch.tensor(negative_index),
            'label_n':len(d3_gt_boxes_camera)}
            torch.save(all_data,self.input_data+'/'+self.ind[i]+'.pt')
            
    def process_anno(self, anno, class_name=['Car']):
        ind = np.where(anno['name'] == class_name[0])
        loc = anno['location'][ind]
        dim = anno['dimensions'][ind]
        rot = anno['rotation_y'][ind]
        if len(loc) == 0:
            d3_box = []
        else:
            d3_box = np.concatenate((loc,dim,rot.reshape(-1,1)),axis=1)
        return np.array(d3_box)

    def train_stage_2(self, _3d_result,top_predictions):
        box_preds = _3d_result['boxes_lidar']
        final_box_preds = box_preds
        predictions_dicts = []
        locs = _3d_result['location']
        dims = _3d_result['dimensions']
        angles = _3d_result['rotation_y'].reshape(-1,1)
        final_box_preds_camera = np.concatenate((locs,dims,angles),axis=1)
        box_2d_preds = _3d_result['bbox']
        final_scores = _3d_result['score']
        img_idx = _3d_result['frame_id']
        # predictions
        predictions_dict = {
            "bbox": box_2d_preds,
            "box3d_camera": final_box_preds_camera,
            "box3d_lidar": final_box_preds,
            "scores": final_scores,
            #"label_preds": label_preds,
            "image_idx": img_idx,
        }
        predictions_dicts.append(predictions_dict)
        dis_to_lidar = torch.norm(torch.tensor(box_preds[:,:2]),p=2,dim=1,keepdim=True).numpy()/82.0
        box_2d_detector = np.zeros((200, 4))
        box_2d_detector[0:top_predictions.shape[0],:]=top_predictions[:,:4]
        box_2d_detector = top_predictions[:,:4]
        box_2d_scores = top_predictions[:,4].reshape(-1,1)
        time_iou_build_start=time.time()
        overlaps1 = np.zeros((900000,4),dtype=np.float32)
        tensor_index1 = np.zeros((900000,2),dtype=np.float32)
        overlaps1[:,:] = -1.0
        tensor_index1[:,:] = -1.0
            #final_scores[final_scores<0.1] = 0
            #box_2d_preds[(final_scores<0.1).reshape(-1),:] = 0 
        iou_test,tensor_index, max_num = build_stage2_training(box_2d_preds,
                                            box_2d_detector,
                                            -1,
                                            final_scores,
                                            box_2d_scores,
                                            dis_to_lidar,
                                            overlaps1,
                                            tensor_index1)
        time_iou_build_end=time.time()
        iou_test_tensor = torch.FloatTensor(iou_test)  #iou_test_tensor shape: [160000,4]
        tensor_index_tensor = torch.LongTensor(tensor_index)
        iou_test_tensor = iou_test_tensor.permute(1,0)
        iou_test_tensor = iou_test_tensor.reshape(4,900000)
        tensor_index_tensor = tensor_index_tensor.reshape(-1,2)
        if max_num == 0:
            non_empty_iou_test_tensor = torch.zeros(4,2)
            non_empty_iou_test_tensor[:,:] = -1
            non_empty_tensor_index_tensor = torch.zeros(2,2)
            non_empty_tensor_index_tensor[:,:] = -1
        else:
            non_empty_iou_test_tensor = iou_test_tensor[:,:max_num]
            non_empty_tensor_index_tensor = tensor_index_tensor[:max_num,:]

        return predictions_dicts, non_empty_iou_test_tensor, non_empty_tensor_index_tensor

    def __getitem__(self, index):
        idx = self.ind[index]
        all_data = torch.load(self.input_data+'/'+idx+'.pt')
        inpu_data = all_data['input_data']
        label = all_data['label']
        fusion_input = torch.tensor(inpu_data['fusion_input']).reshape(4,1,-1)
        tensor_index = torch.tensor(inpu_data['tensor_index']).reshape(-1,2)
        target_for_fusion = label['target_for_fusion'].reshape(-1,1)
        positive_index = label['positive_index'].reshape(-1)
        negative_index = label['negative_index'].reshape(-1)
        label_n = label['label_n']

        if self.val:
            # get 3d result
            return fusion_input,tensor_index,(self._3d_path+"/"+idx+".pt")
        else:
            positives = positive_index.type(torch.float32)
            negatives = negative_index.type(torch.float32)
            one_hot_targets = target_for_fusion.type(torch.float32)
            return fusion_input,tensor_index,positives,negatives,one_hot_targets,label_n,idx

    def __len__(self):
        return len(self.ind)

# pang added to build the tensor for the second stage of training
@numba.jit(nopython=True,parallel=True)
def build_stage2_training(boxes, query_boxes, criterion, scores_3d, scores_2d, dis_to_lidar_3d,overlaps,tensor_index):
    N = boxes.shape[0] #20000
    K = query_boxes.shape[0] #30
    max_num = 900000
    ind=0
    ind_max = ind
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[ind,0] = iw * ih / ua
                    overlaps[ind,1] = scores_3d[n]
                    overlaps[ind,2] = scores_2d[k,0]
                    overlaps[ind,3] = dis_to_lidar_3d[n,0]
                    tensor_index[ind,0] = k
                    tensor_index[ind,1] = n
                    ind = ind+1

                elif k==K-1:
                    overlaps[ind,0] = -10
                    overlaps[ind,1] = scores_3d[n]
                    overlaps[ind,2] = -10
                    overlaps[ind,3] = dis_to_lidar_3d[n,0]
                    tensor_index[ind,0] = k
                    tensor_index[ind,1] = n
                    ind = ind+1
            elif k==K-1:
                overlaps[ind,0] = -10
                overlaps[ind,1] = scores_3d[n]
                overlaps[ind,2] = -10
                overlaps[ind,3] = dis_to_lidar_3d[n,0]
                tensor_index[ind,0] = k
                tensor_index[ind,1] = n
                ind = ind+1
    if ind > ind_max:
        ind_max = ind
    return overlaps, tensor_index, ind
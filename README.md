## CLOCs_LQS: An Implementation of CLOCs

CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection. CLOCs is a novel Camera-LiDAR Object Candidates fusion network. This is my implementation based on Open-PCdet and MMDetection. The paper information can be found below. 

```
@article{pang2020clocs,
  title={CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection},
  author={Pang, Su and Morris, Daniel and Radha, Hayder},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020}
  organization={IEEE}
}
```

Thanks to the original [implementation](https://github.com/pangsu0613/CLOCs), it helps me a lot. 

### Environment

Python 3.7, Pytorch 1.5, Ubuntu 18.04

### Performance

new 40 recall points

![Result](.\Result.png)

### Install PCdet

The code is partly based on [Open-PCdet](https://github.com/open-mmlab/OpenPCDet), you need to install it first. Please follow the step [here](https://github.com/open-mmlab/OpenPCDet). Please remind that the PCdet should be installed in another folder not in this folder. 

### Get Fusion Results

#### Get detection results

1. You need to prepare the 3D detection results and 2D detection results. Please note that the results that do not go through NMS are better. 
2. For the 3D detection results, you can just use the PCdet that you have installed to train your own model. Then you need to get the prediction results. If you want to get no NMS results, then you need to modified the dataset yaml file in  `OpenPCDet/tools/cfgs/dataset_configs` and model prediction file in `OpenPCDet/tools/cfgs/*_models`. How to modified them depends on your model and your dataset.
3. In `OpenPCDet/tools/eval_utils/eval_utils.py` line 61：

```python
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
```

​	Here, `annos` denotes one detection results for one input. You can add one line `torch.save(***)` or other method to save the detection results. Here you need to predict all dataset results that training and validation dataset. So that, you can train your CLOCs model and validate it. Or you can download 3D detection results [here](https://jbox.sjtu.edu.cn/l/OFgs7G). If you can get the results have same format as  I supplied, that will be OK also.

4. You need to get 2D detection results, you can use [mmdetection](https://github.com/open-mmlab/mmdetection) to get your results. Or you can download [here](https://jbox.sjtu.edu.cn/l/hFDjf2). (this file is from original [CLOCs](https://github.com/pangsu0613/CLOCs)). The 2D result format is as below, almost same as kitti format.

```
Car -1 -1 -10 1133.50 278.19 1225.04 329.51 -1 -1 -1 -1000 -1000 -1000 -10 0.0150 
Car -1 -1 -10 1156.30 225.86 1225.01 262.08 -1 -1 -1 -1000 -1000 -1000 -10 0.0139 
Car -1 -1 -10 1044.50 215.57 1112.86 259.75 -1 -1 -1 -1000 -1000 -1000 -10 0.0021 
Car -1 -1 -10 1166.70 225.15 1225.02 246.63 -1 -1 -1 -1000 -1000 -1000 -10 0.0014 
Car -1 -1 -10 751.01 150.31 782.09 177.66 -1 -1 -1 -1000 -1000 -1000 -10 0.0014 
```

#### Generate fusion input

Generate the input data for CLOCs training.  Modify the file `generate_data.py` as below, or you can just skip this step, go next to modify `train.py`, but the rule of `train.py` is same as here.

```
'--d2path' Name of the parent folder where the prediction results are stored
'--d2method' Name of the folder that your results are stored
for example, if your results are stored in '~/data/clocs_data/2D/cascade/***.txt'
'--d2path' should be '~/data/clocs_data/2D'
'--d2method' should be 'cascade'
'--d3path' and '--d3method' same as above
'--infopath' is the path of the file 'kitti_infos_trainval.pkl' produced by pcdet
'--inputpath' is where the input data stored
'--log-path' is the path where you want to store your model
```

if you modified above well, then just 

```
  python generate_data.py
```

then you can get the input data.

#### Train and Evaluation

1. Train the model. You need to modify the file `train.py` same as the `generate_data.py`.  If you skip 5, you need to change `'--generate'`to 1 to generate input data first. If you have generated before, just keep it 0.  Then  

```
python train.py
```

2. Validate your fusion results. Modify the file `eval.py` as above rule.  then

```
python eval.py
```

​	It will validate all your model, you can choose the best one.

### Some Tips

This implementation just store the input data, 3D detection data and 2D detection data in disks, so the training process is much faster than original CLOCs implementation. If you want to use CLOCs in other dataset or other detection method, you just need to modify the data pre-processing or just output the detection results as here defined. By following above, you can apply this fusion network in any method you want and it can be trained well in a short time.

### Acknowledgement

Our codes are inspired by [CLOCs](https://github.com/pangsu0613/CLOCs) a lot and are based on [PCDet](https://github.com/open-mmlab/OpenPCDet) and [MMdetection](https://github.com/open-mmlab/mmdetection). Thanks for their excellent work.




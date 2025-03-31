# ROAD-Waymo Baseline for ROAD, ROAD-Waymo and ROAD-Waymo-trans dataset
This repository contains code for SF3D-MDA, proposed along with [ROAD-Waymo dataset](https://github.com/salmank255/Road-waymo-dataset) and [ROAD dataset](https://github.com/gurkirt/road-dataset). This code contains training and evaluation for ROAD-Waymo, ROAD and Road-Waymo-trans datasets. 



## Table of Contents
- <a href='#Attribution'>Attribution</a>
- <a href='#requirements'>Requirements</a>
- <a href='#training-SF3D-MDA'>Training SF3D-MDA</a>
- <a href='#testing-and-building-tubes'>Testing and Building Tubes</a>
- <a href='#performance'>Performance</a>
- <a href='#citation'>Citation</a>
- <a href='#references'>Reference</a>


## Attribution

ROAD-Waymo-trans dataset was made using the Waymo Open Dataset, provided by Waymo LLC under license terms available at waymo.com/open.

By downloading or using the ROAD-Waymo-trans dataset and/or the Waymo Open Dataset, you are agreeing to the terms of the Waymo Open Dataset License Agreement for Non-Commercial Use—which includes a requirement that you only use the Waymo Open Dataset (or datasets built from it, such as the ROAD-Waymo and ROAD-Waymo-trans dataset) for noncommercial purposes. “Non-commercial Purposes" means research, teaching, scientific publication and personal experimentation. Non-commercial Purposes include use of the dataset to perform benchmarking for purposes of academic or applied research publication. Non-commercial Purposes does not include purposes primarily intended for or directed towards commercial advantage or monetary compensation, or purposes intended for or directed towards litigation, licensing, or enforcement, even in part.


## Requirements
We need three things to get started with training: datasets, kinetics pre-trained weight, and pytorch with torchvision and tensoboardX. 

### Dataset download an pre-process

- We currently support the following three dataset.
    - [ROAD-Waymo dataset](https://github.com/salmank255/Road-waymo-dataset)
    - [ROAD dataset](https://github.com/gurkirt/road-dataset) in dataset release [paper](https://arxiv.org/pdf/2102.11585.pdf)
    - [ROAD-Waymo-trans]Due to technical constraints, you need to manually process labels based on the content of the Label Processing folders to obtain relevant annotations.

- Visit [ROAD-Waymo dataset](https://github.com/salmank255/Road-waymo-dataset) for download and pre-processing. 
- Visit [ROAD dataset](https://github.com/gurkirt/road-dataset) for download and pre-processing. 


### Pytorch and weights

  - Install [Pytorch](https://pytorch.org/) and [torchvision](http://pytorch.org/docs/torchvision/datasets.html)
  - INstall tensorboardX viad `pip install tensorboardx`
  - Pre-trained weight on [kinetics-400](https://deepmind.com/research/open-source/kinetics). Download them by changing current directory to `kinetics-pt` and run the bash file [get_kinetics_weights.sh](./kinetics-pt/get_kinetics_weights.sh). OR Download them from  [Google-Drive](https://drive.google.com/drive/folders/1xERCC1wa1pgcDtrZxPgDKteIQLkLByPS?usp=sharing). Name the folder `kinetics-pt`, it is important to name it right. 



## Training SF3D-MDA
- We assume that you have downloaded and put dataset and pre-trained weight in correct places.    
- To train 3D-RetinaNet using the training script simply specify the parameters listed in `main.py` as a flag or manually change them.

Let's assume that you extracted dataset in `/home/user/road-waymo/` and weights in `/home/user/kinetics-pt/` directory then your train command from the root directory of this repo is going to be:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py /home/user/ /home/user/  /home/user/kinetics-pt/ --MODE=train --ARCH=resnet50 --MODEL_TYPE=I3D --DATASET=road_waymo --TEST_DATASET=road_waymo --TRAIN_SUBSETS=train --VAL_SUBSETS=val --SEQ_LEN=8 --TEST_SEQ_LEN=8 --BATCH_SIZE=4 --LR=0.0041
```

Second instance of `/home/user/` in above command specifies where checkpoint weight and logs are going to be stored. In this case, checkpoints and logs will be in `/home/user/road-waymo/cache/<experiment-name>/`.
```
--ARCH          ---> By default it's resent50 but our code also support resnet101
--MODEL_TYPE    ---> We support six different models including I3D and SlowFast
--DATASET       ---> Dataset specifiy the training dataset as we support multiple datasets including road, road_waymo, and roadpp (both combine)
--TEST_DATASET  ---> Dataset use for evaluation in training MODE
--TRAIN_SUBSETS ---> It will be train in all cased except road where we have multiple splits
--SEQ_LEN       ---> We did experiments for sequence length of 8 but we support other lenths as well
--TEST_SEQ_LEN  ---> Test sequence length is for prediction of frames at a time we support mutliple lens and tested from 8 to 32.
--BATCH_SIZE    ---> The batch size depends upon the number of GPUs and/or your GPU memory, if your GPU memory is 24 GB we recommend a batch per GPU. For A100 80GB of GPU we tested upto 5 batchs per GPU.
```

- Training notes:
  * The VRAM required for a single batch is 16GB, in this case, you will need 4 GPUs (each with at least 16GB VRAM) to run training.
  * During training checkpoint is saved every epoch also log it's frame-level `frame-mean-ap` on a subset of validation split test.
  * Crucial parameters are `LR`, `MILESTONES`, `MAX_EPOCHS`, and `BATCH_SIZE` for training process.
  * `label_types` is very important variable, it defines label-types are being used for training and validation time it is bummed up by one with `ego-action` label type. It is created in `data\dataset.py` for each dataset separately and copied to `args` in `main.py`, further used at the time of evaluations.
  * Event detection and triplet detection is used interchangeably in this code base. 




## Performance

## TODO




##### Download pre-trained weights






## Citation
This work will be published in the following article:
  
  @ARTICLE {yue2025joint,
author = {Chenyi, Yue and Lisheng, Jin and Baicang, Guo and Hongyu, Zhang and Junchen, Liu and Xingchen, Liu and Chuanqiang, An},
journal = {The Visual Computer},
title = {Joint Annotation and Recognition of Road User Behaviors for Autonomous Driving using Vehicle-Mounted Visual In-formation},
}



## References
The baseline model of this project is sourced from( https://github.com/gurkirt/3D-RetinaNet) and( https://github.com/salmank255/ROAD_Waymo_Baseline). The data set sources include:( https://github.com/gurkirt/road-dataset) and (https://github.com/salmank255/Road-waymo-dataset): 

  @ARTICLE {singh2022road,
author = {Singh, Gurkirt and Akrigg, Stephen and Di Maio, Manuele and Fontana, Valentina and Alitappeh, Reza Javanmard and Saha, Suman and Jeddisaravi, Kossar and Yousefi, Farzad and Culley, Jacob and Nicholson, Tom and others},
journal = {IEEE Transactions on Pattern Analysis & Machine Intelligence},
title = {ROAD: The ROad event Awareness Dataset for autonomous Driving},
year = {5555},
volume = {},
number = {01},
issn = {1939-3539},
pages = {1-1},
keywords = {roads;autonomous vehicles;task analysis;videos;benchmark testing;decision making;vehicle dynamics},
doi = {10.1109/TPAMI.2022.3150906},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {feb}
}


@inproceedings{singh2017online,
  title={Online real-time multiple spatiotemporal action localisation and prediction},
  author={Singh, Gurkirt and Saha, Suman and Sapienza, Michael and Torr, Philip HS and Cuzzolin, Fabio},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={3637--3646},
  year={2017}
}

@article{maddern20171,
  title={1 year, 1000 km: The Oxford RobotCar dataset},
  author={Maddern, Will and Pascoe, Geoffrey and Linegar, Chris and Newman, Paul},
  journal={The International Journal of Robotics Research},
  volume={36},
  number={1},
  pages={3--15},
  year={2017},
  publisher={SAGE Publications Sage UK: London, England}
}


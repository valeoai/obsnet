# Triggering Failures: Out-Of-Distribution detection by learning from local adversarial attacks in Semantic Segmentation
Victor Besnier, Andrei Bursuc, David Picard & Alexandre Briot 

In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) 2021

[Our paper](https://arxiv.org/abs/2108.01634)

## TL;DR
We present a new anomaly detection method for road semantic segmentation based on an observer network trained on the failure mode of the target network
![Alt text](img/archi.png "Observer architecture")


## Abstract
In this paper, we propose a new method, named Observer Network, for OOD and error detection for semantic segmentation. 
We separate the segmentation and the error prediction by using a dedicated network to the later task, keeping the segmentation network unchanged.
We trigger failures of the Segmentation Network by applying Local Adversarial Attacks (LAA) on the input image during training. These images serve as proxy OOD to train the observer.  
We show that our method is fast, accurate and memory efficient on three different datasets and compare against multiple baselines.

##  Repository Structure
    ├ Obsnet/
    |    ├── Models/                                <- networks
    |    |      ├── road_anomaly_networks/          <- networks from SegmentMeIfYouCan
    |    |      ├── load_net.py
    |    |      ├── deeplab_v3plus.py
    |    |      ├── resnet.py
    |    |      ├── obsnet.py
    |    |      └── segnet.py  
    |    |    
    |    ├── Dataset/                               <- loading  data
    |    |      ├── BDD_anomaly.py                  <- BDDAnomaly dataset     
    |    |      ├── camvid.py                       <- CamVid dataset     
    |    |      ├── cityscapes.py                   <- CityScapes dataset     
    |    |      ├── woodscapes.py                   <- WoodScape dataset     
    |    |      ├── cea.py                          <- CEA dataset     
    |    |      ├── load_data.py                    <- dataloader   
    |    |      ├── seg_transfo.py                  <- adapt pytorch data augmentation for segmentation     
    |    |      └── street_hazard.py                <- StreetHazards Dataset 
    |    |
    |    ├── Utils/                                 <- useful fct
    |    |      ├── adv_attack.py                   <- fct adversarial attacks      
    |    |      ├── affichage.py                    <- fct for plot viridis & segmentation map       
    |    |      ├── loss.py                         <- focal loss      
    |    |      ├── metrics.py                      <- metrics for evaluation     
    |    |      └── utils.py                        <- useful functions
    |    ├── ckpt/                                  <- models ckpt
    |    ├── logs/                                  <- tensorboard logs
    |    ├── img/                                   <- img for Teaser
    |    ├── train.py                               <- training the observer
    |    ├── evaluation.py                          <- test and evaluation
    |    ├── inference.py                           <- perform inference on custom images
    |    ├── README.md                              <- me :) 
    |    └── main.py                                <- main

## Usage
    
    $ git clone https://github.com/valeoai/obsnet
    $ cd obsnet 
    $ conda env create --file requirements.yml  
    $ conda activate obsnet
      
## Datasets

### CamVid
CamVid Dataset can be download here: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

The CamvidOOD dataset is composed of the same number of images as the classic CamVid. However, each image of the testing set contains an out-of-distribution animal.
You can download the CamVidOOD split in the release "CamVid OOD". Once CamVid and CamVidOOD are downloaded, execute the following line in your shell:
    
    $ cd /path/where/you/download/camvidood/
    $ unzip CamVidOOD.zip
    $ mv -r /path/where/you/download/camvidood/test_ood/ /path/to/dataset/CamVid/
    $ mv -r /path/where/you/download/camvidood/test_ood.txt /path/to/dataset/CamVid/
    $ mv -r /path/where/you/download/camvidood/testannot_ood/ /path/to/dataset/CamVid/
    
Folder Structure:

    ├ CamVid/
    |    ├ test/
    |    ├ testannot/
    |    ├ test_ood/
    |    ├ testannot_ood/
    |    ├ train/
    |    ├ trainannot/
    |    ├ val/
    |    ├ valannot/
    |    ├ test.txt
    |    ├ test_ood.txt
    |    ├ train.txt
    |    └ val.txt

To train an ObsNet on CamVid:

    python main.py --dset_folder "<path to dataset>" --segnet_file "<path to pretrain segnet>" --obsnet_file "./ckpt/camvid/" --data "CamVid" --tboard "./logs/camvid" --num_workers <nb workers> --bsize 8 --adv "min_random_patch" --epsilon 0.025 --lr 0.2 --nclass 12
     
To test:
    
    python main.py --dset_folder "<path to dataset>" --segnet_file "<path to pretrain segnet>" --obsnet_file "./ckpt/camvid/" --data "CamVid" --tboard "./logs/camvid" --num_workers <nb workers> --nclass 12 --test_only --test_multi "obsnet,mcp,mc_dropout" 

### StreetHazards
Dataset can be download here: https://github.com/hendrycks/anomaly-seg

Folder Structure:

    ├ StreetHazards
    |    ├ annotations/
    |    |    ├ test/
    |    |    ├ training/
    |    |    └ validation/
    |    ├ images/
    |    |    ├ test/
    |    |    ├ training/
    |    |    └ validation/
    |    ├ README.txt
    |    ├ test.odgt
    |    ├ train.odgt
    |    └ validation.odgt

To train an ObsNet on Streethazards:

    python main.py --dset_folder "<path to dataset>" --segnet_file "<path to pretrain segnet>" --obsnet_file "./ckpt/streethazards/" --data "StreetHazard" --tboard "./logs/streethazards" --num_workers <nb workers> --bsize 6 --adv "max_random_patch" --epsilon 0.001 --lr 0.02 --nclass 14

To test:
    
    python main.py --dset_folder "<path to dataset>" --segnet_file "<path to pretrain segnet>" --obsnet_file "./ckpt/streethazards/" --data "StreetHazard" --tboard "./logs/bdd" --num_workers <nb workers> --nclass 14 --test_only --test_multi "obsnet,mcp,mc_dropout" 

### BDD Anomaly
Dataset can be download here: https://github.com/hendrycks/anomaly-seg

Folder Structure:

    ├ BDD
    |    ├ bdd100k
    |    |    ├ seg
    |    |    |    ├ color_labels
    |    |    |    ├ images
    |    |    |    └ labels
    |    ├ test.odgt
    |    ├ train.odgt
    |    └ valiadation.odgt

To train an ObsNet on BDD Anomaly:    
    
    python main.py --dset_folder "<path to dataset>" --segnet_file "<path to pretrain segnet>" --obsnet_file "./ckpt/bdd/" --data "BddAnomaly" --tboard "./logs/bdd" --num_workers <nb workers> --bsize 6 --adv "max_random_patch" --epsilon 0.001 --lr 0.02 --nclass 19
    
To test:
    
    python main.py --dset_folder "<path to dataset>" --segnet_file "<path to pretrain segnet>" --obsnet_file "./ckpt/bdd/" --data "BddAnomaly" --tboard "./logs/bdd" --num_workers <nb workers> --nclass 19 --test_only --test_multi "obsnet,mcp,mc_dropout" 
        
## Inference 

You can perfom inference on a batch of images. 

Example:

    python inference.py --img_folder "<path to image folder>" --segnet_file "path to segnet" --obsnet_file "path to obsnet" --data "CityScapes" --model "raod_anomaly"    

## Road Anomaly Segmentation (SegmentMeIfYouCan)

| Model name                | Train w/ OoD       |  AUPR (&uarr;)  |  FPR95TPR (&darr;)  |  sIoU gt (&uarr;)  |  PPV (&uarr;)  |  mean F1 (&uarr;)  |
| --------------------------|:-----------------: | :-------------: | :-----------------: | :----------------: | :------------: | :----------------: |
| **ObsNet (OURS)**         |        :x:         |   **75.44%**  | 26.69%            |  **44.22%**      | **52.56%**   | **45.08%**       |
| NFlowJS                   |        :x:         |   56.92%      | 34.71%            |  36.94%          |  18.01%      | 14.89%           |
| JSRNet                    |        :x:         |   33.64%      | 43.85%            |  20.20%          |  29.27%      | 13.66%           |
| Image Resynthesis         |        :x:         |   52.28%      | **25.93%**        |  39.68%          |  10.95%      | 12.51%           |
| Embedding Density         |        :x:         |   37.52%      | 70.76%            |  33.86%          |  20.54%      | 7.90%            |
| Maximum Softmax           |        :x:         |   27.97%      | 72.05%            |  15.48%          |  15.29%      | 5.37%            |
| ODIN                      |        :x:         |   33.06%      | 71.68%            |  19.53%          |  17.88%      | 5.15%            |
| MC Dropout                |        :x:         |   28.87%      | 69.47%            |  20.49%          |  17.26%      | 4.26%            |
| Ensemble                  |        :x:         |   17.66%      | 91.06%            |  16.44%          |  20.77%      | 3.39%            |
| Mahalanobis               |        :x:         |   20.04%      | 86.99%            |  14.82%          |  10.22%      | 2.68%            |
| | | | | | | |
| DenseHybrid               | :heavy_check_mark: |   77.96%      |  9.81%            |  54.17%          |  24.13%      | 31.08%           |
| Maximized Entropy         | :heavy_check_mark: |   85.47%      | 15.00%            |  49.21%          |  39.51%      | 28.72%           |
| Void Classifier           | :heavy_check_mark: |   36.61%      | 63.49%            |  21.14%          |  22.13%      | 6.49%            |
| SynBoost                  | :heavy_check_mark: |   56.44%      | 61.86%            |  34.68%          |  17.81%      | 9.99%            |

Sample of our results: 

![Alt text](img/teaser_RA.gif)

## Pretrained Models

Pretrained model are available in this [google drive](https://drive.google.com/drive/folders/11S9oK-Bk9PoP2728ldrROQKAa8Q3iUxx?usp=sharing)

## Citation
If you find this repository usefull, please consider citing our [paper](https://arxiv.org/abs/2108.01634):

    @incollection{besnier2021trigger,
       title = {Triggering Failures: Out-Of-Distribution detection by learning from local adversarial attacks in Semantic Segmentation},
       author = {Besnier, Victor and Bursuc, Andrei and Picard, David and Briot Alexandre},
       booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
       year = {2021}
       url= {https://arxiv.org/abs/2108.01634}
    }

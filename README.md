# IPNet

This repository is an official implementation of the AAAI 2023 paper "Two Heads are Better than One: Image-Point Cloud Network for Depth-Based 3D Hand Pose Estimation".

> ####  [Two Heads are Better than One: Image-Point Cloud Network for Depth-Based 3D Hand Pose Estimation](https://ojs.aaai.org/index.php/AAAI/article/view/25310)
> ##### [Pengfei Ren](https://pengfeiren96.github.io/), [Chenyu Chen](https://scholar.google.com/citations?user=v8TFZI4AAAAJ), [Jiachang Hao](https://scholar.google.com/citations?user=XRR603kAAAAJ), [Haifeng Sun](https://scholar.google.com/citations?user=dwhbTsEAAAAJ), [Qi Qi](https://scholar.google.com/citations?user=2W2h0SwAAAAJ), [Jingyu Wang](https://jericwang.github.io/), [Jianxin Liao](https://www.researchgate.net/scientific-contributions/Jianxin-Liao-8024422)

##
## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.10
- CUDA (tested with cuda11.3)
- Other dependencies described in requirements.txt
- Install point operation
  ```bash
  pip install pointnet2_ops_lib/.
  ```
- Install [Manopth]('https://github.com/hassony2/manopth)


## Install MANO 

- Go to [MANO website]('https://mano.is.tue.mpg.de/')
- Download Models and Code (the downloaded file should have the format `mano_v*_*.zip`).
- unzip and copy the `models/MANO_RIGHT.pkl` into the `MANO` folder
- Your folder structure should look like this:
```
code/
  MANO/
    MANO_RIGHT.pkl
```
## Prepare Dataset
### DexYCB
- Download and decompress [DexYCB]('https://dex-ycb.github.io/') 
- Modify the `root_dir` in `config.py` according to your setting.
- Generate json file for data loading (dataloader/DEXYCB2COCO.py) 
- In order to speed up the training, you need to generate the hand mesh corresponding to each image according to the MANO annotation.
- Your folder structure should look like this:
```
DexYCB/
  mesh/
    20200709-subject-01/
        20200709_153548/
            932122062010/
                mesh_000000.txt
                ...
    ...
  20200709-subject-01/
  20200813-subject-02/
  ...
            
```
### NYU
- Download and decompress [NYU]('https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm') 
- Modify the `root_dir` in `config.py` according to your setting.

## Train

### DexYCB
```bash
python train_ho.py
```

### NYU
```bash
python train.py
```

Remember to change the dataset name in config.py accordingly.
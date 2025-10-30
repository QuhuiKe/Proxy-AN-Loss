# Proxy-AN Loss for Deep Metric Learning
This repository serves as the official PyTorch implementation for the paper: Proxy-AN Loss for Deep Metric Learning.

It offers source code for replicating the experiments conducted on four benchmark datasets (CUB200, Cars196, SOP, and InShop).

## Requirements
+ Python 3.8
+ PyTorch 1.8.1+cu111
+ numpy
+ tqdm
+ tensorboardX
+ scikit-learn
+ scipy

## Datasets
1. Download four benchmark datasets.
    + [CUB-200-2011](https://data.caltech.edu/records/65de6-vp158)
    + [Cars196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
    + [Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/)
    + [InShop Clothes Retrieval](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
2. Extract the tgz or zip file into `./data/dataset_name/original/` folder and run the data convert scripts to transform data format. In particular, the InShop dataset does not require any data transformation.
```
python scripts/data_process/xxx_convert.py
```
3. The data folder is constructed as followed:
```
data:
├──  CUB200/CARS196/SOP
│  └──  class 
│    ├──  train 
│    │  ├──  Catrgory 1
│    │  ├──  Catrgory ...
│    │  └──  Catrgory M
│    └──  test
│       ├──  Catrgory 1
│       ├──  Catrgory ...
│       └──  Catrgory N
└──  IN_SHOP
   ├── img  
   │  ├──  MEN
   │  └──  WOMEN
   └──  list_eval_partition.txt

```

## Training Process

1. Run the training process under full data setting.
```
bash train.sh
```

2. Run the training process under partial data setting (only support for CUB200 and Cars196 datasets).
```
bash partial.sh
```

3. Run the training process under class imbalance setting (only support for CUB200 and Cars196 datasets).
```
bash imbalance.sh
```

## Citation
```
@article{peng2025proxy,
  title = {Proxy-AN Loss for Deep Metric Learning},
  author = {Peng, Wenjie and Ke, Quhui and Liang, Jinglin and Huang, Shuangping and Chen, Tianshui}
  journal = {Neural Networks},
  pages = {108254},
  year = {2025},
  doi = {10.1016/j.neunet.2025.108254},
}
```
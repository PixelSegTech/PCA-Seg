# PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation


Abstract:—Recent advances in vision-language models (VLMs) have garnered substantial attention in open-vocabulary semantic and part segmentation (OSPS). However, existing methods extract image-text alignment cues from cost volumes through a serial structure of spatial and class aggregations, leading to knowledge interference between class-level semantics and spatial context. Therefore, this paper proposes a simple yet effective parallel cost aggregation (PCA-Seg) paradigm to alleviate the above challenge, enabling the model to capture richer vision-language alignment information from cost volumes. Specifically, we design an expert-driven perceptual learning (EPL) module that efficiently integrates semantic and contextual streams. It incorporates a multi-expert parser to extract complementary features from multiple perspectives. In addition, a coefficient mapper is designed to adaptively learn pixel-specific weights for each feature, enabling the integration of complementary knowledge into a unified and robust feature embedding. Furthermore, we propose a feature orthogonalization decoupling (FOD) strategy to mitigate redundancy between the semantic and contextual streams, which allows the EPL module to learn diverse knowledge from orthogonalized features. Extensive experiments on eight benchmarks show that each parallel block in PCA-Seg adds merely 0.35M parameters while achieving state-of-the-art OSPS performance.

# Pipeline
 ![Network](https://github.com/PixelSegTech/PCA-Seg/tree/main/arch.png)



## 🌈Environment
- Linux with Python == 3.10.0
- CUDA 11.7 or CUDA 11.8
- The provided environment is suggested for reproducing our results, similar configurations may also work.

## 🚀Quick Start

### 1. Create Conda Environment
```
conda create -n PCA-Seg python=3.10.0
conda activate PCA-Seg
Install a torch that matches your CUDA version from the official website: https://pytorch.org/get-started/previous-versions/, The environment we are using is CUDA11.7+TORCH2.0.0.

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
pip install -r requirements.txt
pip install -e . -v
```

### 2. Dataset Preparation
We follow CAT-Seg for preparing datasets. Thanks to the CAT-Seg's high-quality open source. Please use this [Link](https://github.com/cvlab-kaist/CAT-Seg/blob/main/datasets/README.md) to organize the dataset. Don't forget to set `os.environ["DETECTRON2_DATASETS"] = "path_to_your_dataset"` in `train_net.py`. For example: `os.environ["DETECTRON2_DATASETS"] = "/mnt/SSD8T/home/wjj/dataset".`

### 3. Training
Before starting the training, please modify the paths in the training config `configs/eva_vitb_384.yaml` and `eva_vitl_336.yaml`.

**Note:**  
In our current code, the default PCA-Seg dense feature extraction method is set to `csa` (i.e., SCLIP, qq+kk). If your PCA-Seg is distilled using the `qq` mode, please modify the input parameter of the `encode_dense` function to `mode='qq'` in two places in `cat_seg/cat_seg_model.py` (corresponding to training and inference, respectively).
``` 
CLIP_PRETRAINED: "EVA02-CLIP-B-16" 
CACHE_DIR: "path_to_your_declip_ckpt"
``` 

To train the PCA-Seg with CAT-Seg, please run the following script:
``` 
sh run.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [OPTS]

# For DeCLIP-B variant
sh run.sh configs/eva_vitb_384.yaml 4 output/

# For DeCLIP-L variant
sh run.sh configs/eva_vitl_336.yaml 4 output/
```
### 4. Evaluation
```
sh run.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [OPTS]

# For DeCLIP-B variant
sh eval.sh configs/eva_vitb_384.yaml 4 output/ MODEL.WEIGHTS path/to/trained_weights.pth

# For DeCLIP-L variant
sh eval.sh configs/eva_vitl_336.yaml 4 output/ MODEL.WEIGHTS path/to/trained_weights.pth
```




## 🙏 Citing PCA-Seg 

```bibtex
@InProceedings{Yin_2026_CVPR,
    author    = {Yin, Jianjian and Chen, Tao and Chen, Yi and Pei, Gensheng and Shu, Xiangbo and Yao, Yazhou and Shen, Fumin},
    title     = {PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {27633-27643}
}
```
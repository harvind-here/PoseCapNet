# Multi-Task Person Detection & Description

## Architecture
- Encoder: ResNet34 (pretrained ImageNet)
- Caption: BERT (6 layers)
- Pose: HRNet-style network
- Input size: 224x224
- Shared feature dimension: 512

## Training Results
Best model achieved w/ (coco dataset):
- Pose Loss: 0.0825
- Caption Loss: 0.1691      ------ can be slightly even more less like < 0.15 but acceptable i g
- Validation Loss: 0.2517

Training specs:
- Batch size: 16
- Learning rate: 1e-4
- Device: CUDA
- Dataset split: 70-20-10

## Test Results
Model evaluation metrics:
- Pose MSE: 0.1210 (good - within 0.1-0.15 range)
- BLEU-1: 0.2246 (needs improvement, target >0.4) -----shud work on BERT (layer norm/grad clippin)
- BLEU-4: 0.1758 (close to target 0.2)

## Quick Setup
```bash
pip install pytorch torchvision
pip install transformers nltk tqdm
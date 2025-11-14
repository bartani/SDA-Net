# SDA-Net

## Low-Light Image Enhancement via Self-Degradation-Aware and Semantic-Perceptual Guidance Networks
### Authors: O. Sedeeq Ahmad, S.A. Anjuman, S. Sulaiman, Ako Bartani
### Published in: Knowledge-Based Systems (2025)

### If you see this message, it means we are uploading our codes, please do not download them.

#### This paper proposed SDA-Net, a self-degradation-aware enhancement framework that dynamically adapts enhancement strategies to varying low-light conditions. Our method introduces two key components: (1) a Contrastive-Based Degradation Feature Extractor, which learns discriminative representations of degradation, enabling adaptive enhancement tailored to degradation features. This degradation-aware approach enhances stability and performance in different low-light scenarios. (2) a Semantic-Perceptual Guidance Network, which generates intermediate visual representations to amplify inputs. These representations serve as prior knowledge to enhance the images, enabling sharper edge reconstruction, effective noise suppression, and the preservation of natural illumination.

![Alt text](img/visualperformance.jpg)

## Requirements
Python 3.8+

PyTorch >= 1.10 (CUDA-enabled for GPU training)

torchvision

numpy, scipy, pillow

opencv-python

scikit-image

tqdm, yaml, matplotlib

(optional) TensorBoard or WandB for logging


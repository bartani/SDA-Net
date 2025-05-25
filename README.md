# SDA-Net

## Low-Light Image Enhancement via Degradation-Aware and Semantic-Perceptual Guidance Networks

Low-light images in real-world scenarios suffer from varying levels of degradation. However, existing LLIE methods typically apply a one-size-fits-all enhancement strategy, regardless of the input degradation severity. As a result, they often fail to adapt to different degradation levels and structures. This limitation frequently leads to suboptimal and inconsistent performance, as these models lack explicit awareness of degradation severity—resulting in either over-enhancement or insufficient enhancement of the input images. With this motivation, we propose SDA-Net (Self-Degradation-Aware Network), a novel LLIE framework that dynamically adapts enhancement strategies based on input degradation characteristics. Our approach introduces two key innovations: First, a Contrastive-Based Degradation Feature Extractor (CBDFE) that learns discriminative representations to degradation severity at global and local levels, enabling adaptive enhancement tailored to degradation patterns. Second, a Semantic and Perceptual Guidance Network (SPG-Net) that generates semantic-aware visual priors to amplify the inputs. By delegating structural and coarse-scale recovery to SPG-Net, SDA-Net adapts the enhancement process based on the degradation features extracted by CBDFE and allows the network to focus more effectively on refining brightness and texture. This separation of responsibilities enables a more robust and interpretable enhancement process. Extensive experiments demonstrate that SDA-Net outperforms state-of-the-art methods on benchmark datasets, showing superior robustness in real-world scenarios with complex degradation patterns


![Alt text](img/visualperformance.pdf)

## News
-Paper Download (coming soon)

-Training Code (coming soon)

-Pre-trained Models (coming soon)

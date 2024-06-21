# Sparse-Color-Code-Net

<p align="center">
    <img src="https://github.com/smartslab/Sparse-Color-Code-Net/blob/main/pipeline.png" width="640"> <br />
    <em> Pipeline </em>
</p>

## Abstract
As robotics and augmented reality applications increasingly rely on precise and efficient 6D object pose estimation, real-time performance on edge devices is required for more interactive and responsive systems. Our proposed Sparse Color-Code Net (SCCN) embodies a clear and concise pipeline design to effectively address this requirement. SCCN performs pixel-level predictions on the target object in the RGB image, utilizing the sparsity of essential object geometry features to speed up the Perspective-n-Point (PnP) computation process. Additionally, it introduces a novel pixel-level geometry-based object symmetry representation that seamlessly integrates with the initial pose predictions, effectively addressing symmetric object ambiguities. SCCN notably achieves an estimation rate of 19 frames per second (FPS) and 6 FPS on the benchmark LINEMOD and LINEMOD Occlusion dataset, respectively, for an NVIDIA Jetson AGX Xavier, while consistently maintaining high estimation accuracy at these rates. 

## Prerequisites
* Python >=3.6
* Pytorch >= 1.9.0
* Torchvision >= 0.10.0
* CUDA >= 10.1

## Usage
### Color-Code rendering: 
* use `colorcode_render.py` to render the Color-Code images
### Data augmentation:
* use `colorcode_augment_prepare.py` to generate the augmented training images
### Training:
* use `colorcode_train_seg.py` to train the segmentation section of the pipeline
* use `colorcode_train_seg_multi.py` to train the multi-object segmentation section of the pipeline
* use `colorcode_train_cc.py` to train the color-code estimation section of the pipeline
* use `colorcode_train_cc_multi.py` to train the multi-object color-code estimation section of the pipeline

## Acknowledgement
The code used to render the color-code is based on neural_renderer (https://github.com/daniilidis-group/neural_renderer). Thank you for developing such great tool!

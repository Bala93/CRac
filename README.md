# Class and Region-Adaptive Constraints for Network Calibration (CRac)

## Abstract
In this work, we present a novel approach to calibrate segmentation networks that considers the inherent challenges posed by different categories and object regions. In particular, we present a formulation that integrates class and region-wise constraints into the learning objective, with multiple penalty weights to account for class and region differences. Finding the optimal penalty weights manually, however, might be unfeasible, and potentially hinder the optimization process. To overcome this limitation, we propose an approach based on Class and Region-Adaptive constraints (CRaC), which allows to learn the class and region-wise penalty weights during training. CRaC is based on a general Augmented Lagrangian method, a well-established technique in constrained optimization. Experimental results on two popular segmentation benchmarks, and two well-known segmentation networks, demonstrate the superiority of CRaC compared to existing approaches.

## ACDC
### UNet
```python
python tools/train_net.py wandb.enable=True task="lagmedseg" data="cardiac" model="unet" model.num_classes="4" loss="ce" +lag="spatial_bndry_cls_aug_lag" lag.rho=0.1 lag.margin="5" optim="adam" scheduler="step" wandb.project="unet-cardiac"
```
### NNUnet 
```python
python tools/train_net.py wandb.enable=True task="lagmedseg" data="cardiac" model="nnunet" model.num_classes="4" loss="ce" +lag="spatial_bndry_cls_aug_lag" lag.rho=0.1 lag.margin="5" optim="adam" scheduler="step" wandb.project="unet-cardiac"
```
## FLARE
### UNet
```python
python tools/train_net.py wandb.enable=True task="lagmedseg" data="abdomen" data.batch_size="16" model="unet" model.num_classes="5" loss="ce" +lag="spatial_bndry_cls_aug_lag" lag.rho=0.1 lag.margin=5 optim="adam" scheduler="step" wandb.project="unet-abdomen"
```
### NNUnet 
```python
python tools/train_net.py wandb.enable=True task="lagmedseg" data="abdomen" data.batch_size="16" model="nnunet" model.num_classes="5" loss="ce" +lag="spatial_bndry_cls_aug_lag" lag.rho=0.1 lag.margin=5 optim="adam" scheduler="step" wandb.project="unet-abdomen"
```

## Installation

conda create -n crac python=3.8

conda activate crac

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

pip install -r requirements.txt

python setup.py develop
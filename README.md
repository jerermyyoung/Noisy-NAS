# NAS on noisy data
## Requirements
```
Python == 3.7.4, PyTorch == 1.1.0, torchvision == 0.3.0
```
## Dataset

## Neural Architecture Search
python train_search.py

## Architecture Evaluation
python train.py --auxiliary --cutout # CIFAR-10
python train_tinyimagenet.py --auxiliary            # Tiny ImageNet

## Architecture Test
python test.py --auxiliary --model_path cifar10_model.pt # CIFAR-10
python test_tinyimagenet.py --auxiliary --model_path imagenet_model.pt # Tiny Imagenet

## Acknowledgement
The codes of this paper are based on the codes of DARTS (https://github.com/quark0/darts). We appreciate DARTS's codes and thank the authors of DARTS.

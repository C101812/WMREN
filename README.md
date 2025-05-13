#WMREN
Code for paper "Wavelet Multi-scale Region-Enhanced Network for Medical Image Segmentationr". 

## 1. Environment

- Please prepare an environment with Python 3.8.20, PyTorch 2.4.1, and CUDA 11.8.

## 2. Train/Test

- Train

```bash
python train.py --dataset Synapse --root_path your DATA_DIR --max_epochs 400 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24
```

- Test 

```bash
python test.py --dataset Synapse --is_savenii --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 400 --base_lr 0.05 --img_size 224 --batch_size 24
```

## References
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)

```bibtex
@article{huang2021missformer,
  title={MISSFormer: An Effective Medical Image Segmentation Transformer},
  author={Huang, Xiaohong and Deng, Zhifang and Li, Dandan and Yuan, Xueguang},
  journal={arXiv preprint arXiv:2109.07162},
  year={2021}
}
```

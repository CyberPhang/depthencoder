# depthencoder

A U-Net that takes an RGB image and a corrupted depth map and predicts the clean depth.

The network has five levels and runs at 224×224. RGB and depth are concatenated into 4 channels, and the skip connections are concatenations. It has 2.3 million parameters. The output is one depth channel in `[0, 1]`.

## Data

RGB images are from Pixmo, resized to 224. Depth targets are Depth Anything V2 maps saved as 8-bit JPEG. The split is 80/10/10 with seed 493, which leaves 4,807 test images.

Training corrupts the depth with Gaussian noise (σ = 0.1), rectangular holes filled with 0, and quantization to 32 levels. The loss is L1 plus a multi-scale gradient loss against the clean depth.

## Train

From the repo root:

```
python train.py
```

The best checkpoint is `models/checkpoints/unet_best.pth`. Weights and the image directories are gitignored.

## Evaluate

Clean reconstruction on the test set:

```
python eval.py
```

Denoising, with each corruption scored separately against leaving it uncorrected:

```
python eval_denoise.py
```

## Test results

Clean input, clean target: MAE 0.0051, gradient loss 0.0076, SSIM 0.9955.

| Corruption | Repaired MAE | Left corrupted |
|---|---|---|
| Hole, whole region | 0.095 | 0.439 |
| Hole, center pixel | 0.132 | 0.441 |
| Gaussian noise, σ = 0.1 | 0.015 | 0.076 |
| Quantized to 32 levels | 0.011 | 0.015 |

MAE is the mean of `|prediction - clean depth|` on a depth scale of 0 to 1. Multiply by 255 for gray levels. The corruptions are synthetic. The labels are estimated depth stored as JPEG, not raw sensor depth.

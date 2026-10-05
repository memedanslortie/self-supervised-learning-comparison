# Self-Supervised Visual Representation Learning: A Comparison

> Coursework, Deep Learning, M2 Vision & Machine Intelligence, Université Paris Cité (2025).
> Built on the [Berkeley CS294-158 self-supervised learning demos](https://github.com/wilson1yan/cs294-158-ssl)
> as adapted by C. Kurtz.

This notebook compares **pretext tasks for self-supervised learning** by training a linear
probe on frozen features. My additions:

1. **Relative Position Prediction** ([Doersch et al., ICCV 2015](https://arxiv.org/abs/1505.05192)),
   implemented from scratch ([`relative_position_prediction.py`](relative_position_prediction.py)).
   Given a central patch and one of its 8 neighbors, the network predicts the neighbor's
   position.
2. A **unified benchmark** of the four methods under the same linear-probe protocol.
3. A **generalization test on CIFAR-100**: how well do features learned for 10 classes
   transfer to 100 classes?

## Results (linear probe, test accuracy)

| Pretext task | CIFAR-10 | CIFAR-100 | Drop |
|---|:-:|:-:|:-:|
| SimCLR (contrastive) | **92.84 %** | **49.92 %** | −42.9 |
| Rotation prediction | 79.91 % | 49.90 % | −30.0 |
| Relative position *(my implementation)* | 59.98 % | 30.56 % | −29.4 |
| Context encoder (inpainting) | 45.77 % | 22.74 % | −23.0 |

**Takeaways**
- The contrastive objective (SimCLR) clearly learns the strongest features on the source
  distribution.
- Its advantage disappears on CIFAR-100, where Rotation prediction matches it. The more
  "semantic" the pretext task, the better the transfer.
- The notebook also covers **shortcut solutions** that make a pretext task trivial
  (chromatic aberration, color-histogram cues) and the usual fixes: color dropping and
  color jitter.

## Contents

| File | Description |
|---|---|
| `selfsupervised_demos.ipynb` | Context encoder, rotation, SimCLR, downstream segmentation (Pascal VOC), shortcut analysis, comparison and CIFAR-100 study |
| `relative_position_prediction.py` | Relative Position Prediction model (NIN backbone on CIFAR, AlexNet on ImageNet) |

## Run

The notebook is meant for Google Colab with a GPU runtime and downloads its pretrained
weights automatically.

**Stack:** PyTorch · torchvision · pandas · matplotlib

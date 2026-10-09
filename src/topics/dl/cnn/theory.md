## Why convolutions exist

A fully connected layer on a $224\times224\times3$ image has $150{,}528$ inputs; one hidden layer of 1000 units already costs **150 million weights**, and none of them know that pixels have neighbours. Images have two exploitable regularities:

1. **Locality** — useful features (edges, corners, textures) live in small neighbourhoods.
2. **Translation equivariance** — an edge is an edge wherever it appears.

A convolution bakes both in: one small kernel is **slid across the whole image** (weight sharing), so the parameter count is independent of image size and the same detector fires everywhere. If you shift the input, the feature map shifts with it — *equivariance*. (Invariance, "is there a cat anywhere?", only appears later through pooling and global aggregation.)

<div class="callout">

**Key idea.** A CNN is an MLP with two hard-coded priors: *connect only locally* and *share weights across positions*. Everything else — depth, normalization, skip connections — is about making that stack trainable.

</div>

## Anatomy of a convolutional layer

A layer maps a tensor $C_{in}\times H\times W$ to $C_{out}\times H'\times W'$. It owns $C_{out}$ filters, each of shape $C_{in}\times k\times k$; output channel $o$ is the sum over **all** input channels of a 2-D cross-correlation, plus a bias. The shape-changing knobs:

| Knob | Effect | Typical use |
|---|---|---|
| **Kernel size** $k$ | Size of the local neighbourhood | 3 (dominant), 1 (channel mixing), 7 (stems) |
| **Stride** $s$ | Step between windows → downsamples by $s$ | 2 to halve resolution |
| **Padding** $p$ | Zeros around the border | "same" ($p=\lfloor k/2\rfloor$) keeps size |
| **Dilation** $d$ | Gaps inside the kernel → bigger field, same params | Segmentation, WaveNet |
| **Groups** | Split channels into independent groups | Depthwise conv = groups $=C_{in}$ |

Open the **2D convolution** lab above: the purple window is one multiply-accumulate; the cyan cell is where the result lands. Switch the kernel to see hand-designed filters (Sobel, Laplacian) — a CNN *learns* such kernels from data instead.

## Pooling, strides and downsampling

Spatial resolution is traded for channels as you go deeper (watch the 3D volume view: slabs get thicker while their faces shrink). Max-pooling keeps the strongest activation in each window and gives small translation invariance; average pooling smooths. Modern nets often use **strided convolutions** instead (learnable downsampling) and finish with **global average pooling**, which replaced the huge fully connected heads of AlexNet/VGG.

## Receptive field and the feature hierarchy

The *receptive field* of a unit is the patch of the input that can affect it. With stride-1 $3\times3$ layers it grows by 2 per layer (linear); with strides, pooling or dilation it grows **exponentially** (last tab of the lab). Early layers detect edges and colour blobs, middle layers textures and parts, late layers object-level concepts. This hierarchy is why a deep stack of tiny kernels works so well.

## Making it deep: BatchNorm and residuals

Plain stacks degrade beyond ~20 layers — not from overfitting but from *optimization* difficulty. Two fixes unlocked 100+ layers:

- **Batch Normalization** standardizes activations per channel, stabilizing and accelerating training and permitting larger learning rates.
- **Residual connections** (ResNet, 2015): $y = x + F(x)$. The block only has to learn a *correction*, and the identity path gives gradients a highway all the way back.

The **bottleneck** block ($1\times1$ reduce → $3\times3$ → $1\times1$ expand) keeps compute low in deep variants (ResNet-50/101/152).

## The architecture family tree

| Year | Model | Contribution |
|---|---|---|
| 1998 | LeNet-5 | Conv + pool + FC; digits |
| 2012 | AlexNet | ReLU, dropout, GPU training — started the deep learning era |
| 2014 | VGG | Depth via uniform $3\times3$ stacks |
| 2014 | Inception | Multi-scale branches, $1\times1$ reductions |
| 2015 | ResNet | Skip connections, 100+ layers |
| 2017 | MobileNet | Depthwise-separable convs for edge devices |
| 2019 | EfficientNet | Compound scaling of depth/width/resolution |
| 2022 | ConvNeXt | A ResNet modernized with Transformer-era recipes; competitive with ViTs |

## Where CNNs still win

Vision Transformers dominate large-scale pre-training, but CNNs remain the pragmatic choice with **limited data** (the built-in priors act as a regularizer), on **edge hardware** (convs map beautifully to INT8 accelerators), and in **dense prediction** (U-Net for segmentation, YOLO-style detectors, super-resolution, diffusion U-Nets). Audio spectrograms and even 1-D sequences (WaveNet, TCNs) use the same ideas.

<div class="callout tip">

**Link to the GPU track.** A convolution is executed as an implicit matrix multiply (im2col / implicit GEMM) on Tensor Cores. The tiling ideas in *Shared Memory & Tiled MatMul* are exactly what cuDNN does under the hood.

</div>

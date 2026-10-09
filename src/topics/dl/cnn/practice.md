## Exercises

**Q1.** ResNet's stem is a $7\times7$ conv, stride 2, padding 3, 64 filters on a $224\times224\times3$ image. What are the output shape and the number of parameters?

<details>
<summary>Show answer</summary>

Output size: $\lfloor(224+6-7)/2\rfloor+1=112$, so the output is $64\times112\times112$.

Parameters: $64\cdot(3\cdot49+1)=9{,}472$. MACs: $112^2\cdot64\cdot3\cdot49\approx118$ M — a large share of compute for just one layer, which is why later blocks use $3\times3$ kernels.

</details>

**Q2.** Compare one $5\times5$ conv with two stacked $3\times3$ convs ($C$ channels in and out, ignoring bias).

<details>
<summary>Show answer</summary>

Both have a $5\times5$ receptive field. Parameters: $25C^2$ versus $2\cdot9C^2=18C^2$ (28% fewer). The stack also inserts an extra non-linearity and is cheaper in MACs by the same ratio. This is the VGG argument.

</details>

**Q3.** You swap a $3\times3$ conv with $C_{in}=C_{out}=256$ for a depthwise-separable one. How much compute do you save, and why might the wall-clock gain be smaller?

<details>
<summary>Show answer</summary>

Ratio $=1/256+1/9\approx0.115$, i.e. **~8.7× fewer MACs**. But depthwise convs have very low arithmetic intensity (each channel is processed independently), so they are *memory-bound* and often fall far short of the theoretical speed-up on GPUs. See the roofline lab in the CUDA track.

</details>

**Q4.** Three $3\times3$ stride-1 convs, then a $2\times2$ max-pool with stride 2, then one more $3\times3$ conv. What is the final receptive field?

<details>
<summary>Show answer</summary>

Use $r_\ell=r_{\ell-1}+(k-1)j_{\ell-1}$:

- conv1: $r=3,\ j=1$; conv2: $r=5$; conv3: $r=7$
- pool: $r=7+1\cdot1=8,\ j=2$
- conv4: $r=8+2\cdot2=12$

The pool doubled the "reach" of every later $3\times3$ layer.

</details>

**Q5.** Why does BatchNorm misbehave with a batch size of 2, and what are the alternatives?

<details>
<summary>Show answer</summary>

The batch statistics $\mu_{\mathcal B},\sigma^2_{\mathcal B}$ become very noisy (and train/eval behaviour diverges). Use **GroupNorm** or **LayerNorm**, which normalize within a sample, or accumulate/synchronize BN statistics across GPUs (SyncBN).

</details>

**Q6 (code).** Implement a $3\times3$ conv as a single matrix multiply and verify it against <code>torch.nn.functional.conv2d</code>.

<details>
<summary>Show answer</summary>

See `conv_scratch.py` in the **Code** tab: `im2col` builds the $(H'W')\times(C_{in}k^2)$ patch matrix with `unfold`, then one `@` performs the whole layer.

</details>

## In practice

**Transfer learning recipe.** Start from an ImageNet-pretrained backbone, replace the classifier head, train the head for 1–3 epochs with the backbone frozen, then unfreeze and fine-tune with a learning rate 10–100× smaller for the backbone (discriminative LRs). Use the *same* mean/std normalization the checkpoint was trained with — a classic silent bug.

**Throughput tips (NVIDIA GPUs).**
- Use `channels_last` memory format and mixed precision (`torch.autocast`) so convolutions hit Tensor Cores.
- Choose channel counts that are multiples of 8 (FP16) or 16.
- `torch.backends.cudnn.benchmark = True` for fixed input sizes.
- Fuse Conv+BN+ReLU for inference (`torch.fx`, TensorRT).

**Typical applications.** Image classification, object detection (YOLO, RetinaNet), semantic segmentation (U-Net, DeepLab), medical imaging, OCR, defect inspection on edge devices, spectrogram-based audio tagging, and the denoising U-Nets inside diffusion models.

**Resolution vs depth vs width.** For a fixed FLOP budget, scaling all three together (EfficientNet's *compound scaling*) beats scaling one alone. Resolution is the most expensive: cost grows with $H\cdot W$.

## Common pitfalls

- Forgetting `model.eval()` — BatchNorm and Dropout then behave like training.
- Padding/stride mismatches between your framework and an exported ONNX/TensorRT model, producing off-by-one feature map sizes.
- Strided convolutions and pooling cause **aliasing**; the output is not exactly shift-equivariant. Anti-aliased (blur-pool) variants help.
- Tiny batch + BatchNorm (see Q5), or BatchNorm before the final logits in a regression head.
- Data leakage via augmentation applied before train/val split.

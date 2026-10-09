import{r as e}from"./rolldown-runtime-hePW80VL.js";import{a as t,b as n,n as r,r as i,v as a}from"./r3f-x1z21uF6.js";import{a as o,c as s,d as c,f as l,g as u,i as d,t as f,y as p}from"./viz-CPys2405.js";import{t as m}from"./Stage3D-CWOT2b2l.js";import{a as h,c as g,i as _,l as v,o as y,r as b,s as x,t as S,u as C}from"./hooks-Dw7oo1m3.js";var w=e(n(),1),T=`## Why convolutions exist

A fully connected layer on a $224\\times224\\times3$ image has $150{,}528$ inputs; one hidden layer of 1000 units already costs **150 million weights**, and none of them know that pixels have neighbours. Images have two exploitable regularities:

1. **Locality** — useful features (edges, corners, textures) live in small neighbourhoods.
2. **Translation equivariance** — an edge is an edge wherever it appears.

A convolution bakes both in: one small kernel is **slid across the whole image** (weight sharing), so the parameter count is independent of image size and the same detector fires everywhere. If you shift the input, the feature map shifts with it — *equivariance*. (Invariance, "is there a cat anywhere?", only appears later through pooling and global aggregation.)

<div class="callout">

**Key idea.** A CNN is an MLP with two hard-coded priors: *connect only locally* and *share weights across positions*. Everything else — depth, normalization, skip connections — is about making that stack trainable.

</div>

## Anatomy of a convolutional layer

A layer maps a tensor $C_{in}\\times H\\times W$ to $C_{out}\\times H'\\times W'$. It owns $C_{out}$ filters, each of shape $C_{in}\\times k\\times k$; output channel $o$ is the sum over **all** input channels of a 2-D cross-correlation, plus a bias. The shape-changing knobs:

| Knob | Effect | Typical use |
|---|---|---|
| **Kernel size** $k$ | Size of the local neighbourhood | 3 (dominant), 1 (channel mixing), 7 (stems) |
| **Stride** $s$ | Step between windows → downsamples by $s$ | 2 to halve resolution |
| **Padding** $p$ | Zeros around the border | "same" ($p=\\lfloor k/2\\rfloor$) keeps size |
| **Dilation** $d$ | Gaps inside the kernel → bigger field, same params | Segmentation, WaveNet |
| **Groups** | Split channels into independent groups | Depthwise conv = groups $=C_{in}$ |

Open the **2D convolution** lab above: the purple window is one multiply-accumulate; the cyan cell is where the result lands. Switch the kernel to see hand-designed filters (Sobel, Laplacian) — a CNN *learns* such kernels from data instead.

## Pooling, strides and downsampling

Spatial resolution is traded for channels as you go deeper (watch the 3D volume view: slabs get thicker while their faces shrink). Max-pooling keeps the strongest activation in each window and gives small translation invariance; average pooling smooths. Modern nets often use **strided convolutions** instead (learnable downsampling) and finish with **global average pooling**, which replaced the huge fully connected heads of AlexNet/VGG.

## Receptive field and the feature hierarchy

The *receptive field* of a unit is the patch of the input that can affect it. With stride-1 $3\\times3$ layers it grows by 2 per layer (linear); with strides, pooling or dilation it grows **exponentially** (last tab of the lab). Early layers detect edges and colour blobs, middle layers textures and parts, late layers object-level concepts. This hierarchy is why a deep stack of tiny kernels works so well.

## Making it deep: BatchNorm and residuals

Plain stacks degrade beyond ~20 layers — not from overfitting but from *optimization* difficulty. Two fixes unlocked 100+ layers:

- **Batch Normalization** standardizes activations per channel, stabilizing and accelerating training and permitting larger learning rates.
- **Residual connections** (ResNet, 2015): $y = x + F(x)$. The block only has to learn a *correction*, and the identity path gives gradients a highway all the way back.

The **bottleneck** block ($1\\times1$ reduce → $3\\times3$ → $1\\times1$ expand) keeps compute low in deep variants (ResNet-50/101/152).

## The architecture family tree

| Year | Model | Contribution |
|---|---|---|
| 1998 | LeNet-5 | Conv + pool + FC; digits |
| 2012 | AlexNet | ReLU, dropout, GPU training — started the deep learning era |
| 2014 | VGG | Depth via uniform $3\\times3$ stacks |
| 2014 | Inception | Multi-scale branches, $1\\times1$ reductions |
| 2015 | ResNet | Skip connections, 100+ layers |
| 2017 | MobileNet | Depthwise-separable convs for edge devices |
| 2019 | EfficientNet | Compound scaling of depth/width/resolution |
| 2022 | ConvNeXt | A ResNet modernized with Transformer-era recipes; competitive with ViTs |

## Where CNNs still win

Vision Transformers dominate large-scale pre-training, but CNNs remain the pragmatic choice with **limited data** (the built-in priors act as a regularizer), on **edge hardware** (convs map beautifully to INT8 accelerators), and in **dense prediction** (U-Net for segmentation, YOLO-style detectors, super-resolution, diffusion U-Nets). Audio spectrograms and even 1-D sequences (WaveNet, TCNs) use the same ideas.

<div class="callout tip">

**Link to the GPU track.** A convolution is executed as an implicit matrix multiply (im2col / implicit GEMM) on Tensor Cores. The tiling ideas in *Shared Memory & Tiled MatMul* are exactly what cuDNN does under the hood.

</div>
`,E=`## The operation

Deep-learning "convolution" is really **cross-correlation** (no kernel flip). For input $x\\in\\mathbb R^{C_{in}\\times H\\times W}$, kernel $w\\in\\mathbb R^{C_{out}\\times C_{in}\\times k\\times k}$ and bias $b$:

$$
y_{o,i,j}=b_o+\\sum_{c=1}^{C_{in}}\\sum_{u=0}^{k-1}\\sum_{v=0}^{k-1} w_{o,c,u,v}\\; x_{c,\\;si+du-p,\\;sj+dv-p}
$$

with stride $s$, dilation $d$ and zero padding $p$ (reads outside the image return $0$).

## Output size

$$
H'=\\left\\lfloor \\frac{H+2p-d\\,(k-1)-1}{s}\\right\\rfloor+1
$$

For $d=1$: $H'=\\lfloor (H+2p-k)/s\\rfloor+1$. "Same" padding at stride 1 needs $p=(k-1)/2$.

## Parameters and cost

$$
\\text{params}=C_{out}\\,(C_{in}k^2+1),\\qquad
\\text{MACs}=H'W'\\,C_{out}\\,C_{in}\\,k^2 .
$$

Note that params do **not** depend on $H,W$, but compute does. Doubling resolution quadruples MACs.

**Depthwise-separable** factorization (a $k\\times k$ depthwise conv per channel followed by a $1\\times1$ pointwise conv):

$$
\\frac{\\text{MACs}_{sep}}{\\text{MACs}_{std}}=\\frac{H'W'\\,C_{in}(k^2+C_{out})}{H'W'\\,C_{out}C_{in}k^2}=\\frac1{C_{out}}+\\frac1{k^2}.
$$

## Convolution is a matrix multiply (im2col)

Unfold every $k\\times k\\times C_{in}$ patch into a row of a matrix $X_{col}\\in\\mathbb R^{(H'W')\\times(C_{in}k^2)}$ and flatten the filters into $W_{col}\\in\\mathbb R^{(C_{in}k^2)\\times C_{out}}$:

$$
Y = X_{col}\\,W_{col} + b .
$$

This is how GEMM-based libraries (and Tensor Cores) execute convolutions.

## Backpropagation through a convolution

Let $L$ be the loss and $\\delta=\\partial L/\\partial y$.

**Gradient w.r.t. the kernel** — a correlation of the input with the upstream gradient:

$$
\\frac{\\partial L}{\\partial w_{o,c,u,v}}=\\sum_{i,j}\\delta_{o,i,j}\\;x_{c,\\,si+du-p,\\,sj+dv-p}
$$

**Gradient w.r.t. the input** — a *full* convolution of $\\delta$ with the 180°-rotated kernel (a transposed convolution):

$$
\\frac{\\partial L}{\\partial x_{c,m,n}}=\\sum_{o}\\sum_{u,v}w_{o,c,u,v}\\;\\delta_{o,\\,(m+p-du)/s,\\,(n+p-dv)/s}
$$

where terms with non-integer indices are dropped. Because the same $w$ is reused at every position, its gradient **accumulates** over all positions — that is weight sharing in the backward pass.

## Receptive-field recurrence

Track the receptive field $r_\\ell$ and the "jump" $j_\\ell$ (input pixels between adjacent units):

$$
r_\\ell=r_{\\ell-1}+d_\\ell\\,(k_\\ell-1)\\,j_{\\ell-1},\\qquad j_\\ell=j_{\\ell-1}\\,s_\\ell,\\qquad r_0=j_0=1 .
$$

Stride-1 $3\\times3$ stacks: $r_L=2L+1$ (linear). Dilations $1,2,4,\\dots$: $r_L=2^{L+1}-1$ (exponential).

## Normalization and residuals

**BatchNorm** (per channel, mini-batch $\\mathcal B$):

$$
\\hat x=\\frac{x-\\mu_{\\mathcal B}}{\\sqrt{\\sigma^2_{\\mathcal B}+\\epsilon}},\\qquad y=\\gamma\\hat x+\\beta .
$$

At inference $\\mu,\\sigma^2$ are running averages, so BN can be **folded** into the preceding conv's weights.

**Residual block** $y=x+F(x)$ gives the Jacobian

$$
\\frac{\\partial y}{\\partial x}=I+\\frac{\\partial F}{\\partial x},
$$

so across $L$ blocks the gradient contains the term $\\prod_\\ell (I+J_\\ell)=I+\\sum_\\ell J_\\ell+\\dots$ — a path of **exactly the identity** that cannot vanish.

## He initialization

To keep activation variance constant through ReLU layers, initialize with

$$
w\\sim\\mathcal N\\!\\left(0,\\;\\frac{2}{C_{in}k^2}\\right).
$$
`,D=`## Exercises

**Q1.** ResNet's stem is a $7\\times7$ conv, stride 2, padding 3, 64 filters on a $224\\times224\\times3$ image. What are the output shape and the number of parameters?

<details>
<summary>Show answer</summary>

Output size: $\\lfloor(224+6-7)/2\\rfloor+1=112$, so the output is $64\\times112\\times112$.

Parameters: $64\\cdot(3\\cdot49+1)=9{,}472$. MACs: $112^2\\cdot64\\cdot3\\cdot49\\approx118$ M — a large share of compute for just one layer, which is why later blocks use $3\\times3$ kernels.

</details>

**Q2.** Compare one $5\\times5$ conv with two stacked $3\\times3$ convs ($C$ channels in and out, ignoring bias).

<details>
<summary>Show answer</summary>

Both have a $5\\times5$ receptive field. Parameters: $25C^2$ versus $2\\cdot9C^2=18C^2$ (28% fewer). The stack also inserts an extra non-linearity and is cheaper in MACs by the same ratio. This is the VGG argument.

</details>

**Q3.** You swap a $3\\times3$ conv with $C_{in}=C_{out}=256$ for a depthwise-separable one. How much compute do you save, and why might the wall-clock gain be smaller?

<details>
<summary>Show answer</summary>

Ratio $=1/256+1/9\\approx0.115$, i.e. **~8.7× fewer MACs**. But depthwise convs have very low arithmetic intensity (each channel is processed independently), so they are *memory-bound* and often fall far short of the theoretical speed-up on GPUs. See the roofline lab in the CUDA track.

</details>

**Q4.** Three $3\\times3$ stride-1 convs, then a $2\\times2$ max-pool with stride 2, then one more $3\\times3$ conv. What is the final receptive field?

<details>
<summary>Show answer</summary>

Use $r_\\ell=r_{\\ell-1}+(k-1)j_{\\ell-1}$:

- conv1: $r=3,\\ j=1$; conv2: $r=5$; conv3: $r=7$
- pool: $r=7+1\\cdot1=8,\\ j=2$
- conv4: $r=8+2\\cdot2=12$

The pool doubled the "reach" of every later $3\\times3$ layer.

</details>

**Q5.** Why does BatchNorm misbehave with a batch size of 2, and what are the alternatives?

<details>
<summary>Show answer</summary>

The batch statistics $\\mu_{\\mathcal B},\\sigma^2_{\\mathcal B}$ become very noisy (and train/eval behaviour diverges). Use **GroupNorm** or **LayerNorm**, which normalize within a sample, or accumulate/synchronize BN statistics across GPUs (SyncBN).

</details>

**Q6 (code).** Implement a $3\\times3$ conv as a single matrix multiply and verify it against <code>torch.nn.functional.conv2d</code>.

<details>
<summary>Show answer</summary>

See \`conv_scratch.py\` in the **Code** tab: \`im2col\` builds the $(H'W')\\times(C_{in}k^2)$ patch matrix with \`unfold\`, then one \`@\` performs the whole layer.

</details>

## In practice

**Transfer learning recipe.** Start from an ImageNet-pretrained backbone, replace the classifier head, train the head for 1–3 epochs with the backbone frozen, then unfreeze and fine-tune with a learning rate 10–100× smaller for the backbone (discriminative LRs). Use the *same* mean/std normalization the checkpoint was trained with — a classic silent bug.

**Throughput tips (NVIDIA GPUs).**
- Use \`channels_last\` memory format and mixed precision (\`torch.autocast\`) so convolutions hit Tensor Cores.
- Choose channel counts that are multiples of 8 (FP16) or 16.
- \`torch.backends.cudnn.benchmark = True\` for fixed input sizes.
- Fuse Conv+BN+ReLU for inference (\`torch.fx\`, TensorRT).

**Typical applications.** Image classification, object detection (YOLO, RetinaNet), semantic segmentation (U-Net, DeepLab), medical imaging, OCR, defect inspection on edge devices, spectrogram-based audio tagging, and the denoising U-Nets inside diffusion models.

**Resolution vs depth vs width.** For a fixed FLOP budget, scaling all three together (EfficientNet's *compound scaling*) beats scaling one alone. Resolution is the most expensive: cost grows with $H\\cdot W$.

## Common pitfalls

- Forgetting \`model.eval()\` — BatchNorm and Dropout then behave like training.
- Padding/stride mismatches between your framework and an exported ONNX/TensorRT model, producing off-by-one feature map sizes.
- Strided convolutions and pooling cause **aliasing**; the output is not exactly shift-equivariant. Anti-aliased (blur-pool) variants help.
- Tiny batch + BatchNorm (see Q5), or BatchNorm before the final logits in a regression head.
- Data leakage via augmentation applied before train/val split.
`,O=`import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


class BasicBlock(nn.Module):
    """Two 3x3 convs with a skip connection: y = relu(x + F(x))."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        # 1x1 projection when the shape changes, otherwise identity
        self.skip = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class TinyResNet(nn.Module):
    def __init__(self, num_classes=10, width=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        self.layers = nn.Sequential(
            BasicBlock(width, width),
            BasicBlock(width, width * 2, stride=2),    # 32 -> 16
            BasicBlock(width * 2, width * 4, stride=2),  # 16 -> 8
            BasicBlock(width * 4, width * 8, stride=2),  # 8  -> 4
        )
        self.head = nn.Linear(width * 8, num_classes)

        for m in self.modules():  # He initialization
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.layers(self.stem(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # global average pooling
        return self.head(x)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
    test = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=test_tf)
    train_dl = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test, batch_size=256, num_workers=2)

    model = TinyResNet().to(device).to(memory_format=torch.channels_last)
    epochs = 15
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=epochs * len(train_dl))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

        model.eval()  # BatchNorm now uses running statistics
        correct = 0
        with torch.no_grad():
            for x, y in test_dl:
                correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        print(f"epoch {epoch + 1:2d}  loss {loss.item():.3f}  test acc {correct / len(test):.3%}")


if __name__ == "__main__":
    main()
`,k=`import numpy as np


def conv2d_naive(x, w, b, stride=1, pad=0):
    """x: (C_in, H, W)   w: (C_out, C_in, k, k)   b: (C_out,)  ->  (C_out, H', W')"""
    c_out, c_in, k, _ = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    h_out = (xp.shape[1] - k) // stride + 1
    w_out = (xp.shape[2] - k) // stride + 1
    y = np.zeros((c_out, h_out, w_out))
    for o in range(c_out):
        for i in range(h_out):
            for j in range(w_out):
                patch = xp[:, i * stride:i * stride + k, j * stride:j * stride + k]
                y[o, i, j] = np.sum(patch * w[o]) + b[o]
    return y


def im2col(x, k, stride=1, pad=0):
    """Unfold every k*k*C_in patch into a row -> (H'*W', C_in*k*k)."""
    c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    h_out = (h + 2 * pad - k) // stride + 1
    w_out = (w + 2 * pad - k) // stride + 1
    cols = np.empty((h_out * w_out, c * k * k))
    for i in range(h_out):
        for j in range(w_out):
            cols[i * w_out + j] = xp[:, i * stride:i * stride + k, j * stride:j * stride + k].ravel()
    return cols, h_out, w_out


def conv2d_gemm(x, w, b, stride=1, pad=0):
    """The whole layer is ONE matrix multiply (this is what GPUs do)."""
    c_out, c_in, k, _ = w.shape
    cols, h_out, w_out = im2col(x, k, stride, pad)
    y = cols @ w.reshape(c_out, -1).T + b   # (H'W', C_in*k*k) @ (C_in*k*k, C_out)
    return y.T.reshape(c_out, h_out, w_out)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 12, 12))
    w = rng.standard_normal((8, 3, 3, 3))
    b = rng.standard_normal(8)

    y1 = conv2d_naive(x, w, b, stride=2, pad=1)
    y2 = conv2d_gemm(x, w, b, stride=2, pad=1)
    print("shape:", y1.shape, " max |naive - gemm| =", np.abs(y1 - y2).max())

    try:  # optional cross-check against PyTorch
        import torch
        import torch.nn.functional as F
        y3 = F.conv2d(torch.tensor(x)[None], torch.tensor(w), torch.tensor(b), stride=2, padding=1)[0].numpy()
        print("max |gemm - torch|  =", np.abs(y2 - y3).max())
    except ImportError:
        pass
`,A=a(),j=12,M=[`............`,`.##########.`,`.##########.`,`.........##.`,`........##..`,`.......##...`,`......##....`,`.....##.....`,`....##......`,`....##......`,`....##......`,`............`];function N(e){let t=Array.from({length:j},()=>Array(j).fill(0));if(e===`seven`)M.forEach((e,n)=>[...e].forEach((e,r)=>{t[n][r]=+(e===`#`)}));else if(e===`shapes`){for(let e=1;e<5;e++)for(let n=1;n<5;n++)t[e][n]=1;for(let e=1;e<11;e++)t[e][8]=1;for(let e=0;e<6;e++)t[5+e][1+e]=1;for(let e=4;e<11;e++)t[9][e]=1}else{for(let e=0;e<j;e++)for(let n=0;n<j;n++)t[e][n]=n/11*.6;for(let e=3;e<9;e++)for(let n=3;n<9;n++)t[e][n]=1}return t}var P={identity:[[0,0,0],[0,1,0],[0,0,0]],sobelx:[[-1,0,1],[-2,0,2],[-1,0,1]],sobely:[[-1,-2,-1],[0,0,0],[1,2,1]],blur:Array.from({length:3},()=>[,,,].fill(1/9)),sharpen:[[0,-1,0],[-1,5,-1],[0,-1,0]],laplace:[[0,1,0],[1,-4,1],[0,1,0]],emboss:[[-2,-1,0],[-1,1,1],[0,1,2]]},F=[[`sobelx`,`Sobel-X (vertical edges)`],[`sobely`,`Sobel-Y (horizontal edges)`],[`laplace`,`Laplacian`],[`blur`,`Box blur`],[`sharpen`,`Sharpen`],[`emboss`,`Emboss`],[`identity`,`Identity`]];function I(e,t,n,r){let i=e.length,a=t.length,o=Math.floor((i+2*r-a)/n)+1,s=(t,n)=>t>=0&&n>=0&&t<i&&n<i?e[t][n]:0;return Array.from({length:o},(e,i)=>Array.from({length:o},(e,o)=>{let c=0;for(let e=0;e<a;e++)for(let l=0;l<a;l++)c+=t[e][l]*s(i*n-r+e,o*n-r+l);return c}))}function L(e){let t=Math.floor(e.length/2);return Array.from({length:t},(n,r)=>Array.from({length:t},(t,n)=>Math.max(e[2*r][2*n],e[2*r][2*n+1],e[2*r+1][2*n],e[2*r+1][2*n+1])))}function R(){let[e,t]=(0,w.useState)(`seven`),[n,r]=(0,w.useState)(`sobelx`),[i,a]=(0,w.useState)(1),[l,m]=(0,w.useState)(0),[h,v]=(0,w.useState)(!1),[y,T]=(0,w.useState)(!0),[E,D]=(0,w.useState)(10),[O,k]=(0,w.useState)(!0),M=(0,w.useRef)({i:0,acc:0}),R=(0,w.useMemo)(()=>N(e),[e]),z=P[n],B=(0,w.useMemo)(()=>I(R,z,i,l),[R,z,i,l]),V=B.length,H=(0,w.useMemo)(()=>y?B.map(e=>e.map(e=>Math.max(0,e))):B,[B,y]),U=(0,w.useMemo)(()=>h&&V>=2?L(H):null,[h,H,V]),W=(0,w.useMemo)(()=>Math.max(1e-6,...H.flat().map(Math.abs)),[H]);(0,w.useEffect)(()=>{M.current={i:0,acc:0}},[e,n,i,l]);let[G]=S(390,(e,t,n,r,a)=>{let m=M.current,h=V*V;if(O)for(m.acc+=a*E;m.acc>=1;)--m.acc,m.i=(m.i+1)%(h+8);let g=Math.min(m.i,h-1),_=Math.min(m.i+1,h),v=Math.floor(g/V),b=g%V,x=j+2*l,S=t<560?20:30,C=t<560?18:34,w=U?U.length:0,T=d((t-28-C*(U?3:2)-3*S)/(x+V+w),5,22),D=14+x*T+C,k=D+3*S+C,A=k+V*T+C;p(e,`Input ${j}×${j}${l?` + pad ${l}`:``}`,14,24,{size:12,color:f.mute,weight:600}),p(e,`Kernel 3×3`,D,24,{size:12,color:f.mute,weight:600}),p(e,`${y?`ReLU(`:``}Feature map${y?`)`:``} ${V}×${V}`,k,24,{size:12,color:f.mute,weight:600}),U&&p(e,`Max-pool 2×2 → ${w}×${w}`,A,24,{size:12,color:f.mute,weight:600});for(let t=0;t<x;t++)for(let n=0;n<x;n++){let r=t-l,i=n-l,a=r>=0&&i>=0&&r<j&&i<j;e.fillStyle=a?c(`#10131f`,`#e7e9f4`,R[r][i]):`#0d0f1a`,e.fillRect(14+n*T,56+t*T,T-1,T-1),a||(e.strokeStyle=`#2a2f4a`,e.setLineDash([2,2]),e.strokeRect(14+n*T+.5,56+t*T+.5,T-2,T-2),e.setLineDash([]))}let N=14+b*i*T,P=56+v*i*T;e.fillStyle=`rgba(139,123,255,.18)`,e.fillRect(N,P,3*T,3*T),e.strokeStyle=f.a,e.lineWidth=2.5,e.strokeRect(N,P,3*T,3*T);for(let t=0;t<3;t++)for(let n=0;n<3;n++)e.fillStyle=o(z[t][n]/2),u(e,D+n*S,56+t*S,S-2,S-2,4),e.fill(),p(e,Math.abs(z[t][n])<.2&&z[t][n]!==0?`⅑`:String(+z[t][n].toFixed(1)),D+n*S+S/2-1,56+t*S+S/2,{size:11,align:`center`,mono:!0,color:`#fff`});e.strokeStyle=f.a,e.lineWidth=1.5,e.setLineDash([4,3]),e.beginPath(),e.moveTo(N+3*T,P),e.lineTo(D,56),e.moveTo(N+3*T,P+3*T),e.lineTo(D,56+3*S),e.stroke(),e.setLineDash([]);for(let t=0;t<V;t++)for(let n=0;n<V;n++)e.fillStyle=t*V+n<_?o(H[t][n]/W):`#0d0f1a`,e.fillRect(k+n*T,56+t*T,T-1,T-1);if(e.strokeStyle=f.b,e.lineWidth=2.5,e.strokeRect(k+b*T,56+v*T,T-1,T-1),U)for(let t=0;t<w;t++)for(let n=0;n<w;n++)e.fillStyle=(2*t+1)*V+2*n+1<_?o(U[t][n]/W):`#0d0f1a`,e.fillRect(A+n*T*2,56+t*T*2,T*2-1,T*2-1);let F=0,I=[];for(let e=0;e<3;e++)for(let t=0;t<3;t++){let n=v*i-l+e,r=b*i-l+t,a=n>=0&&r>=0&&n<j&&r<j?R[n][r]:0;F+=a*z[e][t],a*z[e][t]!==0&&I.push(`${(a*z[e][t]).toFixed(2).replace(/\.?0+$/,``)}`)}let L=56+Math.max(x*T,3*S)+34;p(e,`y[${v},${b}] = Σ x·k = ${I.length?I.slice(0,7).join(` + `).replace(/\+ -/g,`− `)+(I.length>7?` …`:``):`0`} = ${F.toFixed(2).replace(/\.?0+$/,``)||`0`}`,14,L,{size:12.5,mono:!0,color:f.ink}),p(e,`O = ⌊(N + 2P − K)/S⌋ + 1 = ⌊(${j} + ${2*l} − 3)/${i}⌋ + 1 = ${V}      MACs = ${V}²×9 = ${s(V*V*9)}`,14,L+24,{size:12.5,mono:!0,color:f.b})},{animate:!0});return(0,A.jsxs)(A.Fragment,{children:[(0,A.jsx)(`canvas`,{...G}),(0,A.jsxs)(_,{children:[(0,A.jsx)(x,{label:`Image`,value:e,onChange:t,options:[[`seven`,`Digit 7`],[`shapes`,`Shapes`],[`grad`,`Gradient + box`]]}),(0,A.jsx)(x,{label:`Kernel`,value:n,onChange:r,options:F}),(0,A.jsx)(g,{label:`Stride`,min:1,max:3,value:i,onChange:a}),(0,A.jsx)(g,{label:`Padding`,min:0,max:2,value:l,onChange:m}),(0,A.jsx)(C,{label:`ReLU`,value:y,onChange:T}),(0,A.jsx)(C,{label:`Max-pool`,value:h,onChange:v}),(0,A.jsx)(g,{label:`Speed`,min:2,max:40,value:E,onChange:D,fmt:e=>e+`/s`}),(0,A.jsx)(b,{onClick:()=>k(e=>!e),children:O?`Pause`:`Play`}),(0,A.jsx)(b,{onClick:()=>{M.current={i:0,acc:0}},children:`Restart`})]})]})}function z(e){let t=[{name:`Input image`,ch:3,hw:64,params:0,rf:1,op:`RGB 64×64`}],n=3,r=64,i=1,a=1,o=(e,o,s)=>{let c=n*o*9+o;r=Math.floor((r+2-3)/s)+1,i+=2*a,a*=s,t.push({name:e,ch:o,hw:r,params:c,rf:i,op:`3×3 conv, stride ${s}, + BN + ReLU`}),n=o};return o(`conv1`,e,1),o(`conv2`,e*2,2),o(`conv3`,e*4,2),o(`conv4`,e*8,2),t.push({name:`global avg-pool`,ch:n,hw:1,params:0,rf:Math.min(64,i+2*a*4),op:`average over H×W`}),t.push({name:`fc (10 classes)`,ch:10,hw:1,params:n*10+10,rf:64,op:`linear layer`}),t}function B({layer:e,x:t,sel:n,onSelect:a,color:o}){let s=.22+Math.log2(Math.max(2,e.ch))*.2,c=e.hw===1?.28:e.hw/16;return(0,A.jsxs)(`group`,{position:[t,0,0],children:[(0,A.jsxs)(`mesh`,{onClick:e=>{e.stopPropagation(),a()},onPointerOver:()=>document.body.style.cursor=`pointer`,onPointerOut:()=>document.body.style.cursor=``,children:[(0,A.jsx)(`boxGeometry`,{args:[s,c,c]}),(0,A.jsx)(`meshStandardMaterial`,{color:o,transparent:!0,opacity:n?.85:.5,emissive:o,emissiveIntensity:n?.5:.12}),(0,A.jsx)(r,{color:n?`#ffffff`:o})]}),(0,A.jsx)(i,{position:[0,-2.6,0],center:!0,distanceFactor:9,style:{pointerEvents:`none`},children:(0,A.jsxs)(`div`,{style:{font:`600 11px Inter Variable,sans-serif`,color:n?`#fff`:`#9aa1c0`,whiteSpace:`nowrap`,textAlign:`center`},children:[e.name,(0,A.jsx)(`br`,{}),(0,A.jsxs)(`span`,{style:{font:`500 10px JetBrains Mono Variable,monospace`,color:`#22d3ee`},children:[e.ch,`@`,e.hw,`×`,e.hw]})]})})]})}function V({x0:e,x1:n}){let r=(0,w.useRef)();return t(({clock:t})=>{r.current&&(r.current.position.x=e+t.elapsedTime*1.4%1*(n-e))}),(0,A.jsxs)(`mesh`,{ref:r,children:[(0,A.jsx)(`boxGeometry`,{args:[.04,4.4,4.4]}),(0,A.jsx)(`meshBasicMaterial`,{color:`#22d3ee`,transparent:!0,opacity:.1})]})}function H(){let[e,t]=(0,w.useState)(16),[n,r]=(0,w.useState)(1),i=(0,w.useMemo)(()=>z(e),[e]),a=(0,w.useMemo)(()=>{let e=0;return i.map((t,n)=>{let r=.22+Math.log2(Math.max(2,t.ch))*.2;return n&&(e+=(.22+Math.log2(Math.max(2,i[n-1].ch))*.2)/2+r/2+.75),e})},[i]),o=a[a.length-1]/2,c=i.reduce((e,t)=>e+t.params,0),l=i[Math.min(n,i.length-1)],u=[`#e7e9f4`,`#8b7bff`,`#7c8cff`,`#5aa8ff`,`#22d3ee`,`#4ade80`,`#f472b6`];return(0,A.jsxs)(A.Fragment,{children:[(0,A.jsxs)(m,{height:400,camera:[o,3.2,12],target:[o,0,0],overlay:(0,A.jsxs)(A.Fragment,{children:[(0,A.jsx)(`b`,{children:l.name}),` — `,l.op,(0,A.jsx)(`br`,{}),`output `,l.ch,`×`,l.hw,`×`,l.hw,` · params `,s(l.params),(0,A.jsx)(`br`,{}),`receptive field ≈ `,l.rf,`×`,l.rf,` px`]}),children:[i.map((e,t)=>(0,A.jsx)(B,{layer:e,x:a[t],sel:t===n,onSelect:()=>r(t),color:u[t%u.length]},t)),(0,A.jsx)(V,{x0:a[0],x1:a[a.length-1]})]}),(0,A.jsxs)(_,{children:[(0,A.jsx)(g,{label:`Base channels`,min:8,max:64,step:8,value:e,onChange:t}),(0,A.jsxs)(y,{children:[`Total parameters: `,(0,A.jsx)(`b`,{children:s(c)}),` · click a slab to inspect it · note how `,(0,A.jsx)(`b`,{children:`depth (channels) grows`}),` while `,(0,A.jsx)(`b`,{children:`spatial size shrinks`})]})]})]})}var U=[{id:`k3`,name:`3×3 stride 1`,color:f.a,layer:()=>({k:3,s:1,d:1})},{id:`k3s2`,name:`3×3, stride 2 every layer`,color:f.b,layer:()=>({k:3,s:2,d:1})},{id:`dil`,name:`3×3 dilated (1,2,4,8…)`,color:f.e,layer:e=>({k:3,s:1,d:2**e})},{id:`k3pool`,name:`3×3 conv + 2×2 pool`,color:f.d,layer:e=>e%2?{k:2,s:2,d:1}:{k:3,s:1,d:1}}];function W(e,t){let n=1,r=1,i=[1];for(let a=0;a<t;a++){let{k:t,s:o,d:s}=e.layer(a);n+=(t-1)*s*r,r*=o,i.push(n)}return i}function G(){let[e,t]=(0,w.useState)(6),[n,r]=(0,w.useState)(`k3`),i=U.map(e=>({c:e,v:W(e,10)})),a=i.find(e=>e.c.id===n),[o]=S(320,(t,r,o)=>{let s=r*.55-50,c=o-60,u=Math.max(...i.map(e=>e.v[10]),20),d=e=>20+c-Math.log(e)/Math.log(u)*c,m=e=>50+e/10*s;t.strokeStyle=f.grid,t.lineWidth=1;for(let e=0;e<=10;e+=2)t.beginPath(),t.moveTo(m(e),20),t.lineTo(m(e),20+c),t.stroke(),p(t,e,m(e),20+c+14,{size:11,align:`center`,color:f.mute,mono:!0});[1,3,10,30,100,300,1e3].filter(e=>e<=u).forEach(e=>{t.beginPath(),t.moveTo(50,d(e)),t.lineTo(50+s,d(e)),t.stroke(),p(t,e,42,d(e),{size:11,align:`right`,color:f.mute,mono:!0})}),p(t,`layers →`,50+s/2,o-8,{size:11,align:`center`,color:f.mute}),p(t,`receptive field (log scale)`,50,8,{size:11,color:f.mute}),i.forEach(({c:r,v:i})=>{l(t,i.map((e,t)=>[m(t),d(e)]),r.color,r.id===n?3:1.6),t.fillStyle=r.color,t.beginPath(),t.arc(m(e),d(i[e]),4.5,0,7),t.fill()}),t.strokeStyle=`rgba(255,255,255,.25)`,t.setLineDash([3,3]),t.beginPath(),t.moveTo(m(e),20),t.lineTo(m(e),20+c),t.stroke(),t.setLineDash([]);let h=Math.min(r*.4/41,(o-60)/41),g=r*.58;p(t,`Selected: ${a.c.name} @ ${e} layers`,g,12,{size:11.5,color:f.mute});let _=a.v[e];for(let e=0;e<41;e++)for(let n=0;n<41;n++){let r=Math.abs(n-20)<=(_-1)/2&&Math.abs(e-20)<=(_-1)/2;t.fillStyle=r?a.c.color:`#141830`,t.globalAlpha=r?.85:1,t.fillRect(g+n*h,30+e*h,h-.5,h-.5)}t.globalAlpha=1,t.fillStyle=`#fff`,t.fillRect(g+20*h,30+20*h,h,h),p(t,`RF = ${_}×${_}${_>41?` (exceeds this 41px view)`:``}`,g,30+41*h+16,{size:12,mono:!0,color:f.ink})});return(0,A.jsxs)(A.Fragment,{children:[(0,A.jsx)(`canvas`,{...o}),(0,A.jsxs)(_,{children:[(0,A.jsx)(g,{label:`Layers`,min:1,max:10,value:e,onChange:t}),(0,A.jsx)(x,{label:`Highlight`,value:n,onChange:r,options:U.map(e=>[e.id,e.name])}),(0,A.jsx)(h,{items:U.map(e=>[e.color,e.name])})]})]})}function K(){return(0,A.jsx)(v,{views:[{id:`conv`,label:`2D convolution`,render:()=>(0,A.jsx)(R,{})},{id:`vol`,label:`3D feature volumes`,render:()=>(0,A.jsx)(H,{})},{id:`rf`,label:`Receptive field growth`,render:()=>(0,A.jsx)(G,{})}]})}var q={Lab:K,vizTitle:`Slide a kernel, build a CNN in 3D, grow a receptive field`,tryIt:[`Pick **Sobel-X** on the digit 7 and watch only the vertical strokes light up; switch to **Sobel-Y** for the horizontal bar.`,`Set **stride = 2** and **padding = 1** and read the output-size formula update live.`,`In the 3D view, drag the *Base channels* slider and watch the parameter count scale ~quadratically.`,`In the receptive-field view compare plain 3×3 stacks with **dilated** convolutions — the gap is exponential.`],theory:T,math:E,practice:D,code:[{title:`A small ResNet for CIFAR-10 (PyTorch)`,lang:`python`,note:`Residual blocks, BatchNorm, global average pooling, AMP and channels_last — the practical defaults.`,src:O},{title:`Convolution from scratch: loops vs im2col`,lang:`python`,note:`Shows that a convolution is just one big matrix multiply — exactly what the GPU does (see the CUDA track).`,src:k}],quiz:[{q:`A $7\\times7$ conv with stride 2, padding 3 on a $224\\times224$ input gives an output of size:`,options:[`$224$`,`$112$`,`$111$`,`$109$`],answer:1,why:`$\\lfloor(224+6-7)/2\\rfloor+1 = 112$. This is the ResNet stem.`},{q:`Why do two stacked $3\\times3$ convs usually beat one $5\\times5$ conv?`,options:[`They have more parameters`,`Same receptive field, fewer parameters, extra non-linearity`,`They are faster only on CPUs`,`They avoid padding`],answer:1,why:`$2\\cdot9C^2=18C^2 < 25C^2$ with the same $5\\times5$ field and an additional ReLU.`},{q:`The gradient of the loss with respect to the *input* of a conv layer is computed by:`,options:[`A max-pool`,`A transposed convolution with the same kernel`,`A matrix inverse`,`Batch normalization`],answer:1,why:`Backprop through a convolution is a convolution of the upstream gradient with the flipped kernel (a transposed conv).`},{q:`A depthwise-separable $3\\times3$ conv with $C_{in}=C_{out}=256$ costs about what fraction of a standard conv?`,options:[`1/2`,`1/4`,`≈1/9`,`≈1/100`],answer:2,why:`Ratio $=1/C_{out}+1/k^2 \\approx 0.004+0.111\\approx 0.115$.`},{q:`What does the skip connection in a residual block mainly fix?`,options:[`Overfitting`,`Optimization of very deep nets (gradient flow)`,`Memory use`,`Translation invariance`],answer:1,why:`$\\partial(x+F(x))/\\partial x = I + \\partial F/\\partial x$ guarantees a direct gradient path.`}]};export{q as default};
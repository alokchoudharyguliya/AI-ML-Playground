import torch
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

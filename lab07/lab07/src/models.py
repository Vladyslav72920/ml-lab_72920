import torch
import torch.nn as nn

class OversizedCNN(nn.Module):
    def __init__(self, c1=64, c2=128, c3=256, fc=384):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4, fc),
            nn.ReLU(),
            nn.Linear(fc, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class StudentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4, 224),
            nn.ReLU(),
            nn.Linear(224, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def build_structured_pruned_model(teacher, keep_ratio):
    c1 = max(1, int(round(64 * keep_ratio)))
    c2 = max(1, int(round(128 * keep_ratio)))
    c3 = max(1, int(round(256 * keep_ratio)))
    pruned = OversizedCNN(c1=c1, c2=c2, c3=c3, fc=384)

    with torch.no_grad():
        w1 = teacher.features[0].weight.data
        idx1 = torch.topk(w1.view(w1.size(0), -1).norm(p=2, dim=1), c1).indices
        pruned.features[0].weight.data.copy_(w1[idx1])
        pruned.features[0].bias.data.copy_(teacher.features[0].bias.data[idx1])
        pruned.features[1].weight.data.copy_(teacher.features[1].weight.data[idx1])
        pruned.features[1].bias.data.copy_(teacher.features[1].bias.data[idx1])
        pruned.features[1].running_mean.copy_(teacher.features[1].running_mean[idx1])
        pruned.features[1].running_var.copy_(teacher.features[1].running_var[idx1])

        w2 = teacher.features[4].weight.data[:, idx1, :, :]
        idx2 = torch.topk(w2.view(w2.size(0), -1).norm(p=2, dim=1), c2).indices
        pruned.features[4].weight.data.copy_(w2[idx2])
        pruned.features[4].bias.data.copy_(teacher.features[4].bias.data[idx2])
        pruned.features[5].weight.data.copy_(teacher.features[5].weight.data[idx2])
        pruned.features[5].bias.data.copy_(teacher.features[5].bias.data[idx2])
        pruned.features[5].running_mean.copy_(teacher.features[5].running_mean[idx2])
        pruned.features[5].running_var.copy_(teacher.features[5].running_var[idx2])

        w3 = teacher.features[8].weight.data[:, idx2, :, :]
        idx3 = torch.topk(w3.view(w3.size(0), -1).norm(p=2, dim=1), c3).indices
        pruned.features[8].weight.data.copy_(w3[idx3])
        pruned.features[8].bias.data.copy_(teacher.features[8].bias.data[idx3])
        pruned.features[9].weight.data.copy_(teacher.features[9].weight.data[idx3])
        pruned.features[9].bias.data.copy_(teacher.features[9].bias.data[idx3])
        pruned.features[9].running_mean.copy_(teacher.features[9].running_mean[idx3])
        pruned.features[9].running_var.copy_(teacher.features[9].running_var[idx3])

        fc1_w = teacher.classifier[1].weight.data.view(384, 256, 4)[:, idx3, :].reshape(384, c3 * 4)
        pruned.classifier[1].weight.data.copy_(fc1_w)
        pruned.classifier[1].bias.data.copy_(teacher.classifier[1].bias.data)
        pruned.classifier[3].weight.data.copy_(teacher.classifier[3].weight.data)
        pruned.classifier[3].bias.data.copy_(teacher.classifier[3].bias.data)

    return pruned

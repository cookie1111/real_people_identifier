input_dim = 768

from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetRobertaEnsamble(nn.Module):
    def __init__(self, num_classes, input_dim, resnet, roberta):
        super(ResNetRobertaEnsamble, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.classifier = nn.Linear(256, 1)
        self.modelA = resnet
        self.modelB = roberta

    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        return x


def init_resnet(output_size=128):
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    for i in resnet.parameters():
        i.requires_grad = False
    resnet.fc = nn.Linear(512, output_size)
    return resnet


def init_roberta(output_size=128):
    classifier_head = RobertaClassificationHead(num_classes=output_size, input_dim=input_dim)
    roberta = XLMR_BASE_ENCODER.get_model(head=classifier_head)
    roberta.encoder.requires_grad_(False)
    return roberta


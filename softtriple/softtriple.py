import torch
import torch.nn as nn
from . import bninception
from . import loss


class SoftTripleNet(nn.Module):
    def __init__(self, embedding_size=64, n_class=99, pretrained=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(SoftTripleNet, self).__init__()
        self.bninception = bninception.bninception(pretrained=pretrained)
        self.loss_fn = loss.SoftTripleLoss(embedding_size, n_class).to(device)

    def forward(self, x, labels, use_loss=True):
        embedding = self.bninception(x)
        if use_loss:
            loss = self.loss_fn(embedding, labels)
            return loss, embedding
        return embedding

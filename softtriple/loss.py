import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class SoftTripleLoss(nn.Module):
    def __init__(self, embedding_size, n_class, n_center=3,
                 lmd=10.0, gamma=0.1, tau=0.2, margin=0.01):
        super(SoftTripleLoss, self).__init__()
        self._lmd = lmd
        self._inv_gamma = 1.0 / gamma
        self._tau = tau
        self._margin = margin
        self._n_class = n_class
        self._n_center = n_center
        self.fc = Parameter(torch.Tensor(n_class * n_center, embedding_size))
        nn.init.kaiming_uniform_(self.fc)
        self.softmax = nn.Softmax(dim=2)

    def infer(self, embedding):
        weight = F.normalize(self.fc, p=2, dim=1)
        x = F.linear(embedding, weight).view(-1, self._n_class, self._n_center)
        prob = self.softmax(self._inv_gamma * x)
        return (prob * x).sum(dim=2), weight

    def forward(self, embeddings, labels,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        h, w = self.infer(embeddings)
        margin_m = torch.zeros(h.shape).to(device)
        margin_m[torch.arange(0, margin_m.shape[0]), labels] = self._margin
        loss_cls = F.cross_entropy(self._lmd * (h - margin_m), labels.squeeze(-1))
        if self._tau > 0 and self._n_center > 1:
            reg = 0.0
            for i in range(self._n_class):
                w_sub = w[i * self._n_center : (i + 1) * self._n_center]
                sub_norm = 1.0 - torch.matmul(w_sub, w_sub.transpose(1, 0))
                sub_norm[sub_norm <= 0.0] = 1e-10
                reg += torch.sqrt(2 * sub_norm.triu(diagonal=1)).sum()
            reg /= self._n_class * self._n_center * (self._n_center - 1.0)
            return loss_cls + self._tau * reg
        else:
            return loss_cls

import torch

import torch.nn.functional as F

from typing import List


def inter_space_loss(concept_spaces: List[torch.Tensor], eps=1e-9):
    loss = 0.0
    for k in range(len(concept_spaces)):
        for l in range(k + 1, len(concept_spaces)):
            loss += (1.0 / (1.0 - F.tanh(concept_spaces[k].mean(dim=0).mean(dim=0) @ concept_spaces[l].mean(dim=0).mean(dim=0)) + eps))
    return loss.mean(0)


def intra_space_loss(concept_spaces: List[torch.Tensor], eps=1e-9):
    loss = 0.0
    for k in range(len(concept_spaces)):
        loss += (1.0 / (concept_spaces[k].var(1) + eps)).sum(1)
    return loss.mean(0)

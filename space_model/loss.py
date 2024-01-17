import torch

import torch.nn.functional as F

from typing import List


def inter_space_loss(concept_spaces, labels: torch.Tensor, m1: float = 0.5, m2: float = 0.5):
    """
    :param concept_spaces: pytorch tensors of shape: (n_concept_spaces, B, seq_len, n_latent)
    :param labels: labels of shape (B)
    :param m1: weight of match loss
    :param m2: weight of miss match loss
    :return: loss (scalar)
    """

    loss = 0.0

    for k in range(len(concept_spaces)):
        # match loss
        loss += m1 * torch.nan_to_num(
            (1 - concept_spaces[k][labels == k]).mean())  # (B', n_latent, seq_len) * (B', seq_len, n_embed)

        # mismatch loss
        loss += m2 * torch.nan_to_num((1 + concept_spaces[k][labels != k]).mean())
    return loss


def intra_space_loss(concept_spaces: List[torch.Tensor], eps=1e-9):
    loss = 0.0
    for k in range(len(concept_spaces)):
        loss += (1.0 / (concept_spaces[k].var(1) + eps)).sum(1)  # (n, B, Max_seq, n_latent)
    return loss.mean(0)

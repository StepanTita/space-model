import torch

import torch.nn.functional as F

from typing import List


def inter_space_loss(concept_spaces: List[torch.Tensor], labels: torch.Tensor, m1: float = 0.5, m2: float = 0.5):
    """
    :param concept_spaces: pytorch tensors of shape: (n_concept_spaces, B, seq_len, n_latent)
    :param labels: labels of shape (B) or of shape (B, labels_dim) (for multi-label classification)
    :param m1: weight of match loss
    :param m2: weight of miss match loss
    :return: loss (scalar)
    """

    concept_spaces = torch.stack(concept_spaces, dim=0)  # (n_concept_spaces, B, seq_len, n_latent)

    if len(labels.shape) == 1:
        # match loss
        match_loss = (1 - concept_spaces[labels == torch.arange(len(concept_spaces))]).mean()
        # mismatch loss
        mismatch_loss = (1 + concept_spaces[labels != torch.arange(len(concept_spaces))]).mean()
    else:
        if len(labels.shape) > 2:
            raise ValueError("labels must be a tensor or an integer: of shape (B) or of shape (B, labels_dim) (for multi-label classification)")

        match_loss = (1 - concept_spaces[labels.T > 0]).mean()
        mismatch_loss = (1 + concept_spaces[labels.T == 0]).mean()

    loss = m1 * match_loss + m2 * mismatch_loss

    return loss.mean()


def intra_space_loss(concept_spaces: List[torch.Tensor], eps=1e-9):
    """
    A function to calculate the intra-space loss, ensuring that the concept spaces are not colliding into a single vector.
    :param concept_spaces: pytorch tensors of shape: (n_concept_spaces, B, seq_len, n_latent)
    :param eps: small value to avoid division by zero
    :return: loss (scalar)
    """
    loss = 0.0
    for k in range(len(concept_spaces)):
        loss += (1.0 / (concept_spaces[k].var(1) + eps)).sum(1)  # (n, B, Max_seq, n_latent)
    return loss.mean(0)

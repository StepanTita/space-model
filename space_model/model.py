import torch
import torch.nn.functional as F

import space_model.loss as losses


class SpaceModelOutput:
    def __init__(self, logits=None, concept_spaces=None):
        self.logits = logits
        self.concept_spaces = concept_spaces

    def __repr__(self):
        return f'{self.logits=},{self.concept_spaces=}'


class SpaceModel(torch.nn.Module):
    def __init__(self, n_embed=3, n_latent=3, n_concept_spaces=2, output_concept_spaces=True):
        super().__init__()

        self.output_concept_spaces = output_concept_spaces

        # project embedding space (B, max_seq_len, n_embed) -> to the concept space (B, n_embed, n_latent) = (B, max_seq_len, n_latent)
        self.concept_spaces = torch.nn.ModuleList([
            torch.nn.Sequential(
                # unlike in the original paper, here we use a combination of Dense layer + tanh instead of cosine similarity
                torch.nn.Linear(in_features=n_embed, out_features=n_latent, bias=False),
                torch.nn.Tanh(),
            ) for _ in range(n_concept_spaces)
        ])

    def forward(self, x):
        projected_x = [space(x) for space in self.concept_spaces]  # (n_concept_spaces, B, max_seq_len, n_latent)

        avg_concept_attention = [x.mean(1) for x in projected_x]  # (n_concept_spaces, B, n_latent)

        concept_logits = torch.cat(avg_concept_attention, dim=-1)  # (B, n_concept_spaces * n_latent)

        concept_spaces = None
        if self.output_concept_spaces:
            concept_spaces = projected_x

        return SpaceModelOutput(concept_logits, concept_spaces)


class SpaceModelForClassification(torch.nn.Module):
    def __init__(self, n_embed=3, n_latent=3, n_concept_spaces=2):
        super().__init__()

        self.space_model = SpaceModel(n_embed, n_latent, n_concept_spaces, output_concept_spaces=True)

        # number of target classes is equal to the number of concepts
        self.concept_classifier = torch.nn.Linear(in_features=n_concept_spaces * n_latent,
                                                  out_features=n_concept_spaces)

    def forward(self, x):
        outputs = self.space_model(x)  # (B, n_concept_spaces * n_latent)

        logits = self.concept_classifier(outputs.logits)  # (B, n_concept_spaces)

        return SpaceModelOutput(logits, outputs.concept_spaces)


class SpaceModelForSequenceClassificationOutput:
    def __init__(self, loss=None, logits=None, concept_spaces=None):
        self.loss = loss
        self.logits = logits
        self.concept_spaces = concept_spaces


class SpaceModelForSequenceClassification(torch.nn.Module):
    def __init__(self, base_model, n_embed=3, n_latent=3, n_concept_spaces=2, l1=1e-3, l2=1e-4, fine_tune=True):
        super().__init__()

        if fine_tune:
            for p in base_model.parameters():
                p.requires_grad_(False)

        self.base_model = base_model

        self.space_model = SpaceModel(n_embed, n_latent, n_concept_spaces, output_concept_spaces=True)

        self.classifier = torch.nn.Linear(n_concept_spaces * n_latent, n_concept_spaces)

        self.l1 = l1
        self.l2 = l2

    def get_concept_spaces(self, input_ids, attention_mask):
        embed = self.base_model(input_ids, attention_mask).last_hidden_state  # (B, max_seq_len, 768)

        out = self.space_model(embed)

        return out.concept_spaces

    def forward(self, input_ids, attention_mask, labels=None):
        embed = self.base_model(input_ids, attention_mask).last_hidden_state  # (B, max_seq_len, 768)

        out = self.space_model(embed)

        concept_hidden = out.logits

        logits = self.classifier(concept_hidden)

        loss = 0.0
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            loss += self.l1 * losses.inter_space_loss(out.concept_spaces) + self.l2 * losses.intra_space_loss(
                out.concept_spaces)

        return SpaceModelForSequenceClassificationOutput(loss, logits, out.concept_spaces)

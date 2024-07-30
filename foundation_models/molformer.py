from torch import nn
from transformers import AutoModel, AutoTokenizer


def get_molformer_tokenizer():
    return AutoTokenizer.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True, use_fast=False
    )


class MolFormerRegressor(nn.Module):
    def __init__(self, tokenizer, n_last_hidden_units=100):
        super().__init__()
        self.tokenizer = tokenizer
        self.feature_extractor = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            deterministic_eval=True,
            trust_remote_code=True,
        )
        self.feature_dim = self.feature_extractor.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, n_last_hidden_units),
            nn.ReLU(),
            nn.Linear(n_last_hidden_units, 1),
        )

    def forward(self, data):
        feat = self.forward_features(data)
        return self.head(feat)

    def forward_features(self, data):
        input_ids, attn_mask = data["input_ids"], data["attention_mask"]
        device = next(self.parameters()).device
        input_ids = input_ids.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        feat = self.feature_extractor(input_ids, attn_mask).pooler_output
        return feat

    def freeze_params(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def unfreeze_params(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = True

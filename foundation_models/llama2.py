import torch
from torch import nn
from transformers import LlamaModel, LlamaConfig, LlamaTokenizer
from utils.configs import LLMFeatureType


def get_llama2_tokenizer(kind):
    kind = f"meta-llama/{kind.capitalize()}-hf"
    return LlamaTokenizer.from_pretrained(kind)


class Llama2Regressor(nn.Module):
    def __init__(
        self,
        kind,
        tokenizer,
        reduction=LLMFeatureType.LAST_TOKEN,
        n_last_hidden_units=100,
    ):
        assert kind in ["llama-2-7b"]
        super().__init__()
        self.tokenizer = tokenizer
        self.reduction = reduction
        kind = f"meta-llama/{kind.capitalize()}-hf"
        self.config = LlamaConfig.from_pretrained(kind)
        self.config.attn_dropout = 0
        self.feature_extractor = LlamaModel.from_pretrained(kind, config=self.config)
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
        input_ids = data["input_ids"]
        device = next(self.parameters()).device
        input_ids = input_ids.to(device, non_blocking=True)
        feat = self.feature_extractor(input_ids)[0]

        if self.reduction == LLMFeatureType.FIRST_TOKEN:
            feat = feat[:, 0, :]
        elif self.reduction == LLMFeatureType.LAST_TOKEN:
            # Find the last token position (before padding)
            sequence_lengths = (
                torch.eq(input_ids, self.tokenizer.pad_token_id).long().argmax(-1) - 1
            ).to(device)
            feat = feat[
                torch.arange(feat.shape[0], device=device), sequence_lengths
            ]  # (batch_size, feature_dim)
        elif self.reduction == LLMFeatureType.AVERAGE:
            # Masking---0 for everything after <eos> and 1 otherwise
            mask = ~input_ids.eq(self.tokenizer.pad_token_id).to(feat.device)
            mask = mask[:, :, None].float()  # (batch_size, seq_len, 1)
            #  (batch_size, 1, hidden_size) / (batch_size, 1, 1) -> (batch_size, 1, hidden_size)
            feat = (feat * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)

        return feat.squeeze(1)

    def freeze_params(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def unfreeze_params(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = True

import torch
from torch import nn
from transformers import T5EncoderModel, T5Config, T5Tokenizer
from utils.configs import LLMFeatureType


def get_t5_tokenizer(kind):
    return T5Tokenizer.from_pretrained(kind, model_max_length=512)


class T5Regressor(nn.Module):
    def __init__(
        self,
        kind,
        tokenizer,
        reduction=LLMFeatureType.LAST_TOKEN,
        n_last_hidden_units=50,
    ):
        assert kind in [
            "t5-small",
            "t5-base",
            "t5-large",
            "GT4SD/multitask-text-and-chemistry-t5-base-augm",
        ]
        super().__init__()
        self.tokenizer = tokenizer
        self.reduction = reduction
        self.config = T5Config.from_pretrained(kind)
        self.config.dropout_rate = 0
        self.feature_extractor = T5EncoderModel.from_pretrained(
            kind, config=self.config
        )
        self.feature_dim = self.feature_extractor.config.d_model
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, n_last_hidden_units),
            nn.ReLU(),
            nn.Linear(n_last_hidden_units, n_last_hidden_units),
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
        # (batch_size, seq_len, hidden_size)
        feat = self.feature_extractor(input_ids).last_hidden_state

        if self.reduction == LLMFeatureType.FIRST_TOKEN:
            feat = feat[:, 0, :]
        elif self.reduction == LLMFeatureType.LAST_TOKEN:
            # Find the last token position (before padding)
            eos_mask = input_ids.eq(self.tokenizer.eos_token_id).to(feat.device)
            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError(
                    "All examples must have the same number of <eos> tokens."
                )
            batch_size, _, hidden_size = feat.shape
            feat = feat[eos_mask, :]  # (batch_size, hidden_size)
            feat = feat.view(batch_size, -1, hidden_size)[:, -1, :]
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

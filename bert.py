import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
from utils import length_to_mask
from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
from transformers.modeling_outputs import BaseModelOutput
from transformers import BertModel, BertConfig, BertTokenizer


@dataclass
class MultiTaskModelOutput(BaseModelOutput):
    tokens_pred: Optional[torch.FloatTensor] = None
    words_pred: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    loss_vocab: Optional[torch.FloatTensor] = None
    loss_token: Optional[torch.FloatTensor] = None


class MultiTaskModel(nn.Module):
    def __init__(self, model, num_tokens=178, num_vocab=84827, hidden_size=768):
        super().__init__()

        self.encoder = model
        self.mask_predictor = nn.Linear(hidden_size, num_tokens)
        self.word_predictor = nn.Linear(hidden_size, num_vocab)
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        phonemes,
        labels=None,
        words=None,
        input_lengths=None,
        masked_indices=None,
        attention_mask=None,
    ):
        if input_lengths is not None:
            text_mask = length_to_mask(torch.Tensor(input_lengths)).to(phonemes.device)
            attention_mask = (~text_mask).int()

        output = self.encoder(
            phonemes, attention_mask=attention_mask, output_hidden_states=True
        )
        tokens_pred = self.mask_predictor(output.last_hidden_state)
        words_pred = self.word_predictor(output.last_hidden_state)

        if words is not None and labels is not None and input_lengths is not None:
            loss_vocab = 0
            for _s2s_pred, _text_input, _text_length, _masked_indices in zip(
                words_pred, words, input_lengths, masked_indices
            ):
                loss_vocab += self.criterion(
                    _s2s_pred[:_text_length], _text_input[:_text_length]
                )
            loss_vocab /= words.size(0)

            loss_token = 0
            sizes = 1
            for _s2s_pred, _text_input, _text_length, _masked_indices in zip(
                tokens_pred, labels, input_lengths, masked_indices
            ):
                if len(_masked_indices) > 0:
                    _text_input = _text_input[:_text_length][_masked_indices]
                    loss_tmp = self.criterion(
                        _s2s_pred[:_text_length][_masked_indices],
                        _text_input[:_text_length],
                    )
                    loss_token += loss_tmp
                    sizes += 1
            loss_token /= sizes

            loss = loss_vocab + loss_token

            return MultiTaskModelOutput(
                last_hidden_state=output.last_hidden_state,
                hidden_states=output.hidden_states,
                attentions=output.attentions,
                tokens_pred=tokens_pred,
                words_pred=words_pred,
                loss=loss,
                loss_vocab=loss_vocab,
                loss_token=loss_token,
            )

        return MultiTaskModelOutput(
            last_hidden_state=output.last_hidden_state,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            tokens_pred=tokens_pred,
            words_pred=words_pred,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        bert_config = BertConfig.from_pretrained(pretrained_model_name_or_path)
        bert_model = BertModel(bert_config)
        model = cls(
            bert_model,
            num_tokens=bert_config.vocab_size,
            num_vocab=len(tokenizer),
            hidden_size=bert_config.hidden_size,
        )

        try:
            state_dict_file = try_to_load_from_cache(
                pretrained_model_name_or_path, "pytorch_model.bin"
            )
        except FileNotFoundError:
            state_dict_file = _CACHED_NO_EXIST

        if state_dict_file is None:
            raise ValueError(
                f"Model {pretrained_model_name_or_path} not found in cache or on Hugging Face Hub."
            )

        model.load_state_dict(
            torch.load(state_dict_file, map_location="cpu"), strict=False
        )

        return model

import torch
from torch import nn
from typing import Optional, List
from dataclasses import dataclass
from utils import length_to_mask
from transformers import BertModel, BertTokenizer, BertConfig
from transformers.modeling_outputs import BaseModelOutput
from text.text_utils import TextCleaner
import os


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

        output = self.encoder(phonemes, attention_mask=attention_mask)
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model_path = os.path.join(os.path.dirname(__file__), "../pl-bert")
bert_config = BertConfig.from_pretrained(bert_model_path)
bert_model = BertModel(bert_config)
tokenizer = BertTokenizer.from_pretrained(bert_model_path)
model = (
    MultiTaskModel(
        bert_model,
        num_vocab=len(tokenizer.get_vocab()),
        num_tokens=477,
        hidden_size=768,
    )
    .to(device)
    .eval()
)
model.load_state_dict(
    torch.load(os.path.join(bert_model_path, "pytorch_model.bin"), map_location=device)
)
text_cleaner = TextCleaner()


@torch.no_grad()
def get_bert(phonemes: List[str], device="cuda"):
    phonemes, _ = text_cleaner(phonemes)
    input_ids = torch.LongTensor([phonemes]).to(device)
    output = model(input_ids)

    return output.last_hidden_state.squeeze(0).cpu().numpy()


if __name__ == "__main__":
    phonemes = "ngo5 hai6 hoeng1 gong2 jan4!"
    bert_output = get_bert(phonemes, device)
    print(bert_output.shape)

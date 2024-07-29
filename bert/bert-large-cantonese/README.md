---
library_name: transformers
language:
  - yue
license: cc-by-4.0
tags:
  - generated_from_trainer
pipeline_tag: fill-mask
widget:
  - text: 香港原本[MASK]一個人煙稀少嘅漁港。
    example_title: 係
model-index:
  - name: bert-large-cantonese
    results: []
---

# bert-large-cantonese

## Description

This model is tranied from scratch on Cantonese text. It is a BERT model with a large architecture (24-layer, 1024-hidden, 16-heads, 326M parameters).

The first training stage is to pre-train the model on 128 length sequences with a batch size of 512 for 1 epoch. the second stage is to continued pre-train the model on 512 length sequences with a batch size of 512 for one more epoch.

## How to use

You can use this model directly with a pipeline for masked language modeling:

```python
from transformers import pipeline

mask_filler = pipeline(
    "fill-mask",
    model="hon9kon9ize/bert-large-cantonese"
)

mask_filler("雞蛋六隻，糖呢就兩茶匙，仲有[MASK]橙皮添。")

; [{'score': 0.08160534501075745,
;   'token': 943,
;   'token_str': '個',
;   'sequence': '雞 蛋 六 隻 ， 糖 呢 就 兩 茶 匙 ， 仲 有 個 橙 皮 添 。'},
;  {'score': 0.06182105466723442,
;   'token': 1576,
;   'token_str': '啲',
;   'sequence': '雞 蛋 六 隻 ， 糖 呢 就 兩 茶 匙 ， 仲 有 啲 橙 皮 添 。'},
;  {'score': 0.04600336775183678,
;   'token': 1646,
;   'token_str': '嘅',
;   'sequence': '雞 蛋 六 隻 ， 糖 呢 就 兩 茶 匙 ， 仲 有 嘅 橙 皮 添 。'},
;  {'score': 0.03743772581219673,
;   'token': 3581,
;   'token_str': '橙',
;   'sequence': '雞 蛋 六 隻 ， 糖 呢 就 兩 茶 匙 ， 仲 有 橙 橙 皮 添 。'},
;  {'score': 0.031560592353343964,
;   'token': 5148,
;   'token_str': '紅',
;   'sequence': '雞 蛋 六 隻 ， 糖 呢 就 兩 茶 匙 ， 仲 有 紅 橙 皮 添 。'}]
```

## Training hyperparameters

The following hyperparameters were used during first training:

- Batch size: 512
- Learning rate: 1e-4
- Learning rate scheduler: linear decay
- 1 Epoch
- Warmup ratio: 0.1

Loss plot on [WanDB](https://api.wandb.ai/links/indiejoseph/v3ljlpmp)

The following hyperparameters were used during second training:

- Batch size: 512
- Learning rate: 5e-5
- Learning rate scheduler: linear decay
- 1 Epoch
- Warmup ratio: 0.1

Loss plot on [WanDB](https://api.wandb.ai/links/indiejoseph/vcm3q1ef)


import re
from text.text_normalize import normalize_text


def is_jyutping(text):
    return re.match(r"^[a-z]+[1-6]{1}", text) is not None


def phonemize(text, phonemizer, tokenizer):
    input_ids = []
    phonemes = []
    tmp_phoneme = None
    text = normalize_text(text)
    phoneme_text = phonemizer(text)
    tokenized_text = tokenizer.encode(phoneme_text, add_special_tokens=False)

    for i, token in enumerate(tokenized_text):
        next_token = tokenized_text[i + 1] if i + 1 < len(tokenized_text) else None

        if token == 113:
            tmp_phoneme = ""
            continue
        elif token == 114:
            if is_jyutping(tmp_phoneme):
                phonemes.append(tmp_phoneme)
            tmp_phoneme = None
            continue
        elif tmp_phoneme != None:
            tmp_phoneme += tokenizer.decode(token, add_special_tokens=False).replace(
                "#", ""
            )
            continue

        input_ids.append(token)

        if next_token != 113:
            token_text = tokenizer.decode(token, add_special_tokens=False)

            if is_jyutping(token_text) or token_text in ".,!?":
                phonemes.append(tokenizer.decode(token, add_special_tokens=False))
            else:
                phonemes.append("[UNK]")

    assert len(input_ids) == len(
        phonemes
    ), f"Length mismatch: {len(input_ids)} != {len(phonemes)}"

    assert len(input_ids) == len(phonemes)

    return {"input_ids": input_ids, "phonemes": phonemes}

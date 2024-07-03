import torch
import safetensors.torch
from style_bert_vits2.nlp.symbols import (
    SYMBOLS,
    NUM_TONES,
    NUM_LANGUAGES,
)
import argparse
import math

old_symbols_punctuation = ["!", "?", "…", ",", ".", "'", "-"]
old_symbols_pu_symbols = old_symbols_punctuation + ["SP", "UNK"]
old_symbols_pad = "_"

old_symbols_zh_symbols = [
    "E",
    "En",
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "f",
    "g",
    "h",
    "i",
    "i0",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "in",
    "ing",
    "iong",
    "ir",
    "iu",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ong",
    "ou",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "ui",
    "un",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
    "w",
    "x",
    "y",
    "z",
    "zh",
    "AA",
    "EE",
    "OO",
]
old_symbols_num_zh_tones = 6

old_symbols_ja_symbols = [
    "N",
    "a",
    "a:",
    "b",
    "by",
    "ch",
    "d",
    "dy",
    "e",
    "e:",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "i:",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "o:",
    "p",
    "py",
    "q",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "u:",
    "w",
    "y",
    "z",
    "zy",
]
old_symbols_num_ja_tones = 2

old_symbols_en_symbols = [
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "uh",
    "uw",
    "V",
    "w",
    "y",
    "z",
    "zh",
]
old_symbols_num_en_tones = 4

old_symbols_normal_symbols = sorted(
    set(old_symbols_zh_symbols + old_symbols_ja_symbols + old_symbols_en_symbols)
)
old_symbols_symbols = (
    [old_symbols_pad] + old_symbols_normal_symbols + old_symbols_pu_symbols
)

old_symbols_num_tones = (
    old_symbols_num_zh_tones + old_symbols_num_ja_tones + old_symbols_num_en_tones
)

old_symbols_language_id_map = {"ZH": 0, "JP": 1, "EN": 2}
old_symbols_num_languages = len(old_symbols_language_id_map.keys())


def resize_embedding_layer(weight, new_symbols, old_symbols):
    old_vocab_size = weight.size(0)
    embedding_dim = weight.size(1)
    new_vocab_size = len(new_symbols)

    if new_vocab_size < old_vocab_size:
        return weight[:new_vocab_size, :]
    elif new_vocab_size == old_vocab_size:
        return weight
    else:
        new_weight = weight.new_zeros(new_vocab_size, embedding_dim)
        new_weight[:old_vocab_size, :] = weight
        old_symbol_to_idx = {symbol: idx for idx, symbol in enumerate(old_symbols)}
        for new_idx, symbol in enumerate(new_symbols):
            if symbol in old_symbol_to_idx:
                old_idx = old_symbol_to_idx[symbol]
                new_weight[new_idx, :] = weight[old_idx, :]
                print("Reused vector for symbol: ", symbol)
            else:
                avg_weight = weight.mean(dim=0, keepdim=True)
                noise_weight = torch.empty((1, embedding_dim)).normal_(
                    mean=0, std=(1.0 / math.sqrt(embedding_dim))
                )
                new_weight[new_idx, :] = avg_weight + noise_weight
        return new_weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight",
        type=str,
        help="path to original weight file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="path to output weight file",
    )
    args = parser.parse_args()

    checkpoint_dict = safetensors.torch.load_file(args.weight)
    checkpoint_dict["enc_p.emb.weight"] = resize_embedding_layer(
        checkpoint_dict["enc_p.emb.weight"], SYMBOLS, old_symbols_symbols
    )
    checkpoint_dict["enc_p.tone_emb.weight"] = resize_embedding_layer(
        checkpoint_dict["enc_p.tone_emb.weight"],
        list(range(NUM_TONES)),
        list(range(old_symbols_num_tones)),
    )
    checkpoint_dict["enc_p.language_emb.weight"] = resize_embedding_layer(
        checkpoint_dict["enc_p.language_emb.weight"],
        list(range(NUM_LANGUAGES)),
        list(range(old_symbols_num_languages)),
    )

    safetensors.torch.save_file(checkpoint_dict, args.output)

    print("Resize weight file done!")

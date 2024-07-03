import re
import unicodedata
import cn2an
import pycantonese
import jieba

from os.path import join, dirname
from style_bert_vits2.nlp.symbols import PUNCTUATIONS


jieba.load_userdict(join(dirname(__file__), "yue_dict.txt"))

jyutping_dict = {}

with open(join(dirname(__file__), "jyutping.csv"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        word, jyutping = line.split(",")

        if word not in jyutping_dict:
            jyutping_dict[word] = [jyutping]
        else:
            jyutping_dict[word].append(jyutping)

# print('Loaded')

def normalizer(x: str) -> str:
    x = cn2an.transform(x, "an2cn")
    return x


def word2jyutping(word: str) -> str:
    jyutpings = [
        pycantonese.characters_to_jyutping(w)[0][1]
        for w in word
        if unicodedata.name(w, "").startswith("CJK UNIFIED IDEOGRAPH")
    ]

    for i, j in enumerate(jyutpings):
        if re.search(r"^(la|ga)[1-6]$", j):
            # la1 -> laa1, ga1 -> gaa1
            jyutpings[i] = jyutpings[i].replace("a", "aa")

    if any(j is None for j in jyutpings):
        raise ValueError(f"Failed to convert {word} to jyutping: {jyutpings}")

    return " ".join(jyutpings)


INITIALS = [
    "",
    "b",
    "c",
    "d",
    "f",
    "g",
    "gw",
    "h",
    "j",
    "k",
    "kw",
    "l",
    "m",
    "n",
    "ng",
    "p",
    "s",
    "t",
    "w",
    "z",
]
FINALS = [
    "aa",
    "aai",
    "aau",
    "aam",
    "aan",
    "aang",
    "aap",
    "aat",
    "aak",
    "ai",
    "au",
    "am",
    "an",
    "ang",
    "ap",
    "at",
    "ak",
    "e",
    "ei",
    "eu",
    "em",
    "eng",
    "ep",
    "ek",
    "i",
    "iu",
    "im",
    "in",
    "ing",
    "ip",
    "it",
    "ik",
    "o",
    "oi",
    "ou",
    "on",
    "ong",
    "ot",
    "ok",
    "oe",
    "oeng",
    "oek",
    "eoi",
    "eon",
    "eot",
    "u",
    "ui",
    "un",
    "ung",
    "ut",
    "uk",
    "yu",
    "yun",
    "yut",
    "m",
    "ng",
]


def word_segmentation(text: str) -> list[str]:
    words = jieba.cut(text)
    return list(words)


def jyutping_to_initials_finals_tones(
    jyutping_syllables: list[str],
) -> tuple[list[str], list[int], list[int]]:
    initials_finals = []
    tones = []
    word2ph = []

    for syllable in jyutping_syllables:
        if syllable in PUNCTUATIONS:
            initials_finals.append(syllable)
            tones.append(0)
            word2ph.append(1)  # Add 1 for punctuation
        else:
            init, final, tone = parse_jyutping(syllable)
            initials_finals.extend([init, final])
            tones.extend([tone, tone])
            word2ph.append(2)

    assert len(initials_finals) == len(tones)
    return initials_finals, tones, word2ph


def get_jyutping(text: str) -> list[str]:
    words = word_segmentation(text)
    jyutping_array = []

    for word in words:
        if word in PUNCTUATIONS:
            jyutping_array.append(word)
        else:
            jyutpings = ""

            if word in jyutping_dict:
                jyutpings = jyutping_dict[word][0]
            else:
                jyutpings = word2jyutping(word)

            if "la1" in jyutpings:
                print(text, words, jyutpings)

            # match multple jyutping eg: liu4 ge3, or single jyutping eg: liu4
            if not re.search(r"^([a-z]+[1-6]+[ ]?)+$", jyutpings):
                raise ValueError(f"Failed to convert {word} to jyutping: {jyutpings}")

            jyutping_array.extend(jyutpings.split(" "))

    return jyutping_array


def parse_jyutping(jyutping: str) -> tuple[str, str, int]:
    orig_jyutping = jyutping

    if len(jyutping) < 2:
        raise ValueError(f"Jyutping string too short: {jyutping}")
    init = ""
    if jyutping[0] == "n" and jyutping[1] == "g" and len(jyutping) == 3:
        init = ""
    elif jyutping[0] == "m" and len(jyutping) == 2:
        init = ""
    elif jyutping[0] == "n" and jyutping[1] == "g":
        init = "ng"
        jyutping = jyutping[2:]
    elif jyutping[0] == "g" and jyutping[1] == "w":
        init = "gw"
        jyutping = jyutping[2:]
    elif jyutping[0] == "k" and jyutping[1] == "w":
        init = "kw"
        jyutping = jyutping[2:]
    elif jyutping[0] in "bpmfdtnlgkhwzcsj":
        init = jyutping[0]
        jyutping = jyutping[1:]
    else:
        jyutping = jyutping
    try:
        tone = int(jyutping[-1])
        jyutping = jyutping[:-1]
    except:
        raise ValueError("Jyutping string does not end with a tone number")
    final = jyutping

    assert init in INITIALS, f"Invalid initial: {init}, in {orig_jyutping}"

    if final not in FINALS:
        raise ValueError(f"Invalid final: {final}, in {orig_jyutping}")

    return init, final, tone


def g2p(text: str) -> tuple[list[str], list[int], list[int]]:
    word2ph = []
    jyuping = get_jyutping(text)
    phones, tones, word2ph = jyutping_to_initials_finals_tones(jyuping)
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    # print(phones)
    return phones, tones, word2ph


if __name__ == "__main__":
    from style_bert_vits2.nlp.cantonese.bert_feature import extract_bert_feature
    from style_bert_vits2.nlp.cantonese.normalizer import normalize_text

    text = "啊！但是《原神》是由,米哈游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = normalize_text(text)
    print(text)
    phones, tones, word2ph = g2p(text)
    bert = extract_bert_feature(text, word2ph, "cuda")

    print(phones, tones, word2ph, bert.shape)


# 示例用法
# text = "这是一个示例文本：,你好！这是一个测试...."
# print(g2p_paddle(text))  # 输出: 这是一个示例文本你好这是一个测试

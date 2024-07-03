import re
import unicodedata
import cn2an

from style_bert_vits2.nlp.symbols import PUNCTUATIONS


__REPLACE_MAP = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}


replacement_chars = {
    "\n": " ",
    "ㄧ": "一",
    "—": "一",
    "更": "更",
    "不": "不",
    "料": "料",
    "聯": "聯",
    "行": "行",
    "利": "利",
    "謢": "護",
    "岀": "出",
    "鎭": "鎮",
    "戯": "戲",
    "旣": "既",
    "立": "立",
    "來": "來",
    "年": "年",
    "㗇": "蝦",
}


def replace_chars(text: str) -> str:
    for k, v in replacement_chars.items():
        text = text.replace(k, v)
    return text


def normalize_text(text: str) -> str:
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    text = replace_punctuation(text)
    text = replace_chars(text)
    return text


def replace_punctuation(text: str) -> str:

    # text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in __REPLACE_MAP))

    replaced_text = pattern.sub(lambda x: __REPLACE_MAP[x.group()], text)

    replaced_text = "".join(
        c
        for c in replaced_text
        if unicodedata.name(c, "").startswith("CJK UNIFIED IDEOGRAPH")
        or c in PUNCTUATIONS
    )

    return replaced_text

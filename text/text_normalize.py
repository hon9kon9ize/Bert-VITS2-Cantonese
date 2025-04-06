from tn.chinese.normalizer import Normalizer as ZhNormalizer

zh_tn_model = ZhNormalizer(
    remove_erhua=False,
    overwrite_cache=True,
    full_to_half=False,
    remove_interjections=False,
    traditional_to_simple=False,
)

rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": ".",
    "⋯": ".",
    "$": "",
    "“": "",
    "”": "",
    '"': "",
    "‘": "",
    "’": "",
    "（": "",
    "）": "",
    "(": "",
    ")": "",
    "《": "",
    "》": "",
    "【": "",
    "】": "",
    "[": "",
    "]": "",
    "—": "",
    "～": "",
    "~": "",
    "「": "",
    "」": "",
}

numeric_translate = str.maketrans("两万点", "兩萬點")


def normalize_punctuation(text):
    for k, v in rep_map.items():
        text = text.replace(k, v)

    return text


def normalize_numeric(text):
    return text.translate(numeric_translate)


def normalize_text(text):
    text = text.lower()
    text = normalize_punctuation(text)
    text = zh_tn_model.normalize(text)
    text = normalize_numeric(text)

    return text


if __name__ == "__main__":
    text = "$10, 簡直666，9同10"
    out = normalize_text(text)
    print(out)

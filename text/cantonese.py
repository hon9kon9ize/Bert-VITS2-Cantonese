from text.symbols import punctuation
import re
import unicodedata
import cn2an
import pycantonese
import ToJyutping
import csv


def normalizer(x):
    x = cn2an.transform(x, "an2cn")

    return x


def word2jyutping(word):
    jyutpings = [pycantonese.characters_to_jyutping(
        w)[0][1] for w in word if unicodedata.name(w, "").startswith("CJK UNIFIED IDEOGRAPH")]

    for i, j in enumerate(jyutpings):
        if re.search(r"^(la|ga)[1-6]$", j):
            # la1 -> laa1, ga1 -> gaa1
            jyutpings[i] = jyutpings[i].replace('a', 'aa')

    if None in jyutpings:
        raise ValueError(f"Failed to convert {word} to jyutping: {jyutpings}")

    return " ".join(jyutpings)


INITIALS = ["", "b", "c", "d", "f", "g", "gw", "h", "j",
            "k", "kw", "l", "m", "n", "ng", "p", "s", "t", "w", "z"]
FINALS = ["aa", "aai", "aau", "aam", "aan", "aang", "aap", "aat", "aak", "ai", "au", "am", "an", "ang", "ap", "at", "ak", "e", "ei", "eu", "em", "eng", "ep", "ek", "i", "iu", "im",
          "in", "ing", "ip", "it", "ik", "o", "oi", "ou", "on", "ong", "ot", "ok", "oe", "oeng", "oek", "eoi", "eon", "eot", "u", "ui", "un", "ung", "ut", "uk", "yu", "yun", "yut", "m", "ng"]

rep_map = {
    "：": ",",
    "︰": ",",
    "；": ",",
    "，": ",",
    "﹐": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "﹖": "?",
    "﹗": "!",
    "\n": ".",
    "·": ",",
    "、": ",",
    "丶": ",",
    "...": "…",
    "⋯": "…",
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
    "_": "-",
}

replacement_chars = {
    "\n": " ",
    'ㄧ': '一',
    '—': '一',
    '更': '更',
    '不': '不',
    '料': '料',
    '聯': '聯',
    '行': '行',
    '利': '利',
    '謢': '護',
    '岀': '出',
    '鎭': '鎮',
    '戯': '戲',
    '旣': '既',
    '立': '立',
    '來': '來',
    '年': '年',
    '㗇': '蝦',
}


def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    replaced_text = "".join(
        c for c in replaced_text if unicodedata.name(c, "").startswith("CJK UNIFIED IDEOGRAPH") or c in punctuation
    )

    return replaced_text


def replace_chars(text):
    for k, v in replacement_chars.items():
        text = text.replace(k, v)
    return text


def text_normalize(text):
    text = text.strip()
    text = normalizer(text)
    text = replace_punctuation(text)
    text = replace_chars(text)
    return text


def jyuping_to_initials_finals_tones(jyuping_syllables):
    initials_finals = []
    tones = []
    word2ph = []

    for syllable in jyuping_syllables:
        if syllable in punctuation:
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


wordshk_juytping = {}

# with open("/notebooks/bert-vits2/Bert-VITS2-Cantonese/wordshk_juytping.csv", "r") as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')

#     for row in csv_reader:
#         wordshk_juytping[text_normalize(row[0])] = row[1]


def get_jyutping(text):
    jyutping_array = []
    punct_pattern = re.compile(
        r"^[{}]+$".format(re.escape("".join(punctuation))))
    syllables = ToJyutping.get_jyutping_list(text)

    for word, syllable in syllables:
        if punct_pattern.match(word):
            puncts = re.split(r"([{}])".format(
                re.escape("".join(punctuation))), word)
            for punct in puncts:
                if len(punct) > 0:
                    jyutping_array.append(punct)
        else:
            # match multple jyutping eg: liu4 ge3, or single jyutping eg: liu4
            if not re.search(r"^([a-z]+[1-6]+[ ]?)+$", syllable):
                raise ValueError(
                    f"Failed to convert {word} to jyutping: {syllable}")

            jyutping_array.append(syllable)

    return jyutping_array


def get_bert_feature(text, word2ph):
    from text import cantonese_bert

    return cantonese_bert.get_bert_feature(text, word2ph)


def parse_jyutping(jyutping):
    orig_jyutping = jyutping

    if len(jyutping) < 2:
        raise ValueError(f"Jyutping string too short: {jyutping}")
    init = ""
    if jyutping[0] == 'n' and jyutping[1] == 'g' and len(jyutping) == 3:
        init = ""
    elif jyutping[0] == 'm' and len(jyutping) == 2:
        init = ""
    elif jyutping[0] == 'n' and jyutping[1] == 'g':
        init = 'ng'
        jyutping = jyutping[2:]
    elif jyutping[0] == 'g' and jyutping[1] == 'w':
        init = 'gw'
        jyutping = jyutping[2:]
    elif jyutping[0] == 'k' and jyutping[1] == 'w':
        init = 'kw'
        jyutping = jyutping[2:]
    elif jyutping[0] in 'bpmfdtnlgkhwzcsj':
        init = jyutping[0]
        jyutping = jyutping[1:]
    else:
        jyutping = jyutping
    try:
        tone = int(jyutping[-1])
        jyutping = jyutping[:-1]
    except:
        raise ValueError(
            f"Jyutping string does not end with a tone number, in {orig_jyutping}")
    final = jyutping

    assert init in INITIALS, f"Invalid initial: {init}, in {orig_jyutping}"

    if final not in FINALS:
        raise ValueError(f"Invalid final: {final}, in {orig_jyutping}")

    return [init, final, tone]


def g2p(text):
    word2ph = []
    jyuping = get_jyutping(text)
    phones, tones, word2ph = jyuping_to_initials_finals_tones(jyuping)
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


if __name__ == "__main__":
    from text.cantonese_bert import get_bert_feature

    # text = "Apple BB 你點解會咁柒㗎？我真係唔該晒你呀！123"
    text = "佢邊係想辭工吖，跳下草裙舞想加人工之嘛。"
    # text = "我個 app 嘅介紹文想由你寫，因為我唔知從一般用家角度要細緻到乜程度"
    # text = "佢哋最叻咪就係去㗇人傷害人,得個殼咋!"
    text = text_normalize(text)
    print('normalized text', text)
    phones, tones, word2ph = g2p(text)
    print(phones, tones, word2ph)
    bert = get_bert_feature(text, word2ph)
    print(bert.shape)

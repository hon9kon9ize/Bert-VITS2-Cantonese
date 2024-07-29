punctuation = ["!", "?", "…", ",", ".", "'", "-"]
pu_symbols = punctuation + ["SP", "UNK"]
pad = "_"

# English
en_symbols = [
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
num_en_tones = 4

# Cantonese
yue_symbols = [
    "",
    "aa",
    "aai",
    "aak",
    "aam",
    "aan",
    "aang",
    "aap",
    "aat",
    "aau",
    "ai",
    "ak",
    "am",
    "an",
    "ang",
    "ap",
    "at",
    "au",
    "b",
    "c",
    "d",
    "e",
    "ei",
    "ek",
    "em",
    "eng",
    "eoi",
    "eon",
    "eot",
    "ep",
    "eu",
    "f",
    "g",
    "gw",
    "h",
    "i",
    "ik",
    "im",
    "in",
    "ing",
    "ip",
    "it",
    "iu",
    "j",
    "k",
    "kw",
    "l",
    "m",
    "m",
    "n",
    "ng",
    "ng",
    "o",
    "oe",
    "oek",
    "oeng",
    "oi",
    "ok",
    "on",
    "ong",
    "ot",
    "ou",
    "p",
    "s",
    "t",
    "u",
    "ui",
    "uk",
    "un",
    "ung",
    "ut",
    "w",
    "yu",
    "yun",
    "yut",
    "z"
]

num_yue_tones = 7

# combine all symbols
normal_symbols = sorted(
    set(en_symbols + yue_symbols))
symbols = [pad] + normal_symbols + pu_symbols
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]

# combine all tones
num_tones = num_en_tones + num_yue_tones

# language maps
language_id_map = {"EN": 0, "YUE": 1}
num_languages = len(language_id_map.keys())

language_tone_start_map = {
    "EN": 0,
    "YUE": num_en_tones,
}

if __name__ == "__main__":
    a = set(yue_symbols)
    b = set(en_symbols)
    print(sorted(a & b))

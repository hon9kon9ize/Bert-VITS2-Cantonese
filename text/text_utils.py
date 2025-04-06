import re


INITIALS = [
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
    "M",
    "ng",
    "Ng",
]

TONE = ["1", "2", "3", "4", "5", "6"]
PAD = "[PAD]"
UNK = "[UNK]"
MASK = "[MASK]"
SEP = "[SEP]"
_punctuation = ".,!? "
_syllables = [f"{i}{j}" for i in INITIALS + FINALS for j in TONE]

# Export all symbols:
symbols = [PAD, UNK, SEP, MASK] + list(_punctuation) + _syllables + list(TONE)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i


def split_jyutping(text):
    """Splits a string of Jyutping into a list of individual syllables and punctuation.

    Args:
      text: The input string containing Jyutping.

    Returns:
      A list of strings, where each element is a Jyutping syllable or punctuation mark.
    """

    # Pattern to match Jyutping syllables (letters followed by a single digit 1-6)
    jyutping_pattern = r"[a-z]+[1-6]"

    # Pattern to match punctuation
    punctuation_pattern = r"[.,!?]"

    # Combine patterns to match either Jyutping or punctuation
    combined_pattern = rf"({jyutping_pattern}|{punctuation_pattern})"

    # Find all matches in the text
    matches = re.findall(combined_pattern, text)

    return matches


def parse_jyutping(jyutping):
    orig_jyutping = jyutping

    if len(jyutping) < 2:
        raise ValueError(f"Jyutping string too short: {jyutping}")
    init = "#"
    if jyutping[0] == "n" and jyutping[1] == "g" and len(jyutping) == 3:
        init = "#"
        jyutping = jyutping.replace("n", "N")
    elif jyutping[0] == "m" and len(jyutping) == 2:
        init = "#"
        jyutping = jyutping.replace("m", "M")
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
        tone = jyutping[-1]
        jyutping = jyutping[:-1]
    except:
        raise ValueError(
            f"Jyutping string does not end with a tone number, in {orig_jyutping}"
        )

    if init != "#":
        init = init + tone

    final = jyutping + tone

    assert (
        init[:-1] in INITIALS or init == "#"
    ), f"Invalid initial: {init[:-1]}, in {orig_jyutping}"

    if final[:-1] not in FINALS:
        raise ValueError(f"Invalid final: {final[:-1]}, in {orig_jyutping}")

    return [init, final]


class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts

    def __call__(self, text):
        indexes = []
        word2ph = []
        chars = split_jyutping(text)

        for i, syllable in enumerate(chars):
            _syllables = re.findall(r"[a-z]+[1-9]{1}", syllable)

            if len(_syllables) == 0:
                _syllables = [syllable]

            _word2ph_ = 0

            for _syllable in _syllables:
                try:
                    if _syllable in _punctuation or _syllable in [PAD, UNK, MASK]:
                        indexes.append(self.word_index_dictionary[_syllable])
                        _word2ph_ += 1
                    else:
                        init, final = parse_jyutping(_syllable)

                        if init != "#":
                            indexes.append(self.word_index_dictionary[init])
                            _word2ph_ += 1

                        indexes.append(self.word_index_dictionary[final])
                        _word2ph_ += 1

                except (KeyError, ValueError):
                    indexes.append(self.word_index_dictionary["[UNK]"])  # unknown token
                    _word2ph_ += 1

            word2ph.append(_word2ph_)

            # Add [SEP] token between syllables
            if i < len(chars) - 1:
                indexes.append(self.word_index_dictionary["[SEP]"])
                word2ph.append(1)

        assert len(indexes) == sum(word2ph), f"{len(indexes)} != {sum(word2ph)}"

        return indexes, word2ph

    def encode(self, text):
        try:
            return self.word_index_dictionary[text]
        except KeyError:
            return self.word_index_dictionary["[UNK]"]

    def decode(self, indexes):
        return [symbols[i] for i in indexes]

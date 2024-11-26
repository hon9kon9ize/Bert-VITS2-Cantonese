import logging

import regex as re

from tools.classify_language import classify_language, split_alpha_nonalpha


def check_is_none(item) -> bool:
    """none -> True, not none -> False"""
    return (
        item is None
        or (isinstance(item, str) and str(item).isspace())
        or str(item) == ""
    )


def markup_language(text: str, target_languages: list = None) -> str:
    pattern = (
        r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`"
        r"\！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」"
        r"『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+"
    )
    sentences = re.split(pattern, text)

    pre_lang = ""
    p = 0

    if target_languages is not None:
        sorted_target_languages = sorted(target_languages)
        if sorted_target_languages in [["en", "zh"], ["en", "ja"], ["en", "ja", "zh"]]:
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(split_alpha_nonalpha(sentence))
            sentences = new_sentences

    for sentence in sentences:
        if check_is_none(sentence):
            continue

        lang = classify_language(sentence, target_languages)

        if pre_lang == "":
            text = text[:p] + text[p:].replace(
                sentence, f"[{lang.upper()}]{sentence}", 1
            )
            p += len(f"[{lang.upper()}]")
        elif pre_lang != lang:
            text = text[:p] + text[p:].replace(
                sentence, f"[{pre_lang.upper()}][{lang.upper()}]{sentence}", 1
            )
            p += len(f"[{pre_lang.upper()}][{lang.upper()}]")
        pre_lang = lang
        p += text[p:].index(sentence) + len(sentence)
    text += f"[{pre_lang.upper()}]"

    return text

def move_numbers(arr):
    result = []
    i = 0
    while i < len(arr):
        current = arr[i]
        text, lang = current
        
        if lang == 'en':
            # Split the text to check for numbers at start or end
            words = text.split()
            if not words:  # Empty string case
                result.append(current)
                i += 1
                continue
                
            # Initialize new parts
            numbers_front = []
            numbers_back = []
            clean_words = []
            
            # Check each word
            for word in words:
                if word.replace('.','').isdigit():
                    if not clean_words:  # If we haven't found non-number words yet
                        numbers_front.append(word)
                    else:
                        numbers_back.append(word)
                else:
                    clean_words.append(word)
            
            # If we found numbers
            if numbers_front or numbers_back:
                # Modify previous zh entry if we have front numbers
                if numbers_front and i > 0 and result[-1][1] == 'zh':
                    prev_text, prev_lang = result[-1]
                    result[-1] = (prev_text + ' ' + ' '.join(numbers_front), prev_lang)
                
                # Add the clean English text
                if clean_words:
                    result.append((' ' + ' '.join(clean_words) + ' ', 'en'))
                
                # Look ahead for zh entry if we have back numbers
                if numbers_back and i+1 < len(arr) and arr[i+1][1] == 'zh':
                    next_text, next_lang = arr[i+1]
                    arr[i+1] = (' '.join(numbers_back) + ' ' + next_text, next_lang)
                
                i += 1
            else:
                result.append(current)
                i += 1
        else:
            result.append(current)
            i += 1
    
    return result

def clean_multiple_spaces(arr):
    return [(' '.join(text.split()), lang) for text, lang in arr]


def split_by_language(text: str, target_languages: list = None) -> list:
    pattern = (
        r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`"
        r"\！？\。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」"
        r"『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+"
    )
    sentences = re.split(pattern, text)

    pre_lang = ""
    start = 0
    end = 0
    sentences_list = []

    if target_languages is not None:
        sorted_target_languages = sorted(target_languages)
        if sorted_target_languages in [["en", "zh"]]:
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(split_alpha_nonalpha(sentence))
            sentences = new_sentences

    for sentence in sentences:
        if check_is_none(sentence):
            continue

        lang = classify_language(sentence, target_languages)

        end += text[end:].index(sentence)
        if pre_lang != "" and pre_lang != lang:
            sentences_list.append((text[start:end], pre_lang))
            start = end
        end += len(sentence)
        pre_lang = lang
    sentences_list.append((text[start:], pre_lang))
    sentences_list = clean_multiple_spaces(move_numbers(sentences_list))
    return sentences_list


def sentence_split(text: str, max: int) -> list:
    pattern = r"[!(),—+\-.:;?？。，、；：]+"
    sentences = re.split(pattern, text)
    discarded_chars = re.findall(pattern, text)

    sentences_list, count, p = [], 0, 0

    # 按被分割的符号遍历
    for i, discarded_chars in enumerate(discarded_chars):
        count += len(sentences[i]) + len(discarded_chars)
        if count >= max:
            sentences_list.append(text[p : p + count].strip())
            p += count
            count = 0

    # 加入最后剩余的文本
    if p < len(text):
        sentences_list.append(text[p:])

    return sentences_list


def sentence_split_and_markup(text, max=50, lang="auto", speaker_lang=None):
    # 如果该speaker只支持一种语言
    if speaker_lang is not None and len(speaker_lang) == 1:
        if lang.upper() not in ["AUTO", "MIX"] and lang.lower() != speaker_lang[0]:
            logging.debug(
                f'lang "{lang}" is not in speaker_lang {speaker_lang},automatically set lang={speaker_lang[0]}'
            )
        lang = speaker_lang[0]

    sentences_list = []
    if lang.upper() != "MIX":
        if max <= 0:
            sentences_list.append(
                markup_language(text, speaker_lang)
                if lang.upper() == "AUTO"
                else f"[{lang.upper()}]{text}[{lang.upper()}]"
            )
        else:
            for i in sentence_split(text, max):
                if check_is_none(i):
                    continue
                sentences_list.append(
                    markup_language(i, speaker_lang)
                    if lang.upper() == "AUTO"
                    else f"[{lang.upper()}]{i}[{lang.upper()}]"
                )
    else:
        sentences_list.append(text)

    for i in sentences_list:
        logging.debug(i)

    return sentences_list


if __name__ == "__main__":
    text = "这几天心里颇不宁静。今晚在院子里坐着乘凉，忽然想起日日走过的荷塘，在这满月的光里，总该另有一番样子吧。月亮渐渐地升高了，墙外马路上孩子们的欢笑，已经听不见了；妻在屋里拍着闰儿，迷迷糊糊地哼着眠歌。我悄悄地披了大衫，带上门出去。"
    print(markup_language(text, target_languages=None))
    print(sentence_split(text, max=50))
    print(sentence_split_and_markup(text, max=50, lang="auto", speaker_lang=None))

    text = "你好，这是一段用来测试自动标注的文本。こんにちは,これは自動ラベリングのテスト用テキストです.Hello, this is a piece of text to test autotagging.你好！今天我们要介绍VITS项目，其重点是使用了GAN Duration predictor和transformer flow,并且接入了Bert模型来提升韵律。Bert embedding会在稍后介绍。"
    print(split_by_language(text, ["zh", "ja", "en"]))

    text = "vits和Bert-VITS2是tts模型。花费3days.花费3天。Take 3 days"

    print(split_by_language(text, ["zh", "ja", "en"]))
    # output: [('vits', 'en'), ('和', 'ja'), ('Bert-VITS', 'en'), ('2是', 'zh'), ('tts', 'en'), ('模型。花费3', 'zh'), ('days.', 'en'), ('花费3天。', 'zh'), ('Take 3 days', 'en')]

    print(split_by_language(text, ["zh", "en"]))
    # output: [('vits', 'en'), ('和', 'zh'), ('Bert-VITS', 'en'), ('2是', 'zh'), ('tts', 'en'), ('模型。花费3', 'zh'), ('days.', 'en'), ('花费3天。', 'zh'), ('Take 3 days', 'en')]

    text = "vits 和 Bert-VITS2 是 tts 模型。花费 3 days. 花费 3天。Take 3 days"
    print(split_by_language(text, ["zh", "en"]))
    # output: [('vits ', 'en'), ('和 ', 'zh'), ('Bert-VITS2 ', 'en'), ('是 ', 'zh'), ('tts ', 'en'), ('模型。花费 ', 'zh'), ('3 days. ', 'en'), ('花费 3天。', 'zh'), ('Take 3 days', 'en')]

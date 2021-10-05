#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020/7/15 09:54

@author: huanghuajunjie
@author: limohan

This is a preprocess util to clean data
"""

import re


def _delete_sign(sentence):
    """
    删除标点符号，若该标点符号前后两个字符均是中文，则不删除，否则删除。
    :param sentence: 输入句子
    :return: 删除标点符号后的句子
    """

    # chs_sign = list("""“”！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰""")
    # eng_sign = list("""!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~""")

    # 去掉大部分中文和英文字符
    chs_sign = list("""“”＂＆＇（）＊＋，－／＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰""")
    eng_sign = list("""&<=>@_`""")

    sentence = re.sub(r'[\n\t\r]', ' ', sentence)
    sentence = re.sub('\u3000', ' ', sentence)
    sentence = re.sub('\xa0', ' ', sentence)
    sentence = re.sub(r'[’‘´]', '\'', sentence)

    # sentence = re.sub(r'([a-zA-Z]{1})/', r'\1 ', sentence)
    # sentence = re.sub(r'(—\*?)([a-zA-Z])', r' \2', sentence)
    sentence = sentence.replace(' / ', ' ')
    sentence = sentence.replace('\\', ' ')

    url = r"""((?<=[^a-zA-Z0-9])(?:https?\:\/\/|[a-zA-Z0-9]{1,}\.{1}|\b)(?:\w{1,}\.{1}){1,5}(?:com|org|edu|gov|uk|net|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|mil|iq|io|ac|ly|sm){1}(?:\/[a-zA-Z0-9]{1,})*)"""
    sentence = re.sub(url, ' ', sentence)

    email = r"""^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"""
    sentence = re.sub(email, ' ', sentence)

    # delete chinese sign
    pattern = r'[{}]'.format(''.join(chs_sign))
    sentence = re.sub(pattern, ' ', sentence)
    # delete english sign
    pattern = r'[{}]'.format(''.join(eng_sign))
    sentence = re.sub(pattern, ' ', sentence)
    # 某些特殊的英文字符也是正则的符号
    sentence = re.sub(r'[\|\{\}\(\)\+\[\]\^]', ' ', sentence)

    return sentence


def _tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


def _is_chinese(strs):
    """
    判断句子是否是中文，只要句子含有中文则认为是中文
    :param strs: 输入句子
    :return: 中文: zh, 英文: en
    """
    strs = str(strs)
    for char in strs:
        cp = ord(char)
        if _is_chinese_char(cp):
            return True
    return False


def data_clean(sent):
    if not sent:
        sent = ''
        return sent, False

    sent = sent.lower()

    is_cn = _is_chinese(sent)
    sent = _delete_sign(sent)

    return sent, is_cn


if __name__ == '__main__':
    data_clean('this is a test')

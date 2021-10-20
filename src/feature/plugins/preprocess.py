#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021/09/24 14:07

@author: limohan
"""

import re
import json
import logging
import numpy as np
import tokenization


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


def _data_clean(sent):
    if not sent:
        sent = ''
        return sent, False

    sent = sent.lower()

    is_cn = _is_chinese(sent)
    sent = _delete_sign(sent)

    return sent, is_cn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


def convert_examples(examples, tokenizer, seq_length):
    """Loads a data file into a list of `InputBatch`s."""

    all_uids, all_input_lens, all_input_ids, all_input_mask, all_input_type_ids = [], [], [], [], []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        if tokens_b:
            input_len = len(tokens_a)+len(tokens_b)
        else:
            input_len = len(tokens_a)

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        all_uids.append(example.unique_id)
        all_input_lens.append(input_len)
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_input_type_ids.append(input_type_ids)

    all_uids = np.array(all_uids)
    all_input_lens = np.array(all_input_lens)
    all_input_ids = np.array(all_input_ids)
    all_input_mask = np.array(all_input_mask)
    all_input_type_ids = np.array(all_input_type_ids)
    return all_uids, all_input_lens, all_input_ids, all_input_mask, all_input_type_ids


def read_examples(data):
    """Read a list of `InputExample`s from an input file."""
    examples = []

    for d in data:
        unique_id = d['content_id']
        text_a = tokenization.convert_to_unicode(d['item_content'])
        text_b = None
        # if not line:
        #     break
        # line = line.strip()
        # text_a = None
        # text_b = None
        # m = re.match(r"^(.*) \|\|\| (.*)$", line)
        # if m is None:
        #     text_a = line
        # else:
        #     text_a = m.group(1)
        #     text_b = m.group(2)
        examples.append(
            InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    return examples


def preprocess_data(data):
    new_data = []
    for d in data:
        content = d['item_content']
        # 帖子文本前面的特殊标记
        content = re.sub(r'(<[A-Z])\|([a-zA-Z0-9_]+)>', ' ', content)
        # 从数据源导出到hive中时替换的特殊文本字符
        content = content.replace('$HH$', ' ').replace('$KG$', ' ')

        if content:
            content, _ = _data_clean(content)
        else:
            content = ''

        new_data.append({'content_id': d['content_id'],
                         'item_content': content})

    # logging.info('preprocessing data done')

    return new_data


def preprocess_pipeline(data, tokenizer, max_seq_length):
    data = preprocess_data(data)
    examples = read_examples(data)
    uids, input_lens, input_ids, input_mask, input_type_ids = convert_examples(examples, tokenizer, max_seq_length)
    return uids, input_lens, input_ids, input_mask, input_type_ids

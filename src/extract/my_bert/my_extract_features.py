#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021/6/24 15:09

@author: limohan
"""

import collections
import json
import os
import re

import numpy as np
import pandas as pd

import modeling
import tokenization
import tensorflow as tf
import joblib

from utils.data_clean import data_clean

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string("output_dir", None, "")

flags.DEFINE_string("layers", "-1,-2,-3,-4", "")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", 256,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("batch_size", 16, "Batch size for predictions.")

flags.DEFINE_integer("log_step_count", 100,
                     "The frequency, in number of global steps, that the "
                     "global step and the loss will be logged during training.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def input_fn_builder(features, seq_length, batch_size):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)

    def input_fn():
        """The actual input function."""

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "unique_ids":
                tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_type_ids":
                tf.constant(
                    all_input_type_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
        })

        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        tvars = tf.trainable_variables()
        (assignment_map,
         initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint)

        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        all_layers = model.get_all_encoder_layers()
        # pooled_output is the [CLS] representation
        pooled_output = model.get_pooled_output()

        predictions = {
            "unique_id": unique_ids,
        }

        for (i, layer_index) in enumerate(layer_indexes):
            predictions["layer_output_%d" % i] = all_layers[layer_index]
        predictions['pooled_output'] = pooled_output

        output_spec = tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, scaffold=None)
        return output_spec

    return model_fn


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
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

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
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

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % (example.unique_id))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


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


def read_examples_from_file(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with tf.gfile.GFile(input_file, "r") as reader:
        while True:
            line = tokenization.convert_to_unicode(reader.readline())
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples


def preprocess_data(df):
    tf.logging.info('preprocessing data')
    df['id'] = np.arange(len(df))

    # df = df[['id', 'content_txt']]
    df = df.rename({'content_txt': 'content'}, axis=1)

    df['content'] = df['content'].apply(lambda x: (re.sub(r'<[A-Z]\|[0-9]+>', ' ', x)).strip())
    # 从数据源导出到hive中时替换的特殊文本字符
    df['content'] = df['content'].apply(lambda x: x.replace('$HH$', ' ').replace('$KG$', ' '))

    df = df[df['content'].notna()]
    df[['content', 'is_cn']] = df.apply(lambda x: data_clean(x['content']), axis=1, result_type='expand')

    df = df[~pd.isnull(df['content'])]

    df = df[df['content'].notna()]
    df = df[~(df['content'] == 'nan')]
    df = df[~(df['content'] == '')]

    tf.logging.info('preprocessing data done')

    return df


def read_examples(df):
    """Read a list of `InputExample`s from an input file."""
    examples = []

    for row in df.itertuples():
        unique_id = row.id
        line = tokenization.convert_to_unicode(row.content)
        if not line:
            break
        line = line.strip()
        text_a = None
        text_b = None
        m = re.match(r"^(.*) \|\|\| (.*)$", line)
        if m is None:
            text_a = line
        else:
            text_a = m.group(1)
            text_b = m.group(2)
        examples.append(
            InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    return examples


def get_features(estimator, input_fn, unique_id_to_feature, layer_indexes, prog_bar, output_dir):
    all_token_json = []
    all_pooled_json = []
    token_features_file = os.path.join(output_dir, 'token_features.json')
    pooled_features_file = os.path.join(output_dir, 'pooled_features.json')

    token_f = open(token_features_file, 'w', encoding='utf-8')
    pooled_f = open(pooled_features_file, 'w', encoding='utf-8')

    for prog, result in enumerate(estimator.predict(input_fn, yield_single_examples=True)):
        # convert numpy.int32 to int
        unique_id = int(result["unique_id"])
        feature = unique_id_to_feature[unique_id]
        token_json = collections.OrderedDict()
        pooled_json = collections.OrderedDict()
        token_json["line_index"] = unique_id
        pooled_json["line_index"] = unique_id

        token_features = []
        for (i, token) in enumerate(feature.tokens):
            all_layers = []
            for (j, layer_index) in enumerate(layer_indexes):
                layer_output = result["layer_output_%d" % j]
                layers = collections.OrderedDict()
                layers["index"] = layer_index
                # convert numpy.float32 to float
                layers["values"] = [
                    round(float(x), 6) for x in layer_output[i:(i + 1)].flat
                ]
                all_layers.append(layers)
            features = collections.OrderedDict()
            features["token"] = token
            features["layers"] = all_layers
            token_features.append(features)
        token_json["features"] = token_features
        all_token_json.append(token_json)
        token_f.write(json.dumps(token_json) + "\n")

        pooled_features = [round(float(x), 6) for x in result["pooled_output"]]
        pooled_json["pooled_features"] = pooled_features
        all_pooled_json.append(pooled_json)
        pooled_f.write(json.dumps(pooled_json) + "\n")

        prog_bar.update(prog)

    token_f.close()
    pooled_f.close()

    return all_token_json, all_pooled_json


def get_pooled_features(estimator, input_fn, prog_bar, output_dir):
    all_pooled_features = {}
    pooled_features_file = os.path.join(output_dir, 'pooled_features.json')

    # pooled_f = open(pooled_features_file, 'w', encoding='utf-8')

    for prog, result in enumerate(estimator.predict(input_fn, yield_single_examples=True)):
        # convert numpy.int32 to int
        unique_id = int(result["unique_id"])
        # pooled_json = collections.OrderedDict()
        # pooled_json["line_index"] = unique_id

        pooled_features = [round(float(x), 6) for x in result["pooled_output"]]
        all_pooled_features[unique_id] = pooled_features

        prog_bar.update(prog)

    # pooled_f.close()
    joblib.dump(all_pooled_features, pooled_features_file + '.gz', compress='gzip')
    # joblib.dump(all_pooled_json, pooled_features_file + '.gz', compress = ('gzip', 3))
    tf.logging.info('dumping pooled features to file done')

    return all_pooled_features


def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    layer_indexes = [int(x) for x in FLAGS.layers.split(",")]
    # layer_indexes = [-1, -2, -3, -4]

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    gpu_options = tf.GPUOptions(allow_growth=True)
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)

    run_config = tf.estimator.RunConfig(
        # session_config=tf_config,
        log_step_count_steps=FLAGS.log_step_count)

    # examples = read_examples_from_file(FLAGS.input_file)
    df = pd.read_csv(FLAGS.input_file, index_col=0)
    df = preprocess_data(df)
    examples = read_examples(df)

    features = convert_examples_to_features(
        examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        layer_indexes=layer_indexes,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)

    input_fn = input_fn_builder(
        features=features, seq_length=FLAGS.max_seq_length, batch_size=FLAGS.batch_size)

    prog_bar = tf.keras.utils.Progbar(len(examples))
    # get_features(estimator, input_fn, unique_id_to_feature, layer_indexes, prog_bar, FLAGS.output_dir)
    get_pooled_features(estimator, input_fn, prog_bar, FLAGS.output_dir)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()

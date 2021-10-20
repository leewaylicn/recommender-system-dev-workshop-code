#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021/09/24 15:27

@author: limohan
"""

from concurrent import futures
import logging
import os
import json
from datetime import datetime
import pickle

import joblib
import requests
import calendar
import time
import random
import sys
import numpy as np
import math
import shutil
import redis
from tensorflow.contrib import predictor
import tokenization

from preprocess import preprocess_pipeline
import cache


# Environments for service
MANDATORY_ENV_VARS = {
    'LOCAL_DATA_FOLDER': '/tmp/rs-data/',

    'REDIS_HOST': 'localhost',
    'REDIS_PORT': 6379,

    'MODEL_EXTRACT_DIR': '/Users/mohanli/opt/ml/model', # for debugging

    'FEATURE_SERVICE_ENDPOINT': 'http://portrait:5300'
}

# lastUpdate
localtime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")


item_content_key = 'item-raw-content'
item_feature_key = 'item-content-feature'
sleep_interval = 30


batch_size = 4
max_seq_length = 256


def data_batch_iter(all_data, batch_size):
    data_size = len(all_data)
    batch_number = math.ceil(data_size / batch_size)
    all_data = np.array(all_data)

    for batch in range(batch_number):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, data_size)

        batched_data = all_data[start_idx:end_idx]

        yield batched_data


# TODO: refactor to unified util class
def unpack_saved_model(saved_model_base_dir, model_name):
    # TODO: model versioning
    filename = f'{saved_model_base_dir}/{model_name}.tar.gz'
    target_dir = os.path.join(saved_model_base_dir, model_name)
    # if debug, comment below
    logging.info(f'extracting {filename} to {target_dir}')
    shutil.unpack_archive(filename, extract_dir=target_dir, format='gztar')
    logging.info(f'extracting done, removing {filename}')
    os.remove(filename)


def load_model():
    saved_model_base_dir = os.environ.get('MODEL_EXTRACT_DIR', default=MANDATORY_ENV_VARS['MODEL_EXTRACT_DIR'])
    base_model_path = os.path.join(saved_model_base_dir, 'base_model')
    value_model_path = os.path.join(saved_model_base_dir, 'value_model')

    base_model = predictor.from_saved_model(base_model_path)
    value_model = predictor.from_saved_model(value_model_path)

    std_scaler_path = os.path.join(saved_model_base_dir, 'std_scaler.bin')
    std_scaler = joblib.load(std_scaler_path)

    vocab_file = os.path.join(saved_model_base_dir, 'base_model', 'assets', 'vocab.txt')
    tokenizer = tokenization.FullTokenizer(
        vocab_file, do_lower_case=True)

    return base_model, value_model, std_scaler, tokenizer


def reload_model():
    pass


# features_spec = {
#     "unique_ids": tf.placeholder(tf.int64, name="unique_ids"),
#     "input_ids": tf.placeholder(tf.int64, shape=[None, FLAGS.max_seq_length], name="input_ids"),
#     "input_mask": tf.placeholder(tf.int64, shape=[None, FLAGS.max_seq_length], name="input_mask"),
#     "input_type_ids": tf.placeholder(tf.int64, shape=[None, FLAGS.max_seq_length], name="input_type_ids"),
# }
def predict_pipeline(data, base_model, value_model, std_scaler, tokenizer):
    data_iter = data_batch_iter(data, batch_size)
    feature_elements = []
    for d in data_iter:
        uids, input_lens, input_ids, input_mask, input_type_ids = preprocess_pipeline(d, tokenizer, max_seq_length=max_seq_length)

        base_input_dict = {"unique_ids": uids, "input_ids": input_ids,
                           "input_mask": input_mask, "input_type_ids": input_type_ids}

        base_feat = base_model(base_input_dict)['pooled_output']
        scaled_input_lens = std_scaler.transform(input_lens.reshape(-1, 1))

        value_input = np.concatenate([base_feat, scaled_input_lens], axis=-1)
        value_input_dict = {'dense_input': value_input}
        value = value_model(value_input_dict)['dense_2']
        value = np.squeeze(value, axis=-1)
        # !numpy.int32, numpy.float32 is not json serializable
        feature_elements.extend([{'content_id': i.item(), 'value_feature': v.item()} for i, v in zip(uids, value)])

    return feature_elements


# 循环读取缓存的帖子内容并得到价值评分
def get_item_feature():
    base_model, value_model, std_scaler, tokenizer = load_model()
    while True:
        try:
            elements = read_from_cache(item_content_key)

            if elements:
                feature_elements = predict_pipeline(elements, base_model, value_model, std_scaler, tokenizer)
                write_to_cache(item_feature_key, feature_elements)
            else:
                logging.info('no content in cache right now')
        except redis.ConnectionError:
            localtime = time.asctime(time.localtime(time.time()))
            logging.info('get ConnectionError, time: {}'.format(localtime))

        time.sleep(sleep_interval)


def read_from_cache(key):
    elements = []
    # while True:
    for i in range(4):
        try:
            elem = rCache.lpop_data_from_list(key)
            if elem:
                elements.append(json.loads(elem))
            else:
                break
        except Exception as e:
            logging.warning(f'error message {e}')
            localtime = time.asctime(time.localtime(time.time()))
            logging.info('error reading element from cache, time: {}'.format(localtime))
    return elements


def write_to_cache(key, elements):
    for elem in elements:
        elem = json.dumps(elem).encode('utf-8')
        try:
            rCache.rpush_data_into_list(key, elem)
        except Exception as e:
            logging.warning(f'error message {e}')
            localtime = time.asctime(time.localtime(time.time()))
            logging.info('error writing element to cache, time: {}'.format(localtime))


def init():
    # Check out environments
    for var in MANDATORY_ENV_VARS:
        if var not in os.environ:
            logging.error("Mandatory variable {%s} is not set, using default value {%s}.", var, MANDATORY_ENV_VARS[var])
        else:
            MANDATORY_ENV_VARS[var] = os.environ.get(var)
    
    # Initial redis connection
    global rCache
    rCache = cache.RedisCache(host=MANDATORY_ENV_VARS['REDIS_HOST'], port=MANDATORY_ENV_VARS['REDIS_PORT'])

    saved_model_base_dir = os.environ.get('MODEL_EXTRACT_DIR', default=MANDATORY_ENV_VARS['MODEL_EXTRACT_DIR'])
    model_names = ['base_model', 'value_model']
    for name in model_names:
        unpack_saved_model(saved_model_base_dir, name)


def serve(plugin_name):
    # server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    # service_pb2_grpc.add_FilterServicer_to_server(Filter(), server)
    # SERVICE_NAMES = (
    #     service_pb2.DESCRIPTOR.services_by_name['Filter'].full_name,
    #     reflection.SERVICE_NAME,
    # )
    # reflection.enable_server_reflection(SERVICE_NAMES, server)
    # logging.info('Plugin - %s is listening at 50051...', plugin_name)
    # server.add_insecure_port('[::]:50051')
    # logging.info('Plugin - %s is ready to serve...', plugin_name)
    # server.start()
    # server.wait_for_termination()

    get_item_feature()


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    print('feature plugin start')
    init()
    serve(os.environ.get("PLUGIN_NAME", "default"))

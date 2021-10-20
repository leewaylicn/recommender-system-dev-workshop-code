#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import redis
import logging

class RedisCache:

    def __init__(self, host='localhost', port=6379, db=0):
        logging.info('Initial RedisCache ...')
        # Initial connection to Redis
        logging.info('Connect to Redis %s:%s ...', host, port)
        self.rCon = redis.Redis(host=host, port=port, db=db)

    def connection_status(self):
        return self.rCon.client_list()

    def rpush_data_into_list(self, key, element):
        self.rCon.rpush(key, element)

    def lpop_data_from_list(self, key):
        return self.rCon.lpop(key)

        




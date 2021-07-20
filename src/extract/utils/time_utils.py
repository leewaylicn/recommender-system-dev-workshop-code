#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020/9/10 14:57

@author: limohan
"""

import time
import functools
from contextlib import contextmanager
import logging

# nano seconds
# time.time_ns()  1ns = 1e9 s

def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        result = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        logging.info('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
        return result
    return newfunc


@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    logging.info('[{}] finished in {} ms'.format(name, int(elapsedTime * 1000)))


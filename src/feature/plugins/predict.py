#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021/09/24 15:27

@author: limohan
"""

from tensorflow.contrib import predictor

input_dict = {}

model_path = './post_opt/saved_model'
model = predictor.from_saved_model(model_path)

output = model(input_dict)
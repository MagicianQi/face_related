# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import numpy as np

from utils.OOP import calculate_matrix_cosine_similarity
from settings import MATCH_THRESHOLD


class MatchingModel(object):
    def __init__(self):
        self._lock = threading.Lock()
        self._embeddings_dict = {}

    def add(self, name, embedding):
        with self._lock:
            if name not in self._embeddings_dict:
                self._embeddings_dict.setdefault(name, [embedding])
            else:
                self._embeddings_dict[name].append(embedding)

    def delete(self, name):
        with self._lock:
            self._embeddings_dict.pop(name)

    def inference(self, embedding):
        cos_list = []
        with self._lock:
            for key, value in self._embeddings_dict.items():
                cos = np.mean(calculate_matrix_cosine_similarity(value, embedding))
                cos_list.append([key, cos])
            result = max(cos_list, key=lambda x: x[1])
            print(result)
            if result[1] >= MATCH_THRESHOLD:
                return result[0]
            else:
                return "None"

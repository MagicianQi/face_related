# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity


def calculate_vector_cosine_similarity(embedding_a, embedding_b):
    cos = np.dot(embedding_a, embedding_b) / (np.linalg.norm(embedding_a) * (np.linalg.norm(embedding_b)))
    sim = 0.5 + 0.5 * cos
    return sim


def calculate_vector_euclidean_distance(embedding_a, embedding_b):
    diff = np.subtract(embedding_a, embedding_b)
    dist = np.sum(np.square(diff), 1)
    return dist


def calculate_matrix_cosine_similarity(matrix_a, matrix_b):
    return cosine_similarity(matrix_a, matrix_b)


def calculate_matrix_euclidean_distance(matrix_a, matrix_b):
    return euclidean_distances(matrix_a, matrix_b)


if __name__ == "__main__":
    pass

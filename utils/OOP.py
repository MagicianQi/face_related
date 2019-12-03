# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import base64
import random
import requests
import numpy as np

from PIL import Image, ImageSequence
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

from settings import *


def process_gif(image_gif):
    frame_list = [frame.copy() for frame in ImageSequence.Iterator(image_gif)]
    frame_list = [frame_list[i] for i in range(len(frame_list)) if i % GIF_FRAME_INTERVAL == 0]
    if len(frame_list) > GIF_MAX_FRAME:
        return random.sample(frame_list, GIF_MAX_FRAME)
    else:
        return frame_list


def download_image(image_url):
    response = requests.get(image_url)
    response = response.content
    bytes_obj = io.BytesIO(response)
    image = Image.open(bytes_obj)
    image = image.convert("RGB")
    return image


def base64_to_image(image_base64):
    data = base64.b64decode(image_base64)
    image = io.BytesIO(data)
    image = Image.open(image)
    image = image.convert("RGB")
    return image


def image_to_base64(image):
    if type(image) == "numpy.ndarray":
        image = Image.fromarray(image)
    image_pil = io.BytesIO()
    image.save(image_pil, format='PNG')
    return base64.b64encode(image_pil.getvalue()).decode('utf-8')


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

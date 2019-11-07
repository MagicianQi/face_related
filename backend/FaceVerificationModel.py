# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import random
import base64
import logging
import requests
import numpy as np

from PIL import Image, ImageSequence
from scipy import misc

from settings import *


class FaceVerificationModel(object):

    def __init__(self,
                 face_detection_server_url=FACE_DETECTION_SERVER_URL,
                 face_verification_server_url=FACE_VERIFICATION_SERVER_URL,
                 face_detection_threshold=FACE_DETECTION_THRESHOLD,
                 image_size=IMAGE_SIZE):
        self.face_detection_server_url = face_detection_server_url
        self.face_verification_server_url = face_verification_server_url
        self.face_detection_threshold = face_detection_threshold
        self.image_size = image_size
        self.gif_frame_interval = GIF_FRAME_INTERVAL
        self.gif_max_frame = GIF_MAX_FRAME

    def get_face_detection_boxes_and_scores(self, img_b64):
        post_json = {
            "instances": [{
                "b64": img_b64
            }]
        }
        response = requests.post(self.face_detection_server_url,
                                 data=json.dumps(post_json))
        response.raise_for_status()
        boxes = response.json()["predictions"][0]["boxes"]
        scores = response.json()["predictions"][0]["scores"]

        return boxes, scores

    def get_face_embedding(self, img_b64):
        post_json = {
            "instances": [{
                "b64": img_b64
            }]
        }
        response = requests.post(self.face_verification_server_url, data=json.dumps(post_json))
        print(response.content)
        response.raise_for_status()
        prediction = response.json()["predictions"][0]
        return prediction

    def get_image_embeddings_from_array(self, image):
        embeddings = []
        transform_boxes = []
        image_height, image_width, _ = image.shape
        try:
            boxes, scores = self.get_face_detection_boxes_and_scores(self.convert_image_to_base64_from_array(image))
            for box, score in zip(boxes, scores):
                if score < self.face_detection_threshold:
                    continue
                box = box * np.array([image_height, image_width, image_height, image_width], dtype='float32')
                transform_boxes.append([int(box[1]), int(box[0]), int(box[3]), int(box[2])])
                cropped = image[int(box[0]):int(box[2]), int(box[1]):int(box[3]), :]
                aligned = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
                embeddings.append(self.get_face_embedding(self.convert_image_to_base64_from_array(aligned,
                                                                                                  url_safe=False)))
        except Exception as e:
            logging.error(e)
        return embeddings, transform_boxes

    def get_image_embeddings_from_local(self, image_path):
        image = misc.imread(os.path.expanduser(image_path), mode='RGB')
        return self.get_image_embeddings_from_array(image)

    def get_image_embeddings_from_url(self, image_url):
        response = requests.get(image_url)
        response = response.content
        bytes_obj = io.BytesIO(response)
        image = Image.open(bytes_obj)
        if image.format == "GIF":
            embeddings = []
            frame_list = self.process_gif(image)
            for frame in frame_list:
                frame = frame.convert("RGB")
                frame = np.array(frame)
                embeddings.extend(self.get_image_embeddings_from_array(frame))
            return embeddings
        else:
            image = image.convert("RGB")
            image = np.array(image)
            return self.get_image_embeddings_from_array(image)

    @staticmethod
    def convert_image_to_base64_from_array(image_array, url_safe=False):
        image = Image.fromarray(image_array)
        image_pil = io.BytesIO()
        image.save(image_pil, format='PNG')
        if url_safe:
            return base64.urlsafe_b64encode(image_pil.getvalue()).decode('utf-8')
        else:
            return base64.b64encode(image_pil.getvalue()).decode('utf-8')

    def process_gif(self, image_gif):
        frame_list = [frame.copy() for frame in ImageSequence.Iterator(image_gif)]
        frame_list = [frame_list[i] for i in range(len(frame_list)) if i % self.gif_frame_interval == 0]
        if len(frame_list) > self.gif_max_frame:
            return random.sample(frame_list, self.gif_max_frame)
        else:
            return frame_list


if __name__ == "__main__":
    pass

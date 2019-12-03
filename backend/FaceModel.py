# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import dlib
import requests
import numpy as np

from PIL import Image
from scipy import misc

from settings import *
from utils.OOP import image_to_base64


class FaceModel(object):

    def __init__(self,
                 face_detection_server_url=FACE_DETECTION_SERVER_URL,
                 face_verification_server_url=FACE_VERIFICATION_SERVER_URL,
                 face_calibration_model_path=FACE_CALIBRATION_MODEL_PATH,
                 face_detection_threshold=FACE_DETECTION_THRESHOLD,
                 image_size=IMAGE_SIZE):
        self.face_detection_server_url = face_detection_server_url
        self.face_verification_server_url = face_verification_server_url
        self.face_detection_threshold = face_detection_threshold
        self.face_calibration_model = dlib.shape_predictor(face_calibration_model_path)
        self.image_size = image_size

    def face_detection_tf(self, image_b64):
        post_json = {
            "instances": [{
                "b64": image_b64
            }]
        }
        response = requests.post(self.face_detection_server_url,
                                 data=json.dumps(post_json))
        response.raise_for_status()
        boxes = response.json()["predictions"][0]["boxes"]
        scores = response.json()["predictions"][0]["scores"]

        return boxes, scores

    def face_calibration_dlib(self, image_pil, box):
        top = int(round(box[0]))
        bottom = int(round(box[2]))
        left = int(round(box[1]))
        right = int(round(box[3]))
        face = dlib.get_face_chip(
            np.array(image_pil),
            self.face_calibration_model(
                np.array(image_pil),
                dlib.rectangle(left=left, top=top, right=right, bottom=bottom)
            ),
            size=self.image_size
        )
        return Image.fromarray(face, mode="RGB")

    def face_encoding_tf(self, image_b64):
        post_json = {
            "instances": [{
                "b64": image_b64
            }]
        }
        response = requests.post(self.face_verification_server_url, data=json.dumps(post_json))
        response.raise_for_status()
        prediction = response.json()["predictions"][0]
        return prediction

    def predict_img_without_dlib(self, image):
        embeddings = []
        transform_boxes = []
        image = np.array(image)
        image_height, image_width, _ = image.shape
        try:
            boxes, scores = self.face_detection_tf(image_to_base64(image))
            for box, score in zip(boxes, scores):
                if score < self.face_detection_threshold:
                    continue
                box = box * np.array([image_height, image_width, image_height, image_width], dtype='float32')
                transform_boxes.append([int(box[1]), int(box[0]), int(box[3]), int(box[2])])
                cropped = image[int(box[0]):int(box[2]), int(box[1]):int(box[3]), :]
                aligned = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
                embeddings.append(self.face_encoding_tf(image_to_base64(aligned)))
        except Exception as e:
            raise Exception(e)
        return embeddings, transform_boxes

    def predict_img(self, image):
        embeddings = []
        transform_boxes = []
        image_width, image_height = image.size
        try:
            boxes, scores = self.face_detection_tf(image_to_base64(image))
            for box, score in zip(boxes, scores):
                if score < self.face_detection_threshold:
                    continue
                box = box * np.array([image_height, image_width, image_height, image_width], dtype='float32')
                transform_boxes.append([int(box[1]), int(box[0]), int(box[3]), int(box[2])])
                calibrated = self.face_calibration_dlib(image, [int(box[0]), int(box[1]), int(box[2]), int(box[3])])
                embeddings.append(self.face_encoding_tf(image_to_base64(calibrated)))
        except Exception as e:
            raise Exception(e)
        return embeddings, transform_boxes


if __name__ == "__main__":
    pass

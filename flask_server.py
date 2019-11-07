# -*- coding: utf-8 -*-

import flask
import json

from backend.FaceVerificationModel import FaceVerificationModel
from utils.OOP import calculate_vector_cosine_similarity

app = flask.Flask(__name__)
face_verification_model = FaceVerificationModel()


@app.route("/")
def homepage():
    return "Welcome to the SOUL Face Verification REST API!"


@app.route("/health")
def health_check():
    return "OK"


@app.route("/api/predict/boxes", methods=["POST", "GET"])
def predict_boxes():
    res = {
        "code": 200,
        "message": "OK"
    }
    if flask.request.method == "POST":
        data = flask.request.data.decode('utf-8')
        data = json.loads(data)
        if "imgUrl" in data:
            _, boxes = face_verification_model.get_image_embeddings_from_url(data["imgUrl"])
            res.update({"boxes": boxes})
        else:
            res.update({"code": 400, "message": "Bad request."})
        print(res)
    return flask.jsonify(res)


@app.route("/api/predict/similarity", methods=["POST", "GET"])
def predict_similarity():
    res = {
        "code": 200,
        "message": "OK"
    }
    if flask.request.method == "POST":
        data = flask.request.data.decode('utf-8')
        data = json.loads(data)
        if "imgUrl" in data:
            embedding_a, _ = face_verification_model.get_image_embeddings_from_url(data["imgUrl"][0])
            embedding_b, _ = face_verification_model.get_image_embeddings_from_url(data["imgUrl"][1])
            cos = calculate_vector_cosine_similarity(embedding_a[0], embedding_b[0])
            res.update({"similarity": cos})
        else:
            res.update({"code": 400, "message": "Bad request."})
        print(res)
    return flask.jsonify(res)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)

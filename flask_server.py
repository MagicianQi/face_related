# -*- coding: utf-8 -*-

import flask
import json

from backend.FaceModel import FaceModel
from utils.OOP import calculate_vector_cosine_similarity, download_image, base64_to_image

app = flask.Flask(__name__)
face_model = FaceModel()


@app.route("/")
def homepage():
    return "Welcome to the Face REST API!"


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
        try:
            data = flask.request.data.decode('utf-8')
            data = json.loads(data)
            if "imgUrl" in data:
                _, boxes = face_model.predict_img(download_image(data["imgUrl"]))
                res.update({"boxes": boxes})
                res.update({"imgUrl": data["imgUrl"]})
            elif "imgBase64" in data:
                _, boxes = face_model.predict_img(base64_to_image(data["imgBase64"]))
                res.update({"boxes": boxes})
            else:
                res.update({"code": 400, "message": "Bad request."})
            print("response\t{}".format(res))
        except Exception as e:
            print("Error\t{}".format(e))
    return flask.jsonify(res)


@app.route("/api/predict/similarity", methods=["POST", "GET"])
def predict_similarity():
    res = {
        "code": 200,
        "message": "OK"
    }
    if flask.request.method == "POST":
        try:
            data = flask.request.data.decode('utf-8')
            data = json.loads(data)
            if "imgPairs_url" in data:
                embedding_a, _ = face_model.predict_img(download_image(data["imgPairs_url"][0]))
                embedding_b, _ = face_model.predict_img(download_image(data["imgPairs_url"][1]))
                cos = calculate_vector_cosine_similarity(embedding_a[0], embedding_b[0])
                res.update({"similarity": cos})
            elif "imgPairs_base64" in data:
                embedding_a, _ = face_model.predict_img(base64_to_image(data["imgPairs_base64"][0]))
                embedding_b, _ = face_model.predict_img(base64_to_image(data["imgPairs_base64"][1]))
                cos = calculate_vector_cosine_similarity(embedding_a[0], embedding_b[0])
                res.update({"similarity": cos})
            else:
                res.update({"code": 400, "message": "Bad request."})
            print("response\t{}".format(res))
        except Exception as e:
            print("Error\t{}".format(e))
    return flask.jsonify(res)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8585)

# -*- coding: utf-8 -*-

import flask
import json

from backend.FaceModel import FaceModel
from backend.MatchingModel import MatchingModel
from utils.OOP import calculate_vector_cosine_similarity, download_image, base64_to_image, draw_boxes

app = flask.Flask(__name__)
face_model = FaceModel()
match_model = MatchingModel()


@app.route("/")
def homepage():
    return "Welcome to the Face REST API!"


@app.route("/health")
def health_check():
    return "OK"


@app.route("/api/embedding/add", methods=["POST", "GET"])
def add_embedding():
    res = {
        "code": 200,
        "message": "OK"
    }
    if flask.request.method == "POST":
        try:
            data = flask.request.data.decode('utf-8')
            data = json.loads(data)
            name = data["name"]
            if "imgUrl" in data:
                embedding, _ = face_model.predict_img(download_image(data["imgUrl"]))
                match_model.add(name, embedding[0])
                res.update({"imgUrl": data["imgUrl"]})
                res.update({"message": "add OK."})
            elif "imgBase64" in data:
                embedding, _ = face_model.predict_img(base64_to_image(data["imgBase64"]))
                match_model.add(name, embedding[0])
                res.update({"message": "add OK."})
            else:
                res.update({"code": 400, "message": "Bad request."})
            print("response\t{}".format(res))
        except Exception as e:
            print("Error\t{}".format(e))
    return flask.jsonify(res)


@app.route("/api/embedding/delete", methods=["POST", "GET"])
def delete_embedding():
    res = {
        "code": 200,
        "message": "OK"
    }
    if flask.request.method == "POST":
        try:
            data = flask.request.data.decode('utf-8')
            data = json.loads(data)
            name = data["name"]
            match_model.delete(name)
            res.update({"message": "delete OK."})
            print("response\t{}".format(res))
        except Exception as e:
            print("Error\t{}".format(e))
    return flask.jsonify(res)


@app.route("/api/predict/name", methods=["POST", "GET"])
def predict_name():
    res = {
        "code": 200,
        "message": "OK"
    }
    if flask.request.method == "POST":
        try:
            data = flask.request.data.decode('utf-8')
            data = json.loads(data)
            if "imgUrl" in data:
                embedding, _ = face_model.predict_img(download_image(data["imgUrl"]))
                name = match_model.inference(embedding)
                res.update({"imgUrl": data["imgUrl"]})
                res.update({"name": name})
            elif "imgBase64" in data:
                embedding, _ = face_model.predict_img(base64_to_image(data["imgBase64"]))
                name = match_model.inference(embedding)
                res.update({"name": name})
            else:
                res.update({"code": 400, "message": "Bad request."})
            print("response\t{}".format(res))
        except Exception as e:
            print("Error\t{}".format(e))
    return flask.jsonify(res)


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


@app.route("/test/detection", methods=["GET"])
def detection_test():
    return flask.render_template("home.html")


@app.route("/api/draw", methods=["GET"])
def draw():
    if flask.request.method == "GET":
        img_url = flask.request.args.get('ImageUrl')
        _, boxes = face_model.predict_img(download_image(img_url))
        return draw_boxes(img_url, ["{}".format(x+1) for x in range(len(boxes))], boxes)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8585)

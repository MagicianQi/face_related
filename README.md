# face_related

* Face Detection(TF) (https://github.com/TropComplique/FaceBoxes-tensorflow)
* Face Calibration(dlib)
* Face Feature Extraction(TF)(https://github.com/davidsandberg/facenet)

## The environment

* docker - https://docs.docker.com/install/linux/docker-ce/ubuntu/
* nvidia-docker - https://github.com/NVIDIA/nvidia-docker

## Docker image(Flask)

https://drive.google.com/open?id=1M9GRzR9k9X-ilGrigqouay17ChuEzhfb

## Run

* `sudo docker pull tensorflow/serving:1.13.0-gpu`
* `sudo docker load -i flask-uwsgi-python-centos.tar`
* `git clone https://github.com/MagicianQi/face_related`
* `cd face_related && wget https://github.com/MagicianQi/face_related/releases/download/v0.1/model.tar.gz && tar -zxvf model.tar.gz`
* Specify the GPU ID: `vim deploy.sh`

    1.https://github.com/MagicianQi/face_related/blob/master/deploy.sh#L5
    2.https://github.com/MagicianQi/face_related/blob/master/deploy.sh#L21
* Specify the model absolute path：`vim deploy.sh`
    1.https://github.com/MagicianQi/face_related/blob/master/deploy.sh#L7
    2.https://github.com/MagicianQi/face_related/blob/master/deploy.sh#L23
* `bash deploy.sh`
* Test: `curl localhost:8080`

## Other

* Other API：https://github.com/MagicianQi/face_related/blob/master/flask_server.py

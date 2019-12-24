#!/usr/bin/env bash
# Face Detection Model

sudo docker run --runtime=nvidia \
-e CUDA_VISIBLE_DEVICES=0 \
--name face_detection \
-d -v /home/qishuo/code/face_related/models/face_detection:/models/face_detection \
--restart always \
-p 6902:6902 \
-t --entrypoint=tensorflow_model_server tensorflow/serving:1.13.0-gpu \
--rest_api_port=6902 \
--model_name=face_detection \
--model_base_path=/models/face_detection \
--per_process_gpu_memory_fraction=0.1

# Face Verification Model

sudo docker run --runtime=nvidia \
--name face_verification \
--restart always \
-d -e CUDA_VISIBLE_DEVICES=0 \
-p 8540:8540 -p 8541:8541 \
--mount type=bind,source=/home/qishuo/code/face_related/models/face_verification,target=/models/face_verification \
-t --entrypoint=tensorflow_model_server tensorflow/serving:1.13.0-gpu \
--port=8540 --rest_api_port=8541 \
--model_name=face_verification \
--model_base_path=/models/face_verification \
--enable_batching \
--per_process_gpu_memory_fraction=0.7

# Build flask docker image

sudo docker build -t face_image .

# Sensitive Model Flask

sudo docker run \
--name face_service \
--link face_detection:face_detection \
--link face_verification:face_verification \
-p 8089:8080 -itd -P face_image

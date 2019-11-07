# 敏感人物识别

接口文档：https://wiki.soulapp-inc.cn/pages/viewpage.action?pageId=7210451

## 环境

* docker 
* python3.6

## How to use

* 修改配置文件settings.py中Redis配置以及人脸检测模型和人脸识别模型的TF Serving地址
* 修改uwsgi.ini和Dockerfile中端口
* bash deploy.sh
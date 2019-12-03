# 指定父镜像
FROM harbor.soulapp-inc.cn/soul-ops/flask-uwsgi-python-centos:3.6.5
# 声明镜像制作人
MAINTAINER "qishuo"
# 将项目拷贝到镜像的/opt/app/目录下
ADD . /opt/app/
# 编译dlib需要
RUN yum install -y cmake
# 安装项目所需的相关依赖包
RUN source /opt/servers/setenv.sh && \
    pip install -r /opt/app/requirements.txt

EXPOSE 8080
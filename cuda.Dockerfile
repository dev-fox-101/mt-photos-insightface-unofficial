FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
USER root
# 镜像加速
COPY ./cuda/sources.list /etc/apt/sources.list

RUN apt-get update && apt-get install -y python3 python3-pip

# 将模型文件复制到镜像内
COPY ./models /root/.insightface/models

WORKDIR /app
COPY ./cuda/requirements.txt .
# 安装依赖包
RUN pip3 install --no-cache-dir -r requirements.txt --index-url=https://pypi.tuna.tsinghua.edu.cn/simple/

COPY ./_insightface_code /root/insightface_code
RUN pip3 install /root/insightface_code

COPY ./cuda/server.py .

ENV API_AUTH_KEY=mt_photos_ai_extra
ENV RECOGNITION_MODEL=buffalo_l
ENV DETECTION_THRESH=0.65
EXPOSE 8066

CMD [ "python3", "server.py" ]

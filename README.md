# MT Photos 人脸识别API

镜像仓库地址：https://hub.docker.com/r/devfox101/mt-photos-insightface-unofficial

> **更新记录：** 
> 
> V1.1.0  -  2025-03-14
> 
> 1、增加容器启动后延迟加载模型，以及2分钟内没有识别任务自动释放模型内存


## 模型选择

insightface提供了3种模型可供选择，从上到下精度逐渐下降，可获得更快的识别速度，默认使用buffalo_l模型(镜像在打包时，已内置这个模型)

可通过环境变量 `RECOGNITION_MODEL`来自定义特征提取模型；

```python
models = [
    "antelopev2",
    "buffalo_l",
    "buffalo_s",
]
recognition_model = os.getenv("RECOGNITION_MODEL", "buffalo_l")
```

初始化时会自动下载指定模型，根据连接速度可能需要等待数分钟时间

所有模型向量长度均为默认512即可





## 安装方法

- 下载镜像

```
docker pull kqstone/mt-photos-insightface-unofficial:latest
```

- 创建及运行容器

```
docker run -i -p 8066:8066 -e API_AUTH_KEY=mt_photos_ai_extra --name mt-photos-insightface-unofficial --restart="unless-stopped" kqstone/mt-photos-insightface-unofficial:latest
```



## 打包docker镜像

### 下载模型文件

从下面的地址下载模型，然后放到 ./models 文件夹里

 - 模型文件下载地址：https://github.com/kqstone/mt-photos-insightface-unofficial/releases/tag/models
 - 或者百度网盘：https://pan.baidu.com/s/1SsY7_2t7aORh2jCvGWtD1A?pwd=1234

比如下载buffalo_l.zip ，然后把文件解压到 ./models/buffalo_l 这个目录

解压完成后，在 ./models/buffalo_l目录下，可以看到 1k3d68.onnx、2d106det.onnx、det_10g.onnx、genderage.onnx、w600k_r50.onnx 5个文件

如果放在的目录错误，在打包镜像时，会提示 `COPY ./models /root/.insightface/models` 这一行错误

### 打包镜像

cpu识别镜像打包
```bash
docker build  . -t mt-photos-insightface-unofficial:latest
```

cuda镜像打包
```bash
docker build -f cuda.Dockerfile . -t mt-photos-insightface-unofficial:cuda-latest
```

### 下载源码本地运行

- 安装python **3.8版本**
- 在文件夹下执行`pip install -r requirements.txt`
- 复制`.env.example`生成`.env`文件，然后修改`.env`文件内的API_AUTH_KEY
- 执行 `python server.py` ，启动服务

看到以下日志，则说明服务已经启动成功
```bash
INFO:     Started server process [27336]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8066 (Press CTRL+C to quit)
```


## API

### /check

检测服务是否可用，及api-key是否正确

```bash
curl --location --request POST 'http://127.0.0.1:8000/check' \
--header 'api-key: api_key'
```

**response:**

```json
{
  "result": "pass"
}
```

### /represent

```bash
curl --location --request POST 'http://127.0.0.1:8000/represent' \
--header 'api-key: api_key' \
--form 'file=@"/path_to_file/test.jpg"'
```

**response:**

- detector_backend : "insightface",
- recognition_model : 识别模型
- result : 识别到的结果

### 返回数据示例
```json
{
  "detector_backend": "insightface",
  "recognition_model": 识别模型,
  "result": [
    {
      "embedding": [ 0.5760641694068909,... 512位向量 ],
      "facial_area": {
        "x": 212,
        "y": 112,
        "w": 179,
        "h": 250,
      },
      "face_confidence": 1.0
    }
  ]
}
```

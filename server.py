from dotenv import load_dotenv
import os
import sys
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Header
import uvicorn
import numpy as np
import cv2
import asyncio
from PIL import Image
from io import BytesIO
import insightface
from insightface.utils import storage
from insightface.app import FaceAnalysis
import logging

logging.basicConfig(level=logging.WARNING)


on_linux = sys.platform.startswith('linux')

load_dotenv()
app = FastAPI()
api_auth_key = os.getenv("API_AUTH_KEY", "mt_photos_ai_extra")
http_port = int(os.getenv("HTTP_PORT", "8066"))


inactive_task = None


detector_backend = os.getenv("DETECTOR_BACKEND", "insightface")


# 人脸检测及特征提取模型
models = [
    "antelopev2",
    "buffalo_l",
    "buffalo_m",
    "buffalo_s",
]
recognition_model = os.getenv("RECOGNITION_MODEL", "buffalo_l")

detection_thresh = float(os.getenv("DETECTION_THRESH", "0.65"))

# 设置下载模型URL
storage.BASE_REPO_URL = 'https://github.com/kqstone/mt-photos-insightface-unofficial/releases/download/models'

on_win = sys.platform.startswith('win')
model_folder_path = '~/.insightface'
if on_win :
    current_folder = os.path.dirname(os.path.abspath(__file__))
    model_folder_path = os.path.join(current_folder, "_insightface_root")

# 初始化人脸识别器
faceAnalysis = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],root=model_folder_path, allowed_modules=['detection', 'recognition'], name=recognition_model)
faceAnalysis.prepare(ctx_id=0, det_thresh=detection_thresh, det_size=(640, 640))


async def check_inactive():
    await asyncio.sleep(3600)
    restart_program()


@app.middleware("http")
async def check_activity(request, call_next):
    global inactive_task
    if inactive_task:
        inactive_task.cancel()

    inactive_task = asyncio.create_task(check_inactive())
    response = await call_next(request)
    return response


async def verify_header(api_key: str = Header(...)):
    # 在这里编写验证逻辑，例如检查 api_key 是否有效
    if api_key != api_auth_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

@app.get("/")
async def top_info():
    return {"title": "unofficial face recognition api for mt-photos, get more info: https://github.com/kqstone/mt-photos-insightface-unofficial", "link": "https://mtmt.tech/docs/advanced/facial_api","detector_backend": detector_backend, "recognition_model": recognition_model}




@app.post("/check")
async def check_req(api_key: str = Depends(verify_header)):
    return {'result': 'pass'}


@app.post("/restart")
async def check_req(api_key: str = Depends(verify_header)):
    # 客户端可调用，触发重启进程来释放内存
    restart_program()


@app.post("/represent")
async def process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    content_type = file.content_type

    image_bytes = await file.read()
    try:
        img = None
        if content_type == 'image/gif':
            # Use Pillow to read the first frame of the GIF file
            with Image.open(BytesIO(image_bytes)) as img:
                if img.is_animated:
                    img.seek(0)  # Seek to the first frame of the GIF
                frame = img.convert('RGB')  # Convert to RGB mode
                np_arr = np.array(frame)  # Convert to NumPy array
                img = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        if img is None:
            # Use OpenCV for other image types
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            err = f"The uploaded file {file.filename} is not a valid image format or is corrupted."
            print(err)
            return {'result': [], 'msg': str(err)}

        height, width, _ = img.shape
        if width > 10000 or height > 10000:
            return {'result': [], 'msg': 'height or width out of range'}

        data = {"detector_backend": detector_backend, "recognition_model": recognition_model}
        embedding_objs = await predict(_represent, img)
        # embedding_objs = DeepFace.represent(
        #     img_path=img,
        #     detector_backend=detector_backend,
        #     model_name=recognition_model,
        #     enforce_detection=True,  # 强制检测，如果为true会报错, 设置为False时可以针对整张照片进行特征识别
        #     align=True,
        # )
        #enforce_detection=True 时，未识别到人脸的错误信息
        #1
        # Face could not be detected in numpy array.Please confirm that the picture is a face photo or consider to set enforce_detection param to False.
        #2
        # Exception while extracting faces from numpy array.Consider to set enforce_detection arg to False.
        del img
        data["result"] = embedding_objs
        # logging.info("detector_backend: %s", detector_backend)
        # logging.info("recognition_model: %s", recognition_model)
        logging.info("detected_img: %s", file.filename)
        logging.info("img_type: %s", content_type)
        logging.info("detected_persons: %d", len(embedding_objs))
        for embedding_obj in embedding_objs:
            logging.info("facial_area: %s", str(embedding_obj["facial_area"]))
            logging.info("face_confidence: %f", embedding_obj["face_confidence"])
        return data
    except Exception as e:
        if 'set enforce_detection' in str(e):
            return {'result': []}
        print(e)
        return {'result': [], 'msg': str(e)}

def _represent(img):
  faces = faceAnalysis.get(img)
  results = []
  for face in faces:
    resp_obj = {}
    embedding = face.normed_embedding.astype(float)
    resp_obj["embedding"] = embedding.tolist()
    # print(len(resp_obj["embedding"]))
    box = face.bbox
    resp_obj["facial_area"] = {"x" : int(box[0]), "y" : int(box[1]), "w" : int(box[2] - box[0]), "h" : int(box[3] - box[1])}
    resp_obj["face_confidence"] = face.det_score.astype(float) 
    results.append(resp_obj)
  return results

async def predict(predict_func, img):
    return await asyncio.get_running_loop().run_in_executor(None, predict_func, img)

def restart_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=http_port)

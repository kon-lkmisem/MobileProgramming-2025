import os
import time

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from torch.cuda.amp import autocast
from torch.backends import cudnn
import yaml
from models.models import Darknet
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import color_list
from utils.torch_utils import select_device, time_synchronized

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 1. 서버 및 모델 초기화
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="YOLOR-P6 Object Detection API (Optimized)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # 허용할 origin 목록
    allow_methods=["*"],              # 모든 HTTP 메소드 허용
    allow_headers=["*"],  
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.0 GPU/CPU 설정
# ─────────────────────────────────────────────────────────────────────────────
# 1.1 cuDNN 최적화
cudnn.benchmark = True

# 1.2 디바이스 선택 (GPU 있으면 0번 GPU, 없으면 CPU)
use_cuda  = torch.cuda.is_available()
device_id = "0" if use_cuda else ""
device    = select_device(device_id)

# 1.3 모델 정의 및 로드
model_cfg = "cfg/yolor_p6.cfg"
model     = Darknet(model_cfg, img_size=(640, 640))
weights   = "runs/train/newconn_model/weights/last.pt"

ckpt = torch.load(weights, map_location=device, weights_only=False)
ckpt = ckpt.get("model", ckpt)
model.load_state_dict(ckpt, strict=True)

# 1.4 FP16 변환
model.half()
model.to(device).eval()

# 1.5 클래스 이름 로드
# names_file = os.path.join(os.path.dirname(__file__), "data", "newconn.names")
# with open(names_file, "r") as f:
#     names = [x.strip() for x in f if x.strip()]
# print(names)
names_file = os.path.join(os.path.dirname(__file__), "data", "newconn.yaml")
with open(names_file, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
names = data.get("names", [])
discriptions = data.get("descriptions", [])
# print(names)
# print(discriptions)

# 1.6 전처리 설정
stride = 32
imgsz   = check_img_size(640, s=stride)

# ─────────────────────────────────────────────────────────────────────────────
# 1.7 엑셀 데이터 로드
# ─────────────────────────────────────────────────────────────────────────────
recycle_location_data = {}
def load_excel_data(path):
    data= {}
    df = pd.read_excel(path, sheet_name=None)
    for row in df["Sheet0"][1:].itertuples():
        # print(row)
        name = str(row[1])+"_"+str(row[2])+"_"+ str(row[3])+"_"+str(row[4])+"_"+str(row[5])
        data[name] = {
            "배출장소유형": row[6],
            "배출장소" : row[7],
            "생활쓰레기배출방법" : row[8],
            "음식물쓰레기배출방법" : row[9],
            "재활용품배출방법" : row[10],
            "일시적다량폐기물배출방법" : row[11],
            "일시적다량폐기물배출장소" : row[12],
            "생활쓰레기배출요일" : row[13],
            "음식물쓰레기배출요일" : row[14],
            "재활용품배출요일" : row[15],
            "생활쓰레기배출시작시각" : row[16],
            "생활쓰레기배출종료시각" : row[17],
            "음식물쓰레기배출시작시각" : row[18],
            "음식물쓰레기배출종료시각" : row[19],
            "재활용품배출시작시각" : row[20],
            "재활용품배출종료시각" : row[21],
            "일시적다량폐기물배출시작시" : row[22],
            "일시적다량폐기물배출종료시" : row[23],
            "미수거일" : row[24],
            "관리부서명" : row[25],
            "관리부서전화번호" : row[26],
            "데이터기준일자" : row[27],
        }
    print(len(data), "개 지역 정보가 로드되었습니다.")
    return data

def find_recycle_info(data,name):
    regions = name.split(" ")
    result = data
    depth = 0
    for region in regions:
        temp_result = {}
        for key in result.keys():
            if region in key:
                temp_result[key] = result[key]
        if not temp_result:
            if depth < 2:
                last_date = ""
                for key, value in temp_result.items():
                    if value["데이터기준일자"] > last_date:
                        last_date = value["데이터기준일자"]
                print(f"최신 데이터 기준일자: {last_date}")
                # last_date인 데이터만 선택
                temp_result = {k: v for k, v in temp_result.items() if v["데이터기준일자"] == last_date}
                return temp_result
            else:
                last_date = ""
                for key, value in result.items():
                    if value["데이터기준일자"] > last_date:
                        last_date = value["데이터기준일자"]
                print(f"최신 데이터 기준일자: {last_date}")
                # last_date인 데이터만 선택
                result = {k: v for k, v in result.items() if v["데이터기준일자"] == last_date}
                return result
        result = temp_result
        depth += 1

    last_date = ""
    for key, value in result.items():
        if value["데이터기준일자"] > last_date:
            last_date = value["데이터기준일자"]
    print(f"최신 데이터 기준일자: {last_date}")
    # last_date인 데이터만 선택
    result = {k: v for k, v in result.items() if v["데이터기준일자"] == last_date}
    
    return result

recycle_location_data = load_excel_data("12_04_04_E_생활쓰레기배출정보.xlsx")

# ─────────────────────────────────────────────────────────────────────────────
# 2. 앱 시작 시 워밍업
# ─────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def warmup():
    dummy = torch.zeros((1, 3, imgsz, imgsz), device=device).half()
    for _ in range(3):
        with torch.no_grad(), autocast():
            _ = model(dummy)

# ─────────────────────────────────────────────────────────────────────────────
# 3. 객체 검출 엔드포인트
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/detect/")
async def detect(image: UploadFile = File(...)):
    # 3.1 파일 형식 검사
    if image.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=415, detail="이미지 파일만 업로드 가능합니다.")

    # 3.2 이미지 디코딩
    contents = await image.read()
    nparr    = np.frombuffer(contents, np.uint8)
    img0     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img0 is None:
        raise HTTPException(status_code=400, detail="이미지 디코딩 실패")

    # 3.3 전처리
    img = cv2.resize(img0, (imgsz, imgsz))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    im_tensor = torch.from_numpy(img).to(device).half().unsqueeze(0)

    # 3.4 추론 (FP16 + autocast)
    with torch.no_grad(), autocast():
        t0 = time_synchronized()
        pred = model(im_tensor, augment=False)
        t1 = time_synchronized()

    if isinstance(pred, tuple):
        pred = pred[0]
    det = non_max_suppression(pred, 0.25, 0.45)[0]

    # 3.5 결과 후처리
    results = []
    if det is not None and det.shape[0]:
        det[:, :4] = scale_coords(im_tensor.shape[2:], det[:, :4], img0.shape).round()
        for *xyxy, conf, cls in det.tolist():
            cls   = int(cls)
            label = names[cls] if cls < len(names) else f"class_{cls}"
            results.append({
                "label":      label,
                "confidence": round(conf, 3),
                "bbox":       [int(x) for x in xyxy],
                "description": discriptions[cls] if cls < len(discriptions) else ""
            })

    return JSONResponse({
        "inference_time_s": round(t1 - t0, 3),
        "detections":       results
    })


# ─────────────────────────────────────────────────────────────────────────────
# 4. 재활용 위치 정보 엔드포인트
# ─────────────────────────────────────────────────────────────────────────────
# @app.post("/recycle-locations/")
@app.api_route(
    "/recycle-locations/",
    methods=["GET", "POST"],
    responses={
        200: {
            "description": "성공적으로 위치 정보를 반환",
            "content": {
                "application/json": {
                    "example": {
                        "result": [
                            {
                                "배출장소유형": "문전수거",
                                "배출장소": "집앞",
                                "생활쓰레기배출방법": "규격봉투에 넣어 지정된 요일에 배출",
                                "음식물쓰레기배출방법": "물기를 최대한 줄여 음식물만 전용봉투에 넣어 배출",
                                "재활용품배출방법": "투명한 비닐봉투에 담거나 끈으로 묶어서 지정된 요일에 배출",
                                "일시적다량폐기물배출방법": "주민센터방문 혹은 종로구홈페이지에 신청 후 배출",
                                "일시적다량폐기물배출장소": "집앞",
                                "생활쓰레기배출요일": "일+월+화+수+목+금",
                                "음식물쓰레기배출요일": "일+월+화+수+목+금",
                                "재활용품배출요일": "일+월+화+수+목+금",
                                "생활쓰레기배출시작시각": "19:00",
                                "생활쓰레기배출종료시각": "21:00",
                                "음식물쓰레기배출시작시각": "19:00",
                                "음식물쓰레기배출종료시각": "21:00",
                                "재활용품배출시작시각": "19:00",
                                "재활용품배출종료시각": "21:00",
                                "일시적다량폐기물배출시": "19:00",
                                "일시적다량폐기물배출종료시": "21:00",
                                "미수거일": "토+구정연휴+추석연휴",
                                "관리부서명": "청소행정과",
                                "관리부서전화번호": "02-2148-2373",
                                "데이터기준일자": "2020-07-10"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def recycle_locations(location: str):
    info = find_recycle_info(recycle_location_data, location)
    print(info.values())
    # nan 값이 있는 경우 빈 문자열로 대체
    for key, value in info.items():
        for k, v in value.items():
            if pd.isna(v):
                value[k] = ""
    return JSONResponse({"result":list(info.values())})

# ─────────────────────────────────────────────────────────────────────────────
# 4. 내장 Uvicorn 실행 (선택)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",    # 이 파일명을 main.py로 저장했다면
        host="0.0.0.0",
        port=8000,
        reload=True
    )
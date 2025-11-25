from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import os, traceback, re


# ------------------------------
# Gemini 設定
# ------------------------------
API_KEY = "AIzaSyDPDMo_nuEBL09UfdxjCelOhS-N-7bhdo8"
genai.configure(api_key=API_KEY)

os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "2"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------
# 圖片壓縮（一定要定義，不可刪）
# ------------------------------
def clean_image_bytes(data: bytes, max_size=(1024, 1024)):
    img = Image.open(BytesIO(data)).convert("RGB")
    img.thumbnail(max_size)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


# ------------------------------
# JSON 抽取（移除 ```json）
# ------------------------------
def extract_json(text: str) -> str:
    print("=== extract_json called ===")

    # 去掉所有 ``` 與 ```json
    text = text.replace("```json", "").replace("```", "").strip()

    # 找到第一個 '['
    start = text.find("[")
    # 找到最後一個 ']'
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return "[]"



# ------------------------------
# API 端點
# ------------------------------
@app.post("/generate_scene", response_class=PlainTextResponse)
async def generate_scene(
    camera1: UploadFile = File(...),
    camera2: UploadFile = File(...)
):
    try:
        img1 = clean_image_bytes(await camera1.read())
        img2 = clean_image_bytes(await camera2.read())

        prompt = """
請分析兩張室內圖片並輸出純 JSON 陣列。
不得包含任何文字或 markdown。
格式示例：
[
  {
    "id": "table_1",
    "type": "table",
    "position": {"x": 0, "y": 0.75, "z": -2},
    "scale": {"x": 2, "y": 0.8, "z": 1}
  }
]
"""

        model = genai.GenerativeModel("models/gemini-2.0-flash-lite")

        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img1}},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img2}},
                ],
            }
        ]

        result = model.generate_content(contents)
        raw = result.text or ""

        print("\n=== Gemini 原始回傳 ===")
        print(raw)

        scene_json = extract_json(raw)

        print("\n=== 抽出的 JSON ===")
        print(scene_json)

        return scene_json

    except Exception as e:
        traceback.print_exc()
        return "[]"


# ------------------------------
# 主程式
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    print("Scene Server 啟動：http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)

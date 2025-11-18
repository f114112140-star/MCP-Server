from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json, os, re, math, traceback, io, base64
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image
import numpy as np
import cv2

# === 基本設定 ===
API_KEY = "AIzaSyBiIoLvMW_PpNEzUulvjsmEmr6uxBGLOkE"
SAVE_PATH = "./scene_layout.json"
genai.configure(api_key=API_KEY)

# 關閉冗長日誌
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "2"

# === 啟動 FastAPI ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 圖片模糊與壓縮 ===
def blur_and_compress(data: bytes, max_size=(512, 512)):
    """模糊人物 + 壓縮圖片"""
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail(max_size)
        np_img = np.array(img)
        img_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        # HSV 膚色區域
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([30, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # 模糊膚色區域
        blurred = img_bgr.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 500:
                sub = blurred[y:y+h, x:x+w]
                sub = cv2.GaussianBlur(sub, (31, 31), 15)
                blurred[y:y+h, x:x+w] = sub

        img_blurred = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        img_blurred.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue()
    except Exception as e:
        print(f"圖片處理失敗: {e}")
        return data

# === JSON 工具 ===
def extract_json_from_text(text: str):
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        fixed = (
            match.group(0)
            .replace("```json", "")
            .replace("```", "")
            .replace("Infinity", "0")
            .replace("-Infinity", "0")
            .replace("NaN", "0")
        )
        try:
            return json.loads(fixed)
        except Exception as e:
            print("JSON 解碼失敗:", e)
    return {"scene": {"objects": []}}

def normalize_json_structure(data):
    if isinstance(data, dict) and "scene" in data and "objects" in data["scene"]:
        return data
    if isinstance(data, dict) and "objects" in data:
        return {"scene": {"objects": data["objects"]}}
    if isinstance(data, list):
        return {"scene": {"objects": data}}
    return {"scene": {"objects": []}}

def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    elif isinstance(obj, float):
        return 0.0 if math.isinf(obj) or math.isnan(obj) else obj
    elif isinstance(obj, str) and obj.strip().lower() in ["nan", "infinity", "-infinity", "inf", "-inf"]:
        return "0"
    elif obj is None:
        return 0
    return obj

# === 安全設定 ===
def get_safe_harm_category(names):
    for n in names:
        if hasattr(HarmCategory, n):
            return getattr(HarmCategory, n)
    return None

safety_settings = []
for cat in [
    get_safe_harm_category(["HARM_CATEGORY_HARASSMENT"]),
    get_safe_harm_category(["HARM_CATEGORY_HATE_SPEECH"]),
    get_safe_harm_category(["HARM_CATEGORY_DANGEROUS_CONTENT"]),
    get_safe_harm_category(["HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_SEXUAL"]),
    get_safe_harm_category(["HARM_CATEGORY_VIOLENCE_OR_GORE", "HARM_CATEGORY_VIOLENCE"]),
]:
    if cat:
        safety_settings.append({"category": cat, "threshold": HarmBlockThreshold.BLOCK_NONE})

# === Gemini 主邏輯 ===
@app.post("/generate")
async def generate_scene(camera1: UploadFile = File(...)):
    try:
        print("收到請求，開始 Gemini 單圖分析...")

        prompt = (
            "This is a harmless indoor laboratory photo. "
            "Ignore all humans and faces completely — only detect objects. "
            "Detect visible items: cabinet, chair, computer (or monitor), and table. "
            "Estimate their 3D positions in a 1000x1000 Unity scene, "
            "assuming camera (x=0,y=0,z=16). "
            "Return ONLY valid JSON: "
            "{\"scene\":{\"objects\":[{\"name\":\"chair_1\",\"position\":{\"x\":0,\"y\":0,\"z\":0},"
            "\"rotation\":{\"x\":0,\"y\":0,\"z\":0},\"scale\":{\"x\":1,\"y\":1,\"z\":1}]}}. "
            "No extra text, only pure JSON output."
        )

        img_data = await camera1.read()
        print(f"原始圖片大小: {len(img_data)/1_000_000:.2f} MB")

        # 模糊 + 壓縮
        img_processed = blur_and_compress(img_data)
        print(f"圖片模糊處理完成 ({len(img_processed)/1_000_000:.2f} MB)")

        # base64 編碼
        img_b64 = base64.b64encode(img_processed).decode("utf-8")

        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                ],
            }
        ]

        # 呼叫 Gemini 主模型
        print("呼叫 Gemini API (主模型)...")
        try:
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            result = model.generate_content(
                contents,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.6,
                    "max_output_tokens": 512,
                },
                safety_settings=safety_settings,
                request_options={"timeout": 60.0},
            )
        except Exception as e:
            print("主模型失敗，切換至 flash-lite:", e)
            model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
            result = model.generate_content(
                contents,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.6,
                    "max_output_tokens": 512,
                },
                safety_settings=safety_settings,
                request_options={"timeout": 60.0},
            )

        # === 處理回覆 ===
        raw_text = ""
        try:
            if hasattr(result, "candidates") and result.candidates:
                cand = result.candidates[0]
                if cand.finish_reason == 0 and cand.content.parts:
                    raw_text = cand.content.parts[0].text
                else:
                    print(f"Gemini 結束原因：{cand.finish_reason}")
            else:
                print("Gemini 沒有候選結果（可能被安全過濾）")
        except Exception as e:
            print(f"解析回覆失敗: {e}")

        if hasattr(result, "prompt_feedback"):
            print("Prompt Feedback:", result.prompt_feedback)

        if not raw_text:
            print("無輸出，回傳空 JSON")
            return JSONResponse(content={"scene": {"objects": []}})

        print("Gemini 原始回傳（前300字）:", raw_text[:300])
        data = sanitize_json(normalize_json_structure(extract_json_from_text(raw_text)))

        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"已儲存乾淨 JSON: {SAVE_PATH}")
        return JSONResponse(content=data)

    except Exception as e:
        print("發生錯誤：", e)
        traceback.print_exc()
        return JSONResponse(content={"scene": {"objects": []}, "error": str(e)})

# === 啟動伺服器 ===
if __name__ == "__main__":
    import uvicorn
    print("啟動 Gemini 防誤判伺服器：http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)

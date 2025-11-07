<<<<<<< HEAD
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json, os, re, math, traceback, io
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image

# ===  基本設定 ===
API_KEY = "AIzaSyBiIoLvMW_PpNEzUulvjsmEmr6uxBGLOkE"  # ← 改成你的 Gemini API key
SAVE_PATH = "./scene_layout.json"
genai.configure(api_key=API_KEY)

#  關閉冗長 gRPC 日誌
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

# ===  圖片壓縮工具 ===
def compress_image(data: bytes, max_size=(512, 512)):
    """縮小圖片以避免 Gemini 拒收 (防 503 Illegal metadata)"""
    try:
        img = Image.open(io.BytesIO(data))
        img.thumbnail(max_size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(f" 圖片壓縮失敗: {e}")
        return data

# === JSON 工具 ===
def extract_json_from_text(text: str):
    """從 Gemini 回傳文字中抽出 JSON"""
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        raw_json = match.group(0)
        fixed = (
            raw_json.replace("```json", "")
            .replace("```", "")
            .replace("Infinity", "0")
            .replace("-Infinity", "0")
            .replace("NaN", "0")
        )
        try:
            return json.loads(fixed)
        except Exception as e:
            print(" JSON 解碼失敗:", e)
            return {"scene": {"objects": []}}
    return {"scene": {"objects": []}}

def normalize_json_structure(data):
    """確保 JSON 結構符合 Unity"""
    if isinstance(data, dict) and "scene" in data and "objects" in data["scene"]:
        return data
    if isinstance(data, dict) and "objects" in data:
        return {"scene": {"objects": data["objects"]}}
    if isinstance(data, list):
        return {"scene": {"objects": data}}
    return {"scene": {"objects": []}}

def sanitize_json(obj):
    """遞迴清理 Infinity / NaN / None"""
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return 0.0
        return obj
    elif isinstance(obj, str):
        if obj.strip().lower() in ["nan", "infinity", "-infinity", "inf", "-inf"]:
            return "0"
        return obj
    elif obj is None:
        return 0
    else:
        return obj

# ===  相容版安全設定 ===
def get_safe_harm_category(name_candidates):
    """依序嘗試取得 HarmCategory 的有效屬性"""
    for name in name_candidates:
        if hasattr(HarmCategory, name):
            return getattr(HarmCategory, name)
    return None

harassment = get_safe_harm_category(["HARM_CATEGORY_HARASSMENT"])
hate = get_safe_harm_category(["HARM_CATEGORY_HATE_SPEECH"])
danger = get_safe_harm_category(["HARM_CATEGORY_DANGEROUS_CONTENT"])
sexual = get_safe_harm_category(["HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_SEXUAL"])
violence = get_safe_harm_category(["HARM_CATEGORY_VIOLENCE_OR_GORE", "HARM_CATEGORY_VIOLENCE"])

safety_settings = []
for cat in [harassment, hate, danger, sexual, violence]:
    if cat is not None:
        safety_settings.append({"category": cat, "threshold": HarmBlockThreshold.BLOCK_NONE})

# ===  Gemini 主邏輯 ===
@app.post("/generate")
async def generate_scene(camera1: UploadFile = File(...)):
    try:
        print(" 收到請求，開始 Gemini 單圖分析...")

        #  Prompt（強調圖片安全）
        prompt = (
            "You are analyzing a normal indoor photo of a room (no people, no violence, no sensitive or sexual content). "
            "Identify and list only visible furniture objects: cabinet, chair, computer, and table. "
            "For each detected object, estimate its 3D position in a 1000x1000 Unity world, "
            "assuming the camera is at (x=0,y=0,z=16). "
            "Return ONLY JSON, in this format: "
            "{\"scene\":{\"objects\":[{\"name\":\"chair_1\",\"position\":{\"x\":0,\"y\":0,\"z\":0},"
            "\"rotation\":{\"x\":0,\"y\":0,\"z\":0},\"scale\":{\"x\":1,\"y\":1,\"z\":1}]}}. "
            "Do not include text explanations or descriptions."
        )

        contents = [prompt]

        # ===  處理圖片 ===
        img_data = await camera1.read()
        size_mb = len(img_data) / 1_000_000
        print(f" 原始圖片大小: {size_mb:.2f} MB")

        img_compressed = compress_image(img_data)
        size_after = len(img_compressed) / 1_000_000
        print(f" 壓縮後大小: {size_after:.2f} MB")

        contents.append({"mime_type": "image/png", "data": img_compressed})

        # ===  呼叫 Gemini ===
        print(" 呼叫 Gemini API 中... 這步可能花 5~15 秒")
        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")

        result = model.generate_content(
            contents=contents,
            generation_config={
                "temperature": 0.4,
                "top_p": 0.8,
                "max_output_tokens": 2048
            },
            safety_settings=safety_settings,
            request_options={"timeout": 120.0}
        )

        # ===  安全擷取回覆 ===
        raw_text = ""
        try:
            if hasattr(result, "candidates") and len(result.candidates) > 0:
                cand = result.candidates[0]
                if cand.finish_reason == 0 and cand.content.parts:
                    raw_text = cand.content.parts[0].text
                else:
                    print(f" Gemini 結束原因：{cand.finish_reason}（可能被過濾或提前中止）")
            else:
                print(" Gemini 沒有候選結果（可能被安全過濾）")
        except Exception as e:
            print(f" 解析回覆失敗：{e}")
            raw_text = ""

        # 顯示過濾原因（若有）
        if hasattr(result, "prompt_feedback"):
            print(" Prompt Feedback:", result.prompt_feedback)

        # 若沒有輸出文字，仍回傳空 JSON
        if not raw_text:
            print(" Gemini 沒有回傳文字內容，回傳空場景 JSON")
            return JSONResponse(content={"scene": {"objects": []}})

        print(" Gemini 原始回傳（前300字）:", raw_text[:300])

        # ===  JSON 清理流程 ===
        data = extract_json_from_text(raw_text)
        data = normalize_json_structure(data)
        data = sanitize_json(data)

        # ===  儲存結果 ===
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f" 已儲存乾淨 JSON: {SAVE_PATH}")

        return JSONResponse(content=data)

    except Exception as e:
        print(" 發生錯誤：", e)
        traceback.print_exc()
        return JSONResponse(content={"scene": {"objects": []}, "error": str(e)})

# ===  啟動伺服器 ===
if __name__ == "__main__":
    import uvicorn
    print(" 啟動 Gemini 單圖伺服器：http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
=======
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json, os, re, math, traceback, io
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image

# ===  基本設定 ===
API_KEY = "AIzaSyBiIoLvMW_PpNEzUulvjsmEmr6uxBGLOkE"  # ← 改成你的 Gemini API key
SAVE_PATH = "./scene_layout.json"
genai.configure(api_key=API_KEY)

#  關閉冗長 gRPC 日誌
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

# ===  圖片壓縮工具 ===
def compress_image(data: bytes, max_size=(512, 512)):
    """縮小圖片以避免 Gemini 拒收 (防 503 Illegal metadata)"""
    try:
        img = Image.open(io.BytesIO(data))
        img.thumbnail(max_size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(f" 圖片壓縮失敗: {e}")
        return data

# === JSON 工具 ===
def extract_json_from_text(text: str):
    """從 Gemini 回傳文字中抽出 JSON"""
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        raw_json = match.group(0)
        fixed = (
            raw_json.replace("```json", "")
            .replace("```", "")
            .replace("Infinity", "0")
            .replace("-Infinity", "0")
            .replace("NaN", "0")
        )
        try:
            return json.loads(fixed)
        except Exception as e:
            print(" JSON 解碼失敗:", e)
            return {"scene": {"objects": []}}
    return {"scene": {"objects": []}}

def normalize_json_structure(data):
    """確保 JSON 結構符合 Unity"""
    if isinstance(data, dict) and "scene" in data and "objects" in data["scene"]:
        return data
    if isinstance(data, dict) and "objects" in data:
        return {"scene": {"objects": data["objects"]}}
    if isinstance(data, list):
        return {"scene": {"objects": data}}
    return {"scene": {"objects": []}}

def sanitize_json(obj):
    """遞迴清理 Infinity / NaN / None"""
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return 0.0
        return obj
    elif isinstance(obj, str):
        if obj.strip().lower() in ["nan", "infinity", "-infinity", "inf", "-inf"]:
            return "0"
        return obj
    elif obj is None:
        return 0
    else:
        return obj

# ===  相容版安全設定 ===
def get_safe_harm_category(name_candidates):
    """依序嘗試取得 HarmCategory 的有效屬性"""
    for name in name_candidates:
        if hasattr(HarmCategory, name):
            return getattr(HarmCategory, name)
    return None

harassment = get_safe_harm_category(["HARM_CATEGORY_HARASSMENT"])
hate = get_safe_harm_category(["HARM_CATEGORY_HATE_SPEECH"])
danger = get_safe_harm_category(["HARM_CATEGORY_DANGEROUS_CONTENT"])
sexual = get_safe_harm_category(["HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_SEXUAL"])
violence = get_safe_harm_category(["HARM_CATEGORY_VIOLENCE_OR_GORE", "HARM_CATEGORY_VIOLENCE"])

safety_settings = []
for cat in [harassment, hate, danger, sexual, violence]:
    if cat is not None:
        safety_settings.append({"category": cat, "threshold": HarmBlockThreshold.BLOCK_NONE})

# ===  Gemini 主邏輯 ===
@app.post("/generate")
async def generate_scene(camera1: UploadFile = File(...)):
    try:
        print(" 收到請求，開始 Gemini 單圖分析...")

        #  Prompt（強調圖片安全）
        prompt = (
            "You are analyzing a normal indoor photo of a room (no people, no violence, no sensitive or sexual content). "
            "Identify and list only visible furniture objects: cabinet, chair, computer, and table. "
            "For each detected object, estimate its 3D position in a 1000x1000 Unity world, "
            "assuming the camera is at (x=0,y=0,z=16). "
            "Return ONLY JSON, in this format: "
            "{\"scene\":{\"objects\":[{\"name\":\"chair_1\",\"position\":{\"x\":0,\"y\":0,\"z\":0},"
            "\"rotation\":{\"x\":0,\"y\":0,\"z\":0},\"scale\":{\"x\":1,\"y\":1,\"z\":1}]}}. "
            "Do not include text explanations or descriptions."
        )

        contents = [prompt]

        # ===  處理圖片 ===
        img_data = await camera1.read()
        size_mb = len(img_data) / 1_000_000
        print(f" 原始圖片大小: {size_mb:.2f} MB")

        img_compressed = compress_image(img_data)
        size_after = len(img_compressed) / 1_000_000
        print(f" 壓縮後大小: {size_after:.2f} MB")

        contents.append({"mime_type": "image/png", "data": img_compressed})

        # ===  呼叫 Gemini ===
        print(" 呼叫 Gemini API 中... 這步可能花 5~15 秒")
        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")

        result = model.generate_content(
            contents=contents,
            generation_config={
                "temperature": 0.4,
                "top_p": 0.8,
                "max_output_tokens": 2048
            },
            safety_settings=safety_settings,
            request_options={"timeout": 120.0}
        )

        # ===  安全擷取回覆 ===
        raw_text = ""
        try:
            if hasattr(result, "candidates") and len(result.candidates) > 0:
                cand = result.candidates[0]
                if cand.finish_reason == 0 and cand.content.parts:
                    raw_text = cand.content.parts[0].text
                else:
                    print(f" Gemini 結束原因：{cand.finish_reason}（可能被過濾或提前中止）")
            else:
                print(" Gemini 沒有候選結果（可能被安全過濾）")
        except Exception as e:
            print(f" 解析回覆失敗：{e}")
            raw_text = ""

        # 顯示過濾原因（若有）
        if hasattr(result, "prompt_feedback"):
            print(" Prompt Feedback:", result.prompt_feedback)

        # 若沒有輸出文字，仍回傳空 JSON
        if not raw_text:
            print(" Gemini 沒有回傳文字內容，回傳空場景 JSON")
            return JSONResponse(content={"scene": {"objects": []}})

        print(" Gemini 原始回傳（前300字）:", raw_text[:300])

        # ===  JSON 清理流程 ===
        data = extract_json_from_text(raw_text)
        data = normalize_json_structure(data)
        data = sanitize_json(data)

        # ===  儲存結果 ===
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f" 已儲存乾淨 JSON: {SAVE_PATH}")

        return JSONResponse(content=data)

    except Exception as e:
        print(" 發生錯誤：", e)
        traceback.print_exc()
        return JSONResponse(content={"scene": {"objects": []}, "error": str(e)})

# ===  啟動伺服器 ===
if __name__ == "__main__":
    import uvicorn
    print(" 啟動 Gemini 單圖伺服器：http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)


>>>>>>> aae669d8191007526546fddb6ac5105c96cbf258

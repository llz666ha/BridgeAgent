from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uuid,uvicorn

app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有域名访问，生产环境建议指定具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有HTTP头
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    with open(save_path, "wb") as f:
        f.write(await file.read())

    return {
        "file_id": file_id,
        "filename": file.filename,
        "path": str(save_path.absolute()),  # 返回绝对路径
        "content_type": file.content_type,
    }
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8006)
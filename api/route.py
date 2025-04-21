import base64
import io
import numpy as np
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from PIL import Image, ImageChops
from pydantic import BaseModel

router = APIRouter()

# 加载模型参数（路径需要根据Vercel环境调整）
try:
    model_weights = np.load('model/model_weights.npy', allow_pickle=True).item()
    W, b = model_weights['W'], model_weights['b']
except Exception as e:
    print(f"Error loading model: {str(e)}")
    W, b = None, None

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

class ImageRequest(BaseModel):
    image: str

@router.post("/predict")
async def predict(data: ImageRequest = Body(...)):
    if W is None or b is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 提取并解码Base64
        image_base64 = data.image.split(",")[1]
        image_bytes = base64.b64decode(image_base64)
        
        # 图像预处理
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((28, 28), Image.LANCZOS)
        
        if image.mode == 'RGBA':
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        
        image = image.convert("L")
        image = ImageChops.invert(image)
        pixels = np.array(image, dtype=np.float32).reshape(1, 784) / 255.0
        
        # 预测
        logits = np.dot(pixels, W) + b
        probabilities = softmax(logits)[0]
        
        return {
            "prediction": int(np.argmax(probabilities)),
            "probabilities": probabilities.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Vercel需要导出的app实例
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 配置CORS
origins = [
    "https://www.202718.xyz",
    "https://next.202718.xyz",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host='0.0.0.0',port=8000)
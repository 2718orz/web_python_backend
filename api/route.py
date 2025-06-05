from fastapi import FastAPI,Body
from fastapi.exceptions import RequestErrorModel
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .deep_convnet import DeepConvNet
import base64
from PIL import Image,ImageChops
from io import BytesIO
import numpy as np
from .functions import softmax

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.202718.xyz"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class PredictionRequest(BaseModel):
    image: str  # Base64 encoded image string

network = DeepConvNet()
network.load_params("api/deep_convnet_params.pkl")
@app.post("/api/predict")
def predict(data:PredictionRequest = Body(...)):
    try:
        image_base64 = data.image.split(",")[1]
        image_bytes = base64.b64decode(image_base64)

        image = Image.open(BytesIO(image_bytes))
        image = image.resize((28, 28), Image.LANCZOS)
        
        if image.mode == 'RGBA':
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        image = image.convert("L")
        image = ImageChops.invert(image)
        image.save("temp_image.png")  # Save for debugging if needed
        pixels = np.array(image, dtype=np.float32).reshape(1, 1, 28, 28) / 255.0
    
    # Perform prediction
        prediction = network.predict(pixels, train_flg=False)
        y = int(np.argmax(prediction[0]))
        return {"prediction": y,"probabilities": softmax(prediction[0]).tolist()}
    except Exception as e:
        raise RequestErrorModel(detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

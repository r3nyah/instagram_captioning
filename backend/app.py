from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model_loader import load_ai_resources
from torchvision import transforms
from PIL import Image
import io
import torch

app = FastAPI()

# Enable connection from React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model ONCE at startup
MODEL_PATH = "../model_epoch_50.pth" # Adjust path if you moved the file
CSV_PATH = "../data/processed/train.csv"
model, vocab = load_ai_resources(MODEL_PATH, CSV_PATH)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. Read Image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # 2. Process
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # 3. Generate
        tokens = model.caption_image(img_tensor, vocab)
        
        # 4. Clean Text
        caption = " ".join(tokens)
        for trash in ["<SOS>", "<EOS>", "<UNK>", "<PAD>"]:
            caption = caption.replace(trash, "")
            
        return {"status": "success", "caption": caption.strip()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
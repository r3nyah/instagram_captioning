import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import random
import argparse

# --- CONFIGURATION (Must match train.py) ---
EMBED_SIZE = 256
HIDDEN_SIZE = 512
VOCAB_THRESHOLD = 5
DATA_DIR = "../data/processed"
IMG_DIR = "../data/processed/images"
MODEL_PATH = "../model_epoch_50.pth" # Load the final epoch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------------------------

# --- 1. RE-DEFINE CLASSES (Must match train.py exactly) ---
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self): return len(self.itos)

    def build_vocabulary(self, sentence_list):
        freqs = {}
        idx = 4
        for sentence in sentence_list:
            for word in sentence.split():
                freqs[word] = freqs.get(word, 0) + 1
                if freqs[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

class CaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(CaptioningModel, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions):
        # We don't need forward for inference, we generate manually
        pass
    
    def caption_image(self, image, vocabulary, max_length=20):
        result_caption = []
        
        with torch.no_grad():
            x = self.resnet(image).reshape(image.shape[0], -1)
            x = self.embed(x).unsqueeze(1)
            
            states = None
            
            for _ in range(max_length):
                hiddens, states = self.lstm(x, states)
                output = self.linear(hiddens.squeeze(1))
                predicted = output.argmax(1)
                
                # Append predicted word
                word_idx = predicted.item()
                result_caption.append(word_idx)
                
                # Stop if <EOS>
                if vocabulary.itos[word_idx] == "<EOS>":
                    break
                
                # Feed predicted word as next input
                x = self.embedding(predicted).unsqueeze(1)
                
        return [vocabulary.itos[idx] for idx in result_caption]

# --- 2. SETUP & RUN ---
def setup_model():
    print("⏳ Rebuilding Vocabulary from training data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    vocab = Vocabulary(VOCAB_THRESHOLD)
    vocab.build_vocabulary(train_df['caption'].tolist())
    print(f"✅ Vocab size: {len(vocab)} words")

    print(f"⏳ Loading Model from {MODEL_PATH}...")
    model = CaptioningModel(EMBED_SIZE, HIDDEN_SIZE, len(vocab)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model, vocab

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)

def main(custom_image=None):
    model, vocab = setup_model()
    
    # Select Image
    if custom_image:
        img_path = custom_image
        if not os.path.exists(img_path):
            print(f"❌ Error: File {img_path} not found.")
            return
    else:
        # Pick random from Validation set
        val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
        random_row = val_df.sample(1).iloc[0]
        img_filename = random_row['image']
        img_path = os.path.join(IMG_DIR, img_filename)
        print(f"🎲 Selected Random Validation Image: {img_filename}")
        print(f"📝 Actual Caption: {random_row['caption']}")

    # Generate
    print(f"🖼️  Analyzing image...")
    image_tensor = process_image(img_path)
    caption_tokens = model.caption_image(image_tensor, vocab)
    
    # Cleanup text
    sentence = " ".join(caption_tokens)
    sentence = sentence.replace("<SOS>", "").replace("<EOS>", "").replace("<UNK>", "")
    
    print("\n" + "="*40)
    print(f"🤖 AI Generated: {sentence.strip()}")
    print("="*40 + "\n")

    # Optional: Show image (requires matplotlib, uncomment if you want)
    # import matplotlib.pyplot as plt
    # plt.imshow(Image.open(img_path))
    # plt.title(sentence.strip())
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to a custom image file")
    args = parser.parse_args()
    
    main(args.image)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

# --- CONFIGURATION ---
EPOCHS = 50        
BATCH_SIZE = 32       
LEARNING_RATE = 1e-4
EMBED_SIZE = 256
HIDDEN_SIZE = 512
VOCAB_THRESHOLD = 5   
DATA_DIR = "../data/processed"
IMG_DIR = "../data/processed/images"
# ---------------------

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
    
    def numericalize(self, text):
        return [1] + [self.stoi.get(w, 3) for w in text.split()] + [2]

class InstagramDataset(Dataset):
    def __init__(self, csv_file, vocab):
        self.df = pd.read_csv(csv_file)
        self.vocab = vocab
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(IMG_DIR, row['image'])).convert("RGB")
        target = torch.tensor(self.vocab.numericalize(row['caption']))
        return self.transform(image), target

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
        features = self.embed(self.resnet(images).reshape(images.size(0), -1))
        embeddings = self.embedding(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        outputs, _ = self.lstm(inputs)
        return self.linear(outputs)

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.zeros(len(captions), max(len(c) for c in captions)).long()
    for i, cap in enumerate(captions): targets[i, :len(cap)] = cap
    return images, targets

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Training on: {device}")

    # Load Data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    vocab = Vocabulary(VOCAB_THRESHOLD)
    vocab.build_vocabulary(train_df['caption'].tolist())
    train_loader = DataLoader(InstagramDataset(os.path.join(DATA_DIR, "train.csv"), vocab), 
                              batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = CaptioningModel(EMBED_SIZE, HIDDEN_SIZE, len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- EPOCH LOOP ---
    model.train()
    for epoch in range(EPOCHS):
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for imgs, captions in loop:
            imgs, captions = imgs.to(device), captions.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs, captions)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item())

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"../model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
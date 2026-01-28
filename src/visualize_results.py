import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu
from collections import Counter
from tqdm import tqdm
import random

# --- CONFIGURATION ---
EMBED_SIZE = 256
HIDDEN_SIZE = 512
VOCAB_THRESHOLD = 5
DATA_DIR = "../data/processed"
IMG_DIR = "../data/processed/images"
MODEL_PATH = "../model_epoch_50.pth"
RESULT_DIR = "../result"
NUM_TEST_SAMPLES = 15000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------

# --- SETUP FOLDERS ---
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
    print(f"📂 Created result folder at: {RESULT_DIR}")

# --- MODEL DEFINITIONS (Must match training) ---
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
                word_idx = predicted.item()
                result_caption.append(word_idx)
                if vocabulary.itos[word_idx] == "<EOS>": break
                x = self.embedding(predicted).unsqueeze(1)
        return [vocabulary.itos[idx] for idx in result_caption]

# --- UTILS ---
def clean_sentence(sent_list):
    """Removes special tokens for cleaner analysis"""
    ignore = ["<SOS>", "<EOS>", "<UNK>", "<PAD>"]
    return [word for word in sent_list if word not in ignore]

def process_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)

# --- MAIN ANALYSIS SCRIPT ---
def analyze_model():
    print("⏳ Loading Resources...")
    
    # Load Vocab
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    vocab = Vocabulary(VOCAB_THRESHOLD)
    vocab.build_vocabulary(train_df['caption'].tolist())
    
    # Load Model
    model = CaptioningModel(EMBED_SIZE, HIDDEN_SIZE, len(vocab)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load Validation Data
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    
    # Sample Data
    sample_df = val_df.sample(n=min(NUM_TEST_SAMPLES, len(val_df)), random_state=42)
    
    bleu_scores = []
    generated_words = []
    actual_words = []
    
    print(f"🚀 Running Inference on {len(sample_df)} images to calculate metrics...")
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        img_path = os.path.join(IMG_DIR, row['image'])
        actual_caption = str(row['caption']).split()
        
        # Run AI
        if os.path.exists(img_path):
            img_tensor = process_image(img_path)
            output = model.caption_image(img_tensor, vocab)
            
            # Clean data
            pred_clean = clean_sentence(output)
            actual_clean = actual_caption # Already split list
            
            # 1. Calculate BLEU Score (Accuracy metric)
            # Weights (0.5, 0.5) checks for 1-gram and 2-gram overlap
            score = sentence_bleu([actual_clean], pred_clean, weights=(0.5, 0.5, 0, 0))
            bleu_scores.append(score)
            
            # 2. Collect words for frequency analysis
            generated_words.extend(pred_clean)
            actual_words.extend(actual_clean)
            
    # --- PLOTTING RESULTS ---
    print("📊 Generating plots...")
    sns.set_style("whitegrid")
    
    # Plot 1: BLEU Score Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(bleu_scores, bins=20, kde=True, color='purple')
    plt.title(f"Model Performance Distribution (BLEU Scores)\nMean Score: {sum(bleu_scores)/len(bleu_scores):.4f}")
    plt.xlabel("BLEU Score (Higher is Better)")
    plt.ylabel("Number of Images")
    plt.savefig(f"{RESULT_DIR}/bleu_score_distribution.png")
    print(f"✅ Saved plot: {RESULT_DIR}/bleu_score_distribution.png")
    
    # Plot 2: Top 15 Words Comparison
    gen_counts = Counter(generated_words).most_common(15)
    
    words, counts = zip(*gen_counts)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(counts), y=list(words), palette="viridis")
    plt.title("Top 15 Most Used Words by AI")
    plt.xlabel("Frequency")
    plt.savefig(f"{RESULT_DIR}/word_frequency_analysis.png")
    print(f"✅ Saved plot: {RESULT_DIR}/word_frequency_analysis.png")

    print("\n✅ Analysis Complete! Check the 'result' folder.")

if __name__ == "__main__":
    analyze_model()
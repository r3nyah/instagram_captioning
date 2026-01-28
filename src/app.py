import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="InstaCap AI",
    page_icon="📸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS (The Magic Sauce) ---
# This injects CSS to hide Streamlit branding and add "App-like" styling
st.markdown("""
<style>
    /* Dark Mode Background */
    .stApp {
        background-color: #0f172a;
        color: white;
    }
    
    /* Hide Streamlit Header/Footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom Card Styling */
    .css-1r6slb0, .stFileUploader {
        background-color: #1e293b;
        border-radius: 1.5rem;
        padding: 2rem;
        border: 1px solid #334155;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(to right, #db2777, #7c3aed);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 0.75rem;
        font-weight: bold;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(219, 39, 119, 0.5);
    }
    
    /* Success Box */
    .success-box {
        background-color: rgba(34, 197, 94, 0.1);
        border: 1px solid #22c55e;
        border-radius: 1rem;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1.5rem;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
EMBED_SIZE = 256
HIDDEN_SIZE = 512
VOCAB_THRESHOLD = 5
DATA_DIR = "../data/processed"
MODEL_PATH = "../model_epoch_50.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL CLASSES (Hidden for cleanliness) ---
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

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    vocab_path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(vocab_path): return None, None
    train_df = pd.read_csv(vocab_path)
    vocab = Vocabulary(VOCAB_THRESHOLD)
    vocab.build_vocabulary(train_df['caption'].tolist())
    
    if not os.path.exists(MODEL_PATH): return None, None
    model = CaptioningModel(EMBED_SIZE, HIDDEN_SIZE, len(vocab)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, vocab

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

# --- APP LOGIC ---
def main():
    # 1. Fake Startup Animation (Runs once per session)
    if "startup_done" not in st.session_state:
        placeholder = st.empty()
        with placeholder.container():
            st.markdown("""
            <div style='height: 80vh; display: flex; flex-direction: column; justify-content: center; align-items: center;'>
                <h1 style='font-size: 4rem; margin-bottom: 0;'>📸</h1>
                <h1 style='background: -webkit-linear-gradient(#db2777, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem;'>INSTACAP AI</h1>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1.5) # Fake loading delay
        placeholder.empty()
        st.session_state["startup_done"] = True

    # 2. Main UI
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Instagram Caption Generator</h2>", unsafe_allow_html=True)

    model, vocab = load_resources()
    if not model:
        st.error("❌ Critical Error: Model files not found. Please run training first.")
        return

    # Upload Section
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # Centered Image Preview
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, use_container_width=True, output_format="JPEG")
            
            # Generate Button
            if st.button("✨ GENERATE CAPTION"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Fake Processing Steps
                status_text.text("🧠 Reading image...")
                progress_bar.progress(30)
                time.sleep(0.3)
                
                status_text.text("⚡ Running Neural Network...")
                progress_bar.progress(60)
                
                # Real Inference
                img_tensor = process_image(image)
                caption_tokens = model.caption_image(img_tensor, vocab)
                sentence = " ".join(caption_tokens).replace("<SOS>", "").replace("<EOS>", "").replace("<UNK>", "").strip()
                
                progress_bar.progress(100)
                time.sleep(0.2)
                progress_bar.empty()
                status_text.empty()
                
                # Success Animation UI
                st.markdown(f"""
                <div class="success-box">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">✅ Caption Ready</div>
                    <div style="font-size: 1.2rem; font-style: italic; color: #e2e8f0;">"{sentence}"</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 1rem;">Click text to copy (manual copy required on web)</div>
                </div>
                """, unsafe_allow_html=True)
                st.balloons() # Confetti effect!

    else:
        # Placeholder when no image
        st.markdown("""
        <div style='text-align: center; color: #64748b; padding: 2rem;'>
            <p>👆 Tap above to upload your photo</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
import os

# CONFIG
EMBED_SIZE = 256
HIDDEN_SIZE = 512
VOCAB_THRESHOLD = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def load_ai_resources(model_path, csv_path):
    print("⏳ Loading Vocab...")
    train_df = pd.read_csv(csv_path)
    vocab = Vocabulary(VOCAB_THRESHOLD)
    vocab.build_vocabulary(train_df['caption'].tolist())

    print("⏳ Loading Model...")
    model = CaptioningModel(EMBED_SIZE, HIDDEN_SIZE, len(vocab)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model, vocab
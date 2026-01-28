import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from cleantext import clean
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIGURATION ---
KAGGLE_DATASET = "prithvijaunjale/instagram-images-with-captions"
RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"
IMG_SIZE = (256, 256)
MIN_CAPTION_LENGTH = 3 
# ---------------------

def check_gpu():
    print("--- GPU DIAGNOSTIC ---")
    if torch.cuda.is_available():
        print(f"✅ NVIDIA GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU detected. Training will be slow.")
    print("----------------------\n")

def find_file(filename, search_path):
    """Recursively searches for a specific file."""
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

def build_image_map(search_path):
    """
    Scans ALL subfolders to find every image file.
    Returns a dictionary: { 'filename.jpg': '/full/path/to/filename.jpg' }
    """
    print(f"🗺️  Scanning '{search_path}' to build Image Map...")
    image_map = {}
    count = 0
    for root, dirs, files in os.walk(search_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # We store just the filename as the key
                image_map[file] = os.path.join(root, file)
                count += 1
    
    print(f"✅ Found {count} images scattered across subfolders.")
    return image_map

def download_data():
    csv_path = find_file("captions_csv.csv", RAW_DIR)
    if csv_path:
        print(f"✅ Raw data found at: {csv_path}")
    else:
        print(f"⬇️ Downloading dataset...")
        os.system(f"kaggle datasets download -d {KAGGLE_DATASET} -p {RAW_DIR} --unzip")

def clean_caption_text(text):
    if pd.isna(text): return ""
    return clean(text, fix_unicode=True, to_ascii=True, lower=True, 
                 no_urls=True, no_emails=True, no_phone_numbers=True, 
                 no_punct=True, lang="en")

def process_images_and_captions():
    os.makedirs(f"{PROCESSED_DIR}/images", exist_ok=True)
    
    # 1. Load CSV
    csv_path = find_file("captions_csv.csv", RAW_DIR)
    if not csv_path:
        print("❌ Error: Could not find 'captions_csv.csv'. Check download.")
        return

    df = pd.read_csv(csv_path)
    print(f"📄 Loaded CSV with {len(df)} rows.")

    # 2. Build the Global Image Map (The Fix!)
    image_map = build_image_map(RAW_DIR)
    
    if len(image_map) == 0:
        print("❌ CRITICAL ERROR: No images found in data/raw. Check your unzip.")
        return

    # 3. Process
    valid_data = []
    print("🚀 Processing images (Matching CSV to Image Map)...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Get filename from CSV (e.g., 'img/insta1' or just 'insta1')
        raw_name_in_csv = str(row['Image File'])
        base_filename = os.path.basename(raw_name_in_csv)
        
        # Ensure extension exists
        if not base_filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            base_filename = base_filename + ".jpg"

        # LOOKUP: Check if this file exists in our map
        if base_filename not in image_map:
            # Skip if image is missing from disk
            continue

        # Clean Caption
        clean_cap = clean_caption_text(str(row['Caption']))
        if len(clean_cap.split()) < MIN_CAPTION_LENGTH: 
            continue
            
        # Get full path from map
        img_path = image_map[base_filename]

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
                new_filename = f"{idx}.jpg"
                img.save(os.path.join(PROCESSED_DIR, "images", new_filename), "JPEG", quality=85)
                valid_data.append({'image': new_filename, 'caption': clean_cap})
        except Exception:
            continue

    if not valid_data:
        print("❌ No images processed. Complete mismatch between CSV and files.")
        return

    # 4. Save
    new_df = pd.DataFrame(valid_data)
    print(f"✨ Success! Processed {len(new_df)} valid pairs.")
    train_df, val_df = train_test_split(new_df, test_size=0.15, random_state=42)
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    print("✅ processing complete.")

if __name__ == "__main__":
    check_gpu()
    download_data()
    process_images_and_captions()
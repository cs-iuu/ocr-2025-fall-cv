import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import os

# --- CONFIGURATION ---
CSV_FILE = 'HMCC all.csv'  # Change to your filename
OUTPUT_DIR = 'tesseract_benchmark'
# Mongolian alphabet in standard order for label mapping
MONGOLIAN_ALPHABET = [
    'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 
    'О', 'Ө', 'П', 'Р', 'С', 'Т', 'У', 'Ү', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 
    'Ы', 'Ь', 'Э', 'Ю', 'Я'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the dataset
print("Loading data...")
df = pd.read_csv(CSV_FILE)

# Process a small sample (e.g., first 500 rows) for the benchmark
samples_to_extract = 500
test_data = df.head(samples_to_extract)

print(f"Creating {samples_to_extract} benchmark images...")

for i, row in test_data.iterrows():
    # Column 0 is usually the label index
    label_idx = int(row.iloc[0])
    char_label = MONGOLIAN_ALPHABET[label_idx] if label_idx < len(MONGOLIAN_ALPHABET) else f"unknown_{label_idx}"
    
    # Remaining 784 columns are pixels
    pixels = row.iloc[1:].values.astype(np.uint8)
    img_array = pixels.reshape((28, 28))
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # PREPROCESSING FOR TESSERACT: 
    # 1. Invert if the background is black (Tesseract likes black text on white)
    img = ImageOps.invert(img) 
    # 2. Resize from 28x28 to 112x112 (upscaling helps OCR accuracy)
    img = img.resize((112, 112), Image.NEAREST)
    # 3. Add a white border (padding helps Tesseract "see" the shape)
    img = ImageOps.expand(img, border=20, fill='white')

    # Save image
    filename = f"sample_{i}_true_{char_label}.png"
    img.save(os.path.join(OUTPUT_DIR, filename))

print(f"Done! Images saved in '{OUTPUT_DIR}' folder.")
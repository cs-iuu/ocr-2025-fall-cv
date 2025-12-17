import pytesseract
from PIL import Image
import os
import pandas as pd

# 1. SETUP
IMAGE_FOLDER = 'tesseract_benchmark'
# Ensure you have the Mongolian language data installed!
TESS_CONFIG = r'-l mon --psm 10' 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

results = []

print("Starting Tesseract Benchmark...")

# 2. RUN TESSERACT ON EVERY IMAGE
for filename in os.listdir(IMAGE_FOLDER):
    if filename.endswith(".png"):
        # The true label is hidden in the filename: "sample_0_true_А.png"
        true_char = filename.split('_')[-1].replace('.png', '')
        
        # Open image and run OCR
        img_path = os.path.join(IMAGE_FOLDER, filename)
        img = Image.open(img_path)
        
        # Tesseract prediction
        predicted_char = pytesseract.image_to_string(img, config=TESS_CONFIG).strip()
        
        # Log the result
        is_correct = (predicted_char == true_char)
        results.append({
            'filename': filename,
            'true': true_char,
            'predicted': predicted_char,
            'correct': is_correct
        })

# 3. CALCULATE ACCURACY
df_results = pd.DataFrame(results)
accuracy = df_results['correct'].mean() * 100

print(f"\n--- BENCHMARK RESULTS ---")
print(f"Total Images Tested: {len(df_results)}")
print(f"Tesseract Accuracy: {accuracy:.2f}%")

# Save to CSV so you can make a chart for your presentation later
df_results.to_csv('tesseract_vs_groundtruth.csv', index=False)
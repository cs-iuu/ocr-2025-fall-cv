import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

# --- 1. CONFIGURATION ---
# Replace 92.5 with the actual accuracy from your Recognition Team
CUSTOM_MODEL_ACCURACY = 92.5 
RESULTS_CSV = 'tesseract_vs_groundtruth.csv'

# --- 2. DATA CLEANING ---
if not os.path.exists(RESULTS_CSV):
    print(f"Error: {RESULTS_CSV} not found. Run benchmark.py first!")
    exit()

df = pd.read_csv(RESULTS_CSV)

# FIX: Replace NaN (floats) with empty strings to prevent 'float vs str' errors
df = df.fillna("")

# Calculate Tesseract Accuracy
tess_accuracy = (df['correct'].mean() * 100)

print(f"Benchmark Results Loaded.")
print(f"Tesseract Accuracy: {tess_accuracy:.2f}%")
print(f"Custom Model Accuracy: {CUSTOM_MODEL_ACCURACY:.2f}%")

# --- 3. VISUAL: ACCURACY COMPARISON ---
def plot_accuracy():
    plt.figure(figsize=(10, 6))
    labels = ['Tesseract (Baseline)', 'Our Custom Model']
    values = [tess_accuracy, CUSTOM_MODEL_ACCURACY]
    colors = ['#e74c3c', '#2ecc71'] # Red vs Green

    bars = plt.bar(labels, values, color=colors, width=0.6)
    
    # Add percentage labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.title('End-to-End Pipeline Performance: Mongolian Cyrillic', fontsize=14)
    plt.ylabel('Accuracy Percentage (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig('accuracy_comparison.png')
    print("Saved: accuracy_comparison.png")

# --- 4. VISUAL: CONFUSION MATRIX ---
def plot_cm():
    # Only plot if Tesseract actually predicted something to avoid a massive blank plot
    # We use unique values from the 'true' column to keep the matrix manageable
    labels = sorted(df['true'].unique())
    
    # Filter out empty predictions for the matrix to see specific character errors
    df_with_preds = df[df['predicted'] != ""]
    
    if len(df_with_preds) < 2:
        print("Tesseract had too few successful predictions for a Confusion Matrix.")
        return

    cm = confusion_matrix(df['true'], df['predicted'], labels=labels)
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                xticklabels=labels, yticklabels=labels)
    
    plt.title('Tesseract Confusion Matrix (Where Predictions Occurred)')
    plt.xlabel('Predicted Character')
    plt.ylabel('Actual Character (Ground Truth)')
    
    plt.savefig('confusion_matrix.png')
    print("Saved: confusion_matrix.png")

# --- 5. EXECUTE ---
plot_accuracy()
plot_cm()
print("\nSuccess!.")
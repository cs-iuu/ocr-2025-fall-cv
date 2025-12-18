import os
from PIL import Image, ImageDraw, ImageFont
import random

# Settings
CHARACTERS = "АБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯ"  # Mongolian Cyrillic
FONT_PATH = "/System/Library/Fonts/Supplemental/SnellRoundhand.ttc"
OUTPUT_DIR = "synthetic_data"
IMG_SIZE = (64, 64)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_char_image(char, font_path, size):
    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_size = random.randint(40, 50)
    font = ImageFont.truetype(font_path, font_size)

    # Newer way to get text dimensions
    left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
    w = right - left
    h = bottom - top

    # Center with jitter
    position = ((size[0] - w) / 2 - left + random.randint(-3, 3),
                (size[1] - h) / 2 - top + random.randint(-3, 3))

    draw.text(position, char, fill=(0, 0, 0), font=font)
    img = img.rotate(random.randint(-8, 8), fillcolor=(255, 255, 255))
    return img


# Generate 100 variations per character
for char in CHARACTERS:
    char_dir = os.path.join(OUTPUT_DIR, char)
    os.makedirs(char_dir, exist_ok=True)

    for i in range(100):
        img = generate_char_image(char, FONT_PATH, IMG_SIZE)
        img.save(f"{char_dir}/{char}_{i}.png")

print("Synthetic dataset created!")

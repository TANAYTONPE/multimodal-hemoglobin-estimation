import os, glob
import numpy as np
import pandas as pd
from PIL import Image

# Count files
nail_crops = glob.glob('data/crops/nail/*.jpg')
skin_crops = glob.glob('data/crops/skin/*.jpg')
conj_crops = glob.glob('data/crops/conjunctiva/*.jpg')
eye_raw = glob.glob('data/eyes_dataset/*.jpg')
photo_raw = glob.glob('data/photo/*.jpg')

print(f'=== IMAGE COUNTS ===')
print(f'Nail crops: {len(nail_crops)}')
print(f'Skin crops: {len(skin_crops)}')
print(f'Conjunctiva crops: {len(conj_crops)}')
print(f'Eye raw images: {len(eye_raw)}')
print(f'Photo raw images (nail/skin source): {len(photo_raw)}')

def total_size_mb(files):
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f)) / (1024*1024)

print(f'\n=== DATASET SIZES ===')
print(f'Nail crops: {total_size_mb(nail_crops):.1f} MB')
print(f'Skin crops: {total_size_mb(skin_crops):.1f} MB')
print(f'Conj crops: {total_size_mb(conj_crops):.1f} MB')
print(f'Eye raw: {total_size_mb(eye_raw):.1f} MB')
print(f'Photo raw: {total_size_mb(photo_raw):.1f} MB')
total_all = total_size_mb(nail_crops) + total_size_mb(skin_crops) + total_size_mb(conj_crops) + total_size_mb(eye_raw) + total_size_mb(photo_raw)
print(f'TOTAL: {total_all:.1f} MB ({total_all/1024:.2f} GB)')

print(f'\n=== NAIL/SKIN DATASET (metadata.csv) ===')
df = pd.read_csv('data/metadata.csv')
print(f'Total patients: {len(df)}')
print(df['split'].value_counts().to_string())
hb = df['hemoglobin']
print(f'Hb range: {hb.min()} - {hb.max()} g/dL')
print(f'Hb mean: {hb.mean():.2f}, std: {hb.std():.2f} g/dL')

print(f'\n=== CONJUNCTIVA DATASET (conjunctiva_manifest.csv) ===')
df_conj = pd.read_csv('data/conjunctiva_manifest.csv')
print(f'Total patients: {len(df_conj)}')
print(df_conj['split'].value_counts().to_string())
hbc = df_conj['hemoglobin']
print(f'Hb range: {hbc.min()} - {hbc.max()} g/dL')
print(f'Hb mean: {hbc.mean():.2f}, std: {hbc.std():.2f} g/dL')
print(f'Cohorts: {df_conj["cohort"].value_counts().to_dict()}')

print(f'\n=== SAMPLE IMAGE DIMENSIONS ===')
for p in photo_raw[:3]:
    im = Image.open(p)
    sz = os.path.getsize(p) / 1024
    print(f'  {os.path.basename(p)}: {im.size[0]}x{im.size[1]}px, {im.mode}, {sz:.0f} KB')

for p in eye_raw[:3]:
    im = Image.open(p)
    sz = os.path.getsize(p) / 1024
    print(f'  {os.path.basename(p)}: {im.size[0]}x{im.size[1]}px, {im.mode}, {sz:.0f} KB')

for p in conj_crops[:3]:
    im = Image.open(p)
    sz = os.path.getsize(p) / 1024
    print(f'  CROP {os.path.basename(p)}: {im.size[0]}x{im.size[1]}px, {im.mode}, {sz:.0f} KB')

# Check for any missing files
print(f'\n=== MISSING FILE CHECK ===')
missing_nail = [p for p in df['image_path'].tolist() if not os.path.isfile(p)]
print(f'Missing raw photos: {len(missing_nail)}/{len(df)}')

# Total raw images for nail/skin patients × 3 finger crops each
total_nail_expected = len(df) * 3
total_skin_expected = len(df) * 3
print(f'Nail crops found: {len(nail_crops)} (expected {total_nail_expected})')
print(f'Skin crops found: {len(skin_crops)} (expected {total_skin_expected})')

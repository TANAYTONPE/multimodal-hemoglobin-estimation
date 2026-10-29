import sys, os
sys.path.insert(0, 'src')
from PIL import Image
from inference_pipeline import get_inference_engine

print('Loading engine...')
engine = get_inference_engine()
print('Engine loaded! Models available:', list(engine._models.keys()))

# Test with sample images
p_nail = os.path.join('data', 'sample_images', 'sample_fingernail.jpg')
p_skin = os.path.join('data', 'sample_images', 'sample_skin.jpg')
p_conj = os.path.join('data', 'sample_images', 'sample_conjunctiva.jpg')

im_nail = Image.open(p_nail).convert('RGB')
im_skin = Image.open(p_skin).convert('RGB')
im_conj = Image.open(p_conj).convert('RGB')

print('Testing predict_single (fingernail)...')
r1 = engine.predict_single(im_nail, modality='fingernail')
print(f'  Nail Hb: {r1["predicted_hb"]} g/dL | {r1["who_label"]}')

print('Testing predict_single (skin)...')
r2 = engine.predict_single(im_skin, modality='skin')
print(f'  Skin Hb: {r2["predicted_hb"]} g/dL | {r2["who_label"]}')

print('Testing predict_single (conjunctiva)...')
r3 = engine.predict_single(im_conj, modality='conjunctiva')
print(f'  Conj Hb: {r3["predicted_hb"]} g/dL | {r3["who_label"]}')

print('Testing predict_multimodal_consensus (all 3)...')
cons = engine.predict_multimodal_consensus(nail_image=im_nail, skin_image=im_skin, conj_image=im_conj)
print(f'  Consensus Hb: {cons["predicted_hb"]} g/dL | {cons["who_label"]}')
print(f'  Model name: {cons["model_name"]}')
print(f'  Individual preds: {cons["individual_predictions"]}')
print(f'  Weights: {cons["normalized_weights"]}')

print('Testing predict_multimodal_fusion (nail+skin)...')
m4 = engine.predict_multimodal_fusion(im_nail, im_skin)
print(f'  Model4 Hb: {m4["predicted_hb"]} g/dL')

print('\nALL TESTS PASSED!')

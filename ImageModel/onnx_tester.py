#!/usr/bin/env python3
"""Minimal hard-coded ONNX tester for PCOS image model
    NOTE: I HAVE NO IDEA HOW THE CLIENT WILL ACTUALLY HOST THE MODELS OR HOW THEY'LL INTEGRATE THEM:
    a good idea is to just use an API to some server to host the model code might look like this:
    that also means you can just basically use this same python code and interact with it via API calls
    
    app = FastAPI()

    # Load the model once when the server starts
    sess = ort.InferenceSession("pcos_image_model.onnx", providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        # 1. Receive the file and convert to PIL Image
        contents = await file.read()
        raw_img = Image.open(io.BytesIO(contents))
        
        # 2. Run your existing pipeline
        sanitized = apply_sanitizer(raw_img)
        tensor = preprocess_for_model(sanitized)
        
        # 3. Inference
        batch = np.expand_dims(tensor, axis=0)
        raw_logit = sess.run(None, {input_name: batch})[0].flatten()[0]
        
        # 4. Binary Logic
        prob_infected = 1.0 - sigmoid(raw_logit)
        prediction = "Infected" if prob_infected >= 0.5 else "Healthy"
        
        # 5. Return JSON (The "Usable" format for the UI)
        return {
            "filename": file.filename,
            "prediction": prediction,
            "probability": float(prob_infected),
            "status": "success"
        }

        
THIS IS JUST A SAMPLE SCRIPT FOR HOW TO USE THE ONNX MODEL WITH THE APPROPRIATE PREPROCESSING REQUIRED.

"""

from pathlib import Path
import os
import random
import csv
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort
import time
import data_sanitizer as ds
import matplotlib.pyplot as plt

HERE = Path(__file__).parent.resolve()
ONNX_PATH = HERE / 'onnx_models' / 'pcos_image_model.onnx'
DATA_DIR = HERE / 'data'
SAMPLE_COUNT = 10

# sanitizer provides IMG_SIZE=384; training pipeline used IMG_SIZE->resize 224 at train time
SANITIZER_SIZE = ds.IMG_SIZE
MODEL_INPUT_SIZE = 224

# ImageNet normalization used in training notebook
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

OUT_CSV = HERE / 'onnx_test_report.csv'


def list_images(root: Path):
    start_time = time.time()
    print(f"DEBUG: Starting file discovery in {root}...")
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    files = []
    # Using a more efficient rglob strategy
    for ext in exts:
        found = list(root.rglob(f"*{ext}"))
        files.extend(found)
    
    elapsed = time.time() - start_time
    print(f"DEBUG: Found {len(files)} total images in {elapsed:.2f} seconds.")
    return files

#KEY STEP, APPLY SANITIZATION
""""
The pipeline performs the following operations:
    1.  **Color Space Conversion**: Ensures the image is in RGB mode to maintain compatibility 
        with subsequent transformations.
    2.  **Border Removal**: Trims peripheral black borders using a threshold-based detection 
        to isolate the primary subject.
    3.  **Center Cropping**: Extracts the central 85% of the image to reduce peripheral 
        noise and focus on the core features.
    4.  **Contrast Enhancement**: Applies histogram equalization to distribute pixel 
        intensities, improving image clarity and contrast.
    5.  **Grayscale Conversion**: Transforms the image to a 3-channel grayscale format 
        (preserving the RGB structure while removing color information).
    6.  **Rescaling & Letterboxing**: Resizes the image to fit within the defined 
        'SANITIZER_SIZE' while maintaining the original aspect ratio, then centers 
        it on a black square background.
"""

def apply_sanitizer(img: Image.Image) -> Image.Image:
    """Standardizes image through the cleaning pipeline."""
    # Note: No print here to avoid flooding console for every image
    img = img.convert('RGB')
    img = ds.trim_black_border(img, threshold=8, margin=4)
    img = ds.center_crop_percent(img, pct=0.85)
    img = ImageOps.equalize(img)
    img = ds.to_gray3(img)
    img = ImageOps.contain(img, (SANITIZER_SIZE, SANITIZER_SIZE))
    bg = Image.new('RGB', (SANITIZER_SIZE, SANITIZER_SIZE), (0, 0, 0))
    bg.paste(img, ((SANITIZER_SIZE - img.size[0]) // 2, (SANITIZER_SIZE - img.size[1]) // 2))
    return bg

r"""
    Converts a PIL image into a normalized NumPy tensor ready for model inference.

    The preprocessing pipeline includes:
    1.  **Resizing**: Scales the image to the required 'MODEL_INPUT_SIZE' using 
        bilinear interpolation to maintain smooth gradients.
    2.  **Scaling**: Converts pixel values from integers [0, 255] to floating-point 
        values in the range [0.0, 1.0].
    3.  **Normalization**: Applies ImageNet-standard zero-centering and scaling 
        per channel using the formula: $z = \frac{x - \mu}{\sigma}$
    4.  **Dimension Permutation**: Transposes the array from Interleaved format 
        (Height, Width, Channels) to Planar format (Channels, Height, Width) to 
        match NCHW expectations.

    Args:
        img (Image.Image): The processed PIL image from the sanitizer.

    Returns:
        np.ndarray: A float32 array with shape (3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).
"""

def preprocess_for_model(img: Image.Image) -> np.ndarray:
    """Resizes and normalizes for model input."""
    img = img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN.reshape((1, 1, 3))) / IMAGENET_STD.reshape((1, 1, 3))
    arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
    return arr

def sigmoid(x):
    """Standard logistic function to map raw logits to [0, 1] range."""
    return 1.0 / (1.0 + np.exp(-x))

def run():
    """
    Main execution logic for model inference. 
    Steps: Initialization -> Data Loading -> Preprocessing -> Inference -> Post-processing -> Reporting.
    """
    print(f"\n--- STARTING ONNX TESTER ---")
    
    # --- STEP 1: INITIALIZATION ---
    # To use this model in any language:
    # 1. Load the ONNX file into your chosen Inference Engine (ONNX Runtime, TensorRT, etc.)
    # 2. Identify the input node name (usually the first input of the graph).
    if not ONNX_PATH.is_file():
        print(f"ERROR: Model not found at {ONNX_PATH}")
        return

    all_imgs = list_images(DATA_DIR)
    if not all_imgs:
        print("ERROR: No images found.")
        return
    
    random.seed(42)
    sample = random.sample(all_imgs, min(SAMPLE_COUNT, len(all_imgs)))

    print("DEBUG: Loading ONNX Session...")
    sess = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    # --- STEP 2: INFERENCE LOOP ---
    results = []
    plot_data = [] 
    
    print(f"DEBUG: Processing {len(sample)} images...")
    for i, p in enumerate(sample):
        try:
            # A. IMAGE ACQUISITION
            # you can use any method desired by client to load image
            # Load the raw image data into memory.
            raw_img = Image.open(p)
            
            # B. SANITIZATION (The Cleaning Pipeline)
            # This step removes noise, standardizes borders, and equalizes contrast.
            # This must be replicated exactly in other languages to maintain accuracy.
            sanitized_img = apply_sanitizer(raw_img)

            # C. PREPROCESSING (The Tensor Pipeline)
            # 1. Resize: Use Bilinear interpolation to 224x224.
            # 2. Scale: Divide pixel values (0-255) by 255.0 to get 0.0-1.0 range.
            # 3. Normalize: Apply ImageNet Mean [0.485, 0.456, 0.406] and Std [0.229, 0.224, 0.225].
            # 4. Format: Reorder dimensions from HWC (Height, Width, Channels) to CHW (Channels, Height, Width).
            tensor = preprocess_for_model(sanitized_img)
            
            # D. EXECUTION
            # Wrap the tensor in a batch dimension (N, C, H, W) where N=1.
            # Pass to the engine and retrieve the raw output (logit).
            batch = np.expand_dims(tensor, axis=0)
            raw_logit = sess.run(None, {input_name: batch})[0].flatten()[0]
            
            # E. ACTIVATION & INTERPRETATION
            # 1. Map the logit to a 0.0-1.0 scale using the Sigmoid function.
            # 2. LABEL LOGIC: The training process mapped alphabetical order: 
            #    'Infected' = Index 0, 'Non-Infected' = Index 1.
            #    Therefore, the sigmoid result naturally represents P(Non-Infected).
            #    To get P(Infected), we must calculate: 1.0 - P(Non-Infected).
            prob_raw = sigmoid(raw_logit)
            prob_infected = 1.0 - prob_raw 
            
            # F. THRESHOLDING
            # Standard binary decision boundary is 0.5.
            pred = 1 if prob_infected >= 0.5 else 0
            

            #------------------------------------------------------------------#
            #Code above this is the only ones essential for the actual prediction, if according to client we want the prob, just return prob_infected
            #everything below is just for reporting and visualization
            #------------------------------------------------------------------#



            # --- STEP 3: DATA COLLECTION ---
            folder = p.parent.name.lower()
            true_label = 1 if 'infect' in folder and 'non' not in folder else 0
            
            res_entry = {
                'file': p.name,
                'true': true_label,
                'pred': pred,
                'prob': prob_infected,
                'correct': (true_label == pred)
            }
            results.append(res_entry)
            plot_data.append((sanitized_img, res_entry))
            
            print(f" [{i+1}/{len(sample)}] {p.name:<25} | Correct: {res_entry['correct']}")
            
        except Exception as e:
            print(f" [!] Error on {p.name}: {e}")

    # --- STEP 4: OUTPUT GENERATION ---
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = (correct_count / len(results)) * 100 if results else 0
    print(f"\nFinal Accuracy: {accuracy:.2f}% ({correct_count}/{len(results)})")

    # Visual Plotting
    cols = 5
    rows = (len(plot_data) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()

    for idx, (img, res) in enumerate(plot_data):
        axes[idx].imshow(img)
        title_color = 'green' if res['correct'] else 'red'
        title_text = f"File: {res['file'][:15]}...\nTrue: {res['true']} | Pred: {res['pred']}\nProb: {res['prob']:.3f}"
        axes[idx].set_title(title_text, color=title_color, fontsize=9)
        axes[idx].axis('off')

    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    
    # Save results to disk
    with open(OUT_CSV, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"DEBUG: Report saved to {OUT_CSV}")
    plt.show()

if __name__ == '__main__':
    run()

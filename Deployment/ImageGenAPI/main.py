# main.py
import gensim.models.fasttext
from quart import Quart, request, jsonify, Response
from quart_cors import cors
from pydantic import BaseModel
from typing import List
import torch, os, io
import gensim.models.fasttext # <-- Added for FastText
from generator import *
from utils import *

app = Quart(__name__)
app = cors(app)

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Configuration ---
FASTTEXT_DIM = 300
NOISE_DIM = 128            # Dimension of the noise vector
NGF = 256                   # Base number of generator filters
LABEL_NC_G = 256            # Number of channels for SPADE conditioning map
ADD_ATTENTION_G = True      # Use attention in G (must match saved model)
INPUT_CHANNELS = 1         # Grayscale input
OUTPUT_CHANNELS = 1        # Grayscale output

GENERATOR_MODEL_PATH = "./Model_Training/Models/generator_epoch_102.pth" # Example path
MODEL_PATH = "./Model_Training/Models/cc.hi.300.bin" # Make sure this file is accessible
print(os.getcwd())

print("Loading Generator model...")
gen = UNetStyleGenerator(
    fasttext_dim=FASTTEXT_DIM,
    noise_dim=NOISE_DIM,
    input_channels=INPUT_CHANNELS,
    output_channels=OUTPUT_CHANNELS,
    ngf=NGF,
    label_nc=LABEL_NC_G,
    add_attention=ADD_ATTENTION_G
)

state_dict = torch.load(os.path.normpath(GENERATOR_MODEL_PATH), map_location=DEVICE)
gen.load_state_dict(state_dict)
gen.to(DEVICE) # Move model to device
gen.eval()
print("Loaded Generator model...")
print("Loading FastText model...")
model = gensim.models.fasttext.load_facebook_model(MODEL_PATH)
print("Loaded FastText model...")


@app.route('/predict', methods=['POST'])
async def predict():
    data = await request.get_json()  # Retrieve JSON data from the request
    
    if not data or any(key not in data for key in ['sentence', 'font_size', 'canvas_width']):
        return jsonify({"error": "Request Body must contain 'sentence', 'font_size' and 'canvas_width'"}), 400
    
    sentence = data['sentence']
    font_size = data['font_size']
    canvas_width = data['canvas_width']
    
    try:
        # Generate image from the given word
        processed_image = place_words_on_canvas(font_size, sentence, canvas_width, gen, model, DEVICE, debug=False)
        
        # Convert the processed image to bytes for response
        img_byte_arr = io.BytesIO()
        processed_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Return the image in the response
        return Response(img_byte_arr, mimetype='image/png')
    
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

# --- To run the app (optional, usually done via command line) ---
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
    # uvicorn main:app --reload
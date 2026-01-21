import os
import time
import io
from flask import Flask, render_template, request, send_file
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# 1. Load variables from .env
load_dotenv()

# 2. Initialize the Flask App (This fixes your NameError!)
app = Flask(__name__)

# 3. Access the token securely
HF_TOKEN = os.getenv("HF_TOKEN")

# 4. Initialize the Hugging Face Client
client = InferenceClient(
    provider="hf-inference", 
    api_key=HF_TOKEN
)

SAVE_DIR = "static/generated_images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form.get('prompt')
    if not prompt:
        return "Prompt is required", 400

    try:
        # Generate the image
        image = client.text_to_image(
            prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0"
        )

        # Save locally with a unique name
        filename = f"art_{int(time.time())}.png"
        filepath = os.path.join(SAVE_DIR, filename)
        image.save(filepath)

        # Prepare to send to the web browser
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        print(f"Backend Error: {e}")
        return f"Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify, send_file
from diffusers import StableDiffusionXLInpaintPipeline
import torch
from PIL import Image
import io
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the finetuned model from the local directory using the SDXL inpainting pipeline
pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
    r"C:\FYPMODEL\SDXL_Inpaint_Model",  # Load from the local path
    torch_dtype=torch.float16  # Use float16 for faster inference if supported
).to("cuda")  # Move the pipeline to GPU

# Define the prompt
prompt = "Satellite Imagery Residential Houses"
negative_prompt = ""  # Optional: add a negative prompt if desired

@app.route('/inpaint', methods=['POST'])
def inpaint():
    try:
        # Fetch image and mask from the request
        image_file = request.files['image']
        mask_file = request.files['mask']

        # Open the image and mask
        image = Image.open(image_file).convert("RGB").resize((1024, 1024))
        mask = Image.open(mask_file).convert("L").resize((1024, 1024))

        # Run inference with autocast for mixed precision
        with torch.autocast("cuda"):
            output = pipeline(
                prompt=prompt,
                image=image,
                mask_image=mask,
                negative_prompt=negative_prompt,  # Optional
                num_inference_steps=50,  # You can adjust this
                guidance_scale=7.5,      # You can adjust this
            ).images[0]

        # Save the output to a BytesIO object
        img_byte_arr = io.BytesIO()
        output.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Return the image as a response
        return send_file(img_byte_arr, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000)
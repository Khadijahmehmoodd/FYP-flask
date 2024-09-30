import requests
from PIL import Image
import io

# Define the file paths for the image and mask
image_path = r"C:/FYPMODEL/DOP25_LV03_1301_11_2015_1_15_497500.0_120187.5.png"
mask_path = r"C:/FYPMODEL/DOP25_LV03_1301_11_2015_1_15_497500.0_120187.5_label.png"

# Define the API endpoint URL
url = "http://127.0.0.1:5000/inpaint"

# Open the image and mask files
with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
    # Prepare the files for the POST request
    files = {
        "image": image_file,
        "mask": mask_file
    }

    # Send the POST request to the Flask API
    response = requests.post(url, files=files)

    # Check if the request was successful
    if response.status_code == 200:
        # Load the returned image from the response
        output_image = Image.open(io.BytesIO(response.content))

        # Save the output image to a file
        output_image.save("output_image.png")

        print("Output image saved as output_image.png")
    else:
        print(f"Failed to process the image. Error: {response.content.decode()}")
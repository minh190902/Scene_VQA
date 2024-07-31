import base64
from PIL import Image
from io import BytesIO
import re
import requests

def process_user_input(user_input):
    """
    Processes the user's input to separate image data from the text.

    This function looks for an image in the form of a URL or a Base64-encoded string
    within the user's input. If found, it separates the image data from the
    rest of the text, fetches or decodes the image, and returns both the plain text and 
    the image object.

    Parameters:
    - user_input (str): The input string provided by the user, potentially containing
      a question or statement and a URL to an image or a Base64-encoded image.

    Returns:
    - tuple:
        - (str): The text portion of the user input without the image data.
        - (Image or None): A PIL Image object if the image data is successfully fetched
          or decoded, otherwise None if no image data is found.
    """
    # Regex to find Base64 image data
    base64_image_data_regex = re.compile(r"data:image/(jpeg|png|gif|bmp|webp);base64,([A-Za-z0-9+/=]+)")
    # Regex to find URLs, including specific pattern for encrypted-tbn0.gstatic.com
    url_regex = re.compile(r'https://encrypted-tbn0\.gstatic\.com/images\?q=tbn:[^&\s]+|https?://\S+\.(?:jpg|jpeg|png|gif|bmp|webp)')

    # Search for Base64 image data in the input
    base64_match = base64_image_data_regex.search(user_input)

    if base64_match:
        # Extract Base64 image data
        image_data = base64_match.group(2)

        try:
            # Decode the base64 image data
            image_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(image_bytes))

            # Remove the Base64 image data from the user input
            user_prompt = base64_image_data_regex.sub("", user_input).strip()

            # Use default text if no user prompt remains
            if not user_prompt:
                user_prompt = "mô tả bức ảnh"

            return user_prompt, img
        except (base64.binascii.Error, IOError):
            # Handle errors (e.g., decoding issues, image format issues)
            print("Failed to decode the base64 image data.")
            return user_input, None

    else:
        # Search for image URL in the input
        url_match = url_regex.search(user_input)
        if url_match:
            # Extract image URL
            image_url = url_match.group(0)

            try:
                # Make a request to fetch the image
                response = requests.get(image_url)
                response.raise_for_status()

                # Open the image
                img = Image.open(BytesIO(response.content))

                # Remove the URL from the user input
                user_prompt = url_regex.sub("", user_input).strip()

                # Use default text if no user prompt remains
                if not user_prompt:
                    user_prompt = "mô tả bức ảnh"

                return user_prompt, img
            except requests.exceptions.RequestException:
                # Handle errors (e.g., network issues, invalid URL)
                print("Failed to fetch the image from the URL.")
                return user_input, None
        else:
            # No image data found
            return user_input, None

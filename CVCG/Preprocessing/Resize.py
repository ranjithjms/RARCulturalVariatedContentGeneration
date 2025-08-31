from PIL import Image

class Resize:
    def resize_image(self, input_path, output_path):
        new_width = 800
        new_height = 600
        # Open the original image
        with Image.open(input_path) as img:
            # Resize the image
            resized_img = img.resize((new_width, new_height))
            # Save the resized image
            resized_img.save(output_path)
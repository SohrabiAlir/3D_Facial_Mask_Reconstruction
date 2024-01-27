from PIL import Image
import os

def convert_webp_to_png(input_path, output_path):
    try:
        # Open the WebP image
        with Image.open(input_path) as img:
            # Save as PNG
            img.save(output_path, "PNG")
        print(f"Conversion successful: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error converting {input_path} to PNG: {e}")

# Replace these paths with the actual file paths
webp_file_path = "data/9.avif"
png_output_path = "data/9.png"

convert_webp_to_png(webp_file_path, png_output_path)

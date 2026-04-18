import os
from PIL.ExifTags import TAGS
from PIL import Image, ImageChops, ImageEnhance
import cv2

def perform_ela(image_path, quality=90):
    """
    Performs Error Level Analysis (ELA) on an image.
    ELA highlights areas with different compression levels. In a real photo,
    the 'noise' should be relatively uniform across the same texture. 
    Bright spots in the ELA map often suggest manipulation.
    """
    try:
        # 1. Open original and convert to RGB
        original = Image.open(image_path).convert('RGB')
        
        # 2. Save as a temporary JPEG to introduce compression artifacts
        temp_ela = "temp_ela_buffer.jpg"
        original.save(temp_ela, 'JPEG', quality=quality)
        
        # 3. Re-open the recompressed image
        recompressed = Image.open(temp_ela)
        
        # 4. Calculate the absolute difference between original and recompressed
        ela_image = ImageChops.difference(original, recompressed)
        
        # 5. Scale the differences to make them visible (Normalization)
        extrema = ela_image.getextrema()
        # Find the maximum brightness value across all three channels
        max_diff = max([ex[1] for ex in extrema])
        
        if max_diff == 0:
            max_diff = 1 # Prevent division by zero
            
        scale = 255.0 / max_diff
        ela_enhanced = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        # 6. Clean up temporary file
        if os.path.exists(temp_ela):
            os.remove(temp_ela)
            
        return ela_enhanced
        
    except Exception as e:
        print(f"Error during ELA process: {e}")
        return None

def analyze_metadata_anomaly(metadata_dict):
    """
    Checks for specific red flags in metadata that suggest AI generation or manipulation.
    """
    red_flags = []
    
    software = metadata_dict.get('Software', '').lower()
    make = metadata_dict.get('Make', '')
    
    # Signatures of common manipulation or AI tools
    ai_tools = ['stable diffusion', 'midjourney', 'dall-e', 'generative', 'photoshop', 'lightroom']
    for tool in ai_tools:
        if tool in software:
            red_flags.append(f"Software signature found: {tool.capitalize()}")

    # Anomaly: Software present but no Camera 'Make'
    if software and not make:
        red_flags.append("Anomaly: Image has software traces but missing camera hardware info.")

    return red_flags

def blur_Detection(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    return blur_score

def get_metadata(image_path):
    """
    Extracts EXIF metadata from an image file.
    Returns a dictionary of metadata tags and their values.
    """
    metadata = {}
    try:
        image = Image.open(image_path)
        info = image._getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                metadata[decoded] = value
        
        # Add some basic file info even if EXIF is missing
        metadata['Format'] = image.format
        metadata['Mode'] = image.mode
        metadata['Size'] = image.size
        
        return metadata
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return None

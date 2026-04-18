from PIL import Image
from PIL.ExifTags import TAGS

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

from PIL import Image
from base64 import b64decode as decode_string
from io import BytesIO 

IMG_SIZE = (224,224)

def convert_to_bytes(file_or_bytes):
    if isinstance(file_or_bytes, str):
        image = Image.open(file_or_bytes)
    else:
        try:
            image = Image.open(BytesIO(decode_string(file_or_bytes)))
        except Exception as e:
            dataBytesIO = BytesIO(file_or_bytes)
            image = Image.open(dataBytesIO)
    
    if image.size != IMG_SIZE:
        image = image.resize(IMG_SIZE, Image.ANTIALIAS)
    with BytesIO() as image_bytes:
        image.save(image_bytes, format="PNG")
        bytes = image_bytes.getvalue()
        return bytes, image

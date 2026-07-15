from PIL import Image
import numpy as np

def show_np_image(img: np.ndarray):
    image = Image.fromarray(img.astype(np.uint8))
    image.show()

    
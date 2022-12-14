
from frcnn import FRCNN
from PIL import Image

frcnn = FRCNN()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
        image = image.convert("RGB")
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = frcnn.detect_image(image)
        if r_image != None:
            r_image.show()

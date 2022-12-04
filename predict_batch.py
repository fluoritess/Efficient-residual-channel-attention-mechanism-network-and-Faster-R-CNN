from frcnn import FRCNN
from PIL import Image
import os
frcnn = FRCNN()

for root, dirs, files in os.walk(r"img"):
    for file in files:
        img=os.path.join(root, file)
        try:
            image = Image.open(img)
            image = image.convert("RGB")
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = frcnn.detect_image(image)
            if r_image!=None:
                # r_image.show()
                r_image.save("img\out\\" +root.split('\\')[2]+ file)

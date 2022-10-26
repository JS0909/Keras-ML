from PIL import Image
import numpy as np
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
im = Image.open('05.jpg')
im = np.array(im)
print(im)



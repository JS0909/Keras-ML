from PIL import Image
import albumentations as A  # torchvision을 대신할만한 라이브러리. image augmentation 등의 기능이 많고 빠르다
import numpy as np

mean1 = [90, 100, 100]
std1 = [30, 32, 28]

trans = A.Compose(
            [
                A.Normalize(max_pixel_value=255.)
            ])

img = np.array(Image.open('D:\study_data\_data/team_project\predict_img/02.jpg'))
# print(np.max(img), np.min(img))

img2 = (img - np.mean(img)*1) / np.std(img)*1

print(np.mean(img))
print(np.std(img))

print('변형전', img[-1,-1,-1])
img = trans(image=img)

print('수동정규화', img2[-1,-1,-1])
print('A정규화',img['image'][-1,-1,-1])
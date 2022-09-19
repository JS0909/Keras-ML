import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # 제대로 다운로드 안되면 추가하는 코드

import easyocr
import cv2
import matplotlib.pyplot as plt

reader = easyocr.Reader(['ko', 'en'])

# img = cv2.imread('C:\study\project/team/1.png')
img = cv2.imread('C:\study\project/team/2.jpg')
# plt.figure(figsize=(8,8))
# plt.imshow(img[:,:,::-1])
# plt.axis('off')
# plt.show()

result = reader.readtext(img)
# print(result)
print(result[0])

THRESHOLD = 0.3

for bbox, text, conf in result:
    if conf > THRESHOLD:
        print(text)
        cv2.rectangle(img, pt1=bbox[0], pt2=bbox[2], color=(0, 0, 255), thickness=3)

plt.figure(figsize=(8,8))
plt.imshow(img[:,:,::-1])
plt.axis('off')
plt.show()


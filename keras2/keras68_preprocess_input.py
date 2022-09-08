from keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet') # 이미지넷 대회에서 쓴 오브젝트 분류 데이터를 끌어옴
img_path = 'd:/study_data/_data/dog_husky/husky.jpg'
img = image.load_img(img_path, target_size=(224, 224))
print(img) # <PIL.Image.Image image mode=RGB size=224x224 at 0x24F8E8F9EB0>

x = image.img_to_array(img)
print('============================== image.img_to_array(img) ============================================')
print(x, '\n', x.shape) # (224, 224, 3)

x = np.expand_dims(x, axis=0) # axis는 늘려주고 싶은 shape 지점을 지정한다
print('============================== np.expand_dims(x, axis=0) ============================================')
print(x, '\n', x.shape) #  (1, 224, 224, 3)

x = preprocess_input(x) # 0~255 데이터에서 -150~150 데이터로 변경
print('============================== preprocess_input(x) ============================================')
print(x, '\n', x.shape)
print(np.min(x), np.max(x)) # -121.68 134.061

print('============================== model.predict(x) ============================================')
pred = model.predict(x)
print(pred, '\n', pred.shape) #  (1, 1000)
print(np.argmax(pred, axis=1)) # [248]

print('결과는: ', decode_predictions(pred, top=5)[0])
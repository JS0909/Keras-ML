import torch
import random

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
imgs = 'C:\study\project/team\YOLOv7/7.jpg'  # batch of images

# Inference
results = model(imgs)

# Results
# results.show()

# print(results.xyxy[0])  # img1 predictions (tensor)
# print(results.pandas().xyxy[0])  # img1 predictions (pandas)

objects_p = results.pandas().xyxy[0]['name']
obj_list = [x for x in objects_p] # 해당 그림에서 예측한 오브젝트 이름 리스트화
print(obj_list) 

read_obj = random.sample(obj_list, 3) if len(obj_list)>3 else obj_list # 그중에서 랜덤으로 3개 물체 정도 뽑음, 3개보다 적으면 안뽑음
print(read_obj)

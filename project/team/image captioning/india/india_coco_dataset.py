import os
import pickle
import numpy as np
import json

from sklearn.utils import shuffle
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
'''
#================ json 파일 처리 ==================
num_examples= 50000     # 훈련에 사용할 이미지 개수

# annotation json 파일 읽기
with open('D:\study_data\_data/team_project\coco_dataset\json_files/captions_train2014.json', 'r') as f:
  annotations = json.load(f)

# caption과 image name을 vector로 저장
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
  caption = 'start ' + annot['caption'] + ' end'
  image_id = annot['image_id']
  full_coco_image_path = 'D:\study_data\_data/team_project\coco_dataset/train2014/' + 'COCO_train2014_' + '%012d.jpg' % (image_id)

  all_img_name_vector.append(full_coco_image_path)
  all_captions.append(caption)

# caption과 image name들을 섞습니다.(shuffle)
train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector, random_state=1)

# 빠른 학습을 위해서 shuffle된 set에서 처음부터 시작해서 num_examples 개수만큼만 선택합니다.
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]

print('train_captions :', len(train_captions))
print('all_captions :', len(all_captions))

pickle.dump(train_captions, open(os.path.join('D:\study_data\_data/team_project\coco_dataset\img_features', 'train_captions.pkl'), 'wb'))

# print(train_captions[:1])
# print(img_name_vector[:1])
# ['<start> A skateboarder performing a trick on a skateboard ramp. <end>']
# ['D:\\study_data\\_data/team_project\\coco_dataset/train2014/COCO_train2014_000000324909.jpg']
# 순서대로 캡션 하나씩 매칭되어 있음
#=======================================================================================================


#================ 이미지 파일 전처리 (feature extraction) ====================
# load vgg16 model
model = VGG16()
# restructure the model
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# model.summary()

# extract features from image
features = {}

for i, img_path in enumerate(img_name_vector):
    # load the image from file
    image = load_img(img_path, target_size=(224, 224))
    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
    # preprocess image for vgg
    # print(np.max(image), np.min(image)) # 각 픽셀 채널 범위 0 ~ 255 (원본 이미지 포멧)
    image = preprocess_input(image)
    # print(np.max(image), np.min(image)) # 각 픽셀 채널 범위 -151 ~ 151 (이미지넷 대회에서 사용하는 이미지 포맷)
    
    # extract features
    feature = model.predict(image, verbose=1)

    # store feature
    features[str(i)] = feature
    
print(features['0'])

# store features in pickle
pickle.dump(features, open(os.path.join('D:\study_data\_data/team_project\coco_dataset\img_features', 'features.pkl'), 'wb'))
print('img processing done.')


# features = {'0':[이미지숫자화된거], '1':[이미지숫자화된거], '2':[이미지숫자화된거], ...}
#===================================================================================================================================
'''

# features 파일 불러오기
with open(os.path.join('D:\study_data\_data/team_project\coco_dataset\img_features/', 'features.pkl'), 'rb') as f:
  features = pickle.load(f)

#================ 캡션 파일 전처리 ====================
with open(os.path.join('D:\study_data\_data/team_project\coco_dataset\img_features/', 'train_captions.pkl'), 'rb') as f:
  train_captions = pickle.load(f)

print(train_captions)

mapping = {}
# process lines
for i, line in enumerate(train_captions):
  image_id, caption = str(i), line    
  # store the caption
  mapping[image_id] = caption

# print('mapping:', mapping)
# print('mapping_len:', len(mapping))
# mapping = {'0':[캡션문장], '1':[캡션문장], '2':[캡션문장], ...}


def clean(mapping): # 맵핑 딕셔너리 안의 caption을 전처리
  for key, caption in mapping.items():
    # convert to lowercase
    caption = caption.lower()
    # delete digits, special chars, etc.
    caption = caption.replace('[^A-Za-z]', '')# [A-Z] [a-z] : 각각 대문자 알파벳, 소문자 알파벳 모두를 의미
    # delete additional spaces
    caption = caption.replace('\s+', ' ') # [ \t\n\r\f\v] 가 1번 이상 나오면 공백으로 변경
    caption = caption.replace('.', '') # 마침표 제거

    mapping[key]=caption

clean(mapping)
# print(mapping)
# {'0': 'strat a skateboarder performing a trick on a skateboard ramp end', 
#  '1': 'strat a person soaring through the air on skis end', 
#  '2': 'strat a wood door with some boards laid against it end', ...}

# 딕셔너리에서 캡션텍스트만 뽑아오기
all_captions = [caption for key, caption in mapping.items()]

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1 # 패딩토큰 포함

print('vacab_size:', vocab_size)

# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)
print('max_len:', max_length)

#====== train_test_split ========
image_ids = list(mapping.keys())
# split = int(len(image_ids) * 0.90)
train = image_ids[:] # 안함
# test = image_ids[split:]
# validation 세트 따로 있긴한데 솔직히 귀찮잖아? ㅋ
#================================



# create data generator to get data in batch (avoids session crash)
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
  # loop over images
  X1, X2, y = list(), list(), list()
  n = 0
  while 1:
    for key in data_keys:
      n += 1
      caption = mapping[key]
            
    # encode the sequence
      seq = tokenizer.texts_to_sequences([caption])[0] # 리스트 안에 넣고 (한문장씩 들어가 있으니까)
                                                         # 첫문장을 토크나이징하는 것으로 해야함
    # split the sequence into X, y pairs
      for i in range(1, len(seq)):
        # split into input and output pairs
        in_seq, out_seq = seq[:i], seq[i] # 현재 문장을 인풋으로, 다음에 올 단어를 아웃풋으로
        # pad input sequence
        in_seq = pad_sequences([in_seq], maxlen=max_length)[0] # 최대 문장 길이만큼 패딩(0을 앞쪽에 채움)
        # encode output sequence
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

        # store the sequences
        X1.append(features[key][0]) # features 에 하나의 key에 해당하는 이미지 피쳐가 리스트로 묶여있기 때문에 인덱스로 부름
        X2.append(in_seq)
        y.append(out_seq)
      if n == batch_size: # 배치 사이즈만큼 차면 yield로 한묶음 채워서 뱉음
          X1, X2, y = np.array(X1), np.array(X2), np.array(y)
          yield [X1, X2], y
          X1, X2, y = list(), list(), list()
          n = 0



# encoder model
# image feature layers
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')


# train the model
print('start training...')
epochs = 70
batch_size = 64
steps = len(train) // batch_size # 1 batch 당 훈련하는 데이터 수

# 제너레이터 함수에서 yield로 252개의 [X1, X2], y 묶음이 차곡차곡 쌓여 있고  steps_per_epoch=steps 이 옵션으로
# epoch 1번짜리 fit을 돌때 정해준steps번 generator 를 호출함. iterating 을 steps번 함

for i in range(epochs):
  print(f'epoch: {i+1}')
  # create data generator
  generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
  # fit for one epoch
  model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1) # generator -> [X1, X2], y
print('done training.')

# save the model
model.save('D:\study_data\_data/team_project\coco_dataset\model_save/best_model.h5')
print('model saved.')


def idx_to_word(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == integer:
      return word
  return None


def predict_caption(model, image, tokenizer, max_length): # 여기서 image 자리는 vgg 통과해 나온 feature의 자리임
  # add start tag for generation process
  in_text = 'start' # 빈 문장 생성
  # iterate over the max length of sequence
  for i in range(max_length):
    # encode input sequence
    sequence = tokenizer.texts_to_sequences([in_text])[0] # 이거 인덱스 없으면 대괄호 하나 더 있어서 4차원이라 LSTM 이 안먹겠다고 오류남
    # pad the sequence
    sequence = pad_sequences([sequence], max_length)
    # predict next word
    yhat = model.predict([image, sequence], verbose=0) # X1 (feature) / X2 (문장)
    # get index with high probability
    yhat = np.argmax(yhat)
    # convert index to word
    word = idx_to_word(yhat, tokenizer)
    # stop if word not found
    if word is None:
        break
    # append word as input for generating next word
    in_text += " " + word
    # stop if we reach end tag
    if word == 'end':
        break
      
  return in_text

image = load_img('D:\study_data\_data/team_project\predict_img/street-g27099b4c0_1280.jpg', target_size=(224, 224))
# convert image pixels to numpy array
image = img_to_array(image)
# reshape data for model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

print('extracting features..')
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
predic_features = model.predict(image, verbose=1)

print('prediction..')
model = load_model('D:\study_data\_data/team_project\coco_dataset\model_save/best_model.h5')
y_pred = predict_caption(model, predic_features, tokenizer, max_length)
y_pred = y_pred.replace('start', '')
y_pred = y_pred.replace('end', '')
print(y_pred)


# bleu 스코어 따로 안뽑았음

# a white polar bear offering a picture of a polar bear

# 5만장에 50에포.. 강아지두마리 사진
# a black polar bear is looking out the back of a car 
# a penguin is holding a tire on a rock

# 5만장에 50에포 128배치.. 오토바이 사진
# a motorcycle parked on a street next to a street
# a person on a motorcycle with a helmet on the side

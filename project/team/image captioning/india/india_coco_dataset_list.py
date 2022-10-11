import os
import pickle
import numpy as np
import json

from sklearn.utils import shuffle
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50, ResNet101
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
'''
#================ json 파일 처리 ==================
num_examples= 80000     # 훈련에 사용할 이미지 개수

# annotation json 파일 읽기
with open('D:\study_data\_data/team_project\coco_dataset\json_files/captions_train2014.json', 'r') as f:
  annotations = json.load(f)

# caption과 image name을 vector로 저장
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
  caption = 'startseq ' + annot['caption'] + ' endseq'
  image_id = annot['image_id']
  full_coco_image_path = 'D:\study_data\_data/team_project\coco_dataset/train2014/' + 'COCO_train2014_' + '%012d.jpg' % (image_id)

  all_img_name_vector.append(full_coco_image_path)
  all_captions.append(caption)

# caption과 image name들을 섞습니다.(shuffle)
train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector, random_state=13)

# 빠른 학습을 위해서 shuffle된 set에서 처음부터 시작해서 num_examples 개수만큼만 선택합니다.
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]

print('train_captions :', len(train_captions))
print('all_captions :', len(all_captions))

pickle.dump(train_captions, open(os.path.join('D:\study_data\_data/team_project\coco_dataset\img_features', 'res101_train_captions80000.pkl'), 'wb'))
# print(train_captions[:1])
# print(img_name_vector[:1])
# ['startseq A skateboarder performing a trick on a skateboard ramp. endseq']
# ['D:\\study_data\\_data/team_project\\coco_dataset/train2014/COCO_train2014_000000324909.jpg']
# 순서대로 캡션 하나씩 매칭되어 있음
#=======================================================================================================


#================ 이미지 파일 전처리 (feature extraction) ====================
# load vgg16 model
model = ResNet101()
# restructure the model
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
# model.summary()

# extract features from image
features = []

for img_path in img_name_vector:
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
  features.append(feature) 
    

# store features in pickle
pickle.dump(features, open(os.path.join('D:\study_data\_data/team_project\coco_dataset\img_features', 'res101_features80000.pkl'), 'wb'))
print('img processing done.')


# features = [[첫번째이미지피처], [두번째이미지피처], ..., [마지막이미지피처]]
#===================================================================================================================================
'''

# features 파일 불러오기
with open(os.path.join('D:\study_data\_data/team_project\coco_dataset\img_features/', 'res101_features10000.pkl'), 'rb') as f:
  features = pickle.load(f)

#================ 캡션 파일 전처리 ====================
with open(os.path.join('D:\study_data\_data/team_project\coco_dataset\img_features/', 'res101_train_captions10000.pkl'), 'rb') as f:
  captions = pickle.load(f)

print(len(features))
print(len(captions))

# print(features[0][0])

for i in range(len(captions)):
  # convert to lowercase
  captions[i] = captions[i].lower()
  # delete digits, special chars, etc.
  captions[i] = captions[i].replace('[^A-Za-z]', '')# [A-Z] [a-z] : 각각 대문자 알파벳, 소문자 알파벳 모두를 의미
  # delete additional spaces
  captions[i] = captions[i].replace('\s+', ' ') # [ \t\n\r\f\v] 가 1번 이상 나오면 공백으로 변경
  captions[i] = captions[i].replace('.', '') # 마침표 제거

# print(captions[0])

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1 # 패딩토큰 포함

print('vacab_size:', vocab_size)

# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in captions)
print('max_len:', max_length)

#====== train_test_split ========
split = int(len(captions) * 0.90)
train_cap = captions[:split]
test_cap = captions[split:]

train_img = features[:split]
test_img = features[split:]
#================================

# print(train_cap[0])
# print(train_img[0])

# create data generator to get data in batch (avoids session crash)
def data_generator(features, captions, tokenizer, max_length, vocab_size, batch_size):
  # loop over images
  X1, X2, y = list(), list(), list()
  n = 0
  for idx, caption in enumerate(captions):
    n += 1
    # encode the sequence
    seq = tokenizer.texts_to_sequences([caption])[0]
    # split the sequence into X, y pairs
    for i in range(1, len(seq)):
      # split into input and output pairs
      in_seq, out_seq = seq[:i], seq[i] # 현재 문장을 인풋으로, 다음에 올 단어를 아웃풋으로
      # pad input sequence
      in_seq = pad_sequences([in_seq], maxlen=max_length)[0] # 최대 문장 길이만큼 패딩(0을 앞쪽에 채움)
      # encode output sequence
      out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
      
      # store the sequences
      X1.append(features[idx][0])
      X2.append(in_seq)
      y.append(out_seq)
      
    if n == batch_size: # 배치 사이즈만큼 차면 yield로 한묶음 채워서 뱉음
      X1, X2, y = np.array(X1), np.array(X2), np.array(y)
      yield [X1, X2], y
      X1, X2, y = list(), list(), list()
      n = 0


# encoder model
# image feature layers
inputs1 = Input(shape=(1000,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = Dense(256)(se2)

# decoder model
decoder1 = add([fe2, se3])
decoder2 = LSTM(256)(decoder1)
decoder3 = Dense(256, activation='relu')(decoder2)
outputs = Dense(vocab_size, activation='softmax')(decoder3)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

'''
# train the model
print('start training...')
epochs = 40
batch_size = 50
steps = len(train_cap) // batch_size # 1 batch 당 훈련하는 데이터 수

# 제너레이터 함수에서 yield로 252개의 [X1, X2], y 묶음이 차곡차곡 쌓여 있고  steps_per_epoch=steps 이 옵션으로
# epoch 1번짜리 fit을 돌때 정해준steps번 generator 를 호출함. iterating 을 steps번 함

for i in range(epochs):
  print(f'epoch: {i+1}')
  # create data generator
  generator = data_generator(train_img, train_cap, tokenizer, max_length, vocab_size, batch_size)
  # fit for one epoch
  model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1) # generator -> [X1, X2], y
print('done training.')

# save the model
model.save('D:\study_data\_data/team_project\coco_dataset\model_save/res101_model3_10000.h5')
print('model saved.')
'''

def idx_to_word(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == integer:
      return word
  return None


def predict_caption(model, image, tokenizer, max_length): # 여기서 image 자리는 vgg 통과해 나온 feature의 자리임
  # add start tag for generation process
  in_text = 'startseq' # 빈 문장 생성
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
    if word == 'endseq':
        break
      
  return in_text

image = load_img('D:\study_data\_data/team_project\predict_img/street-g27099b4c0_1280.jpg', target_size=(224, 224))
# convert image pixels to numpy array
image = img_to_array(image)
# reshape data for model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

print('extracting features..')
model = ResNet101()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
predic_features = model.predict(image, verbose=1)

print('prediction..')
model = load_model('D:\study_data\_data/team_project\coco_dataset\model_save/res101_model3_10000.h5')
y_pred = predict_caption(model, predic_features, tokenizer, max_length)
y_pred = y_pred.replace('startseq', '')
y_pred = y_pred.replace('endseq', '')
print(y_pred)


# bleu 스코어
from nltk.translate.bleu_score import corpus_bleu
# validate with test data
actual, predicted = list(), list()

for idx, caption in enumerate(test_cap):
  # print('verbose for bleu:', idx)
  # predict the caption for image
  y_pred = predict_caption(model, test_img[idx], tokenizer, max_length) 
  # split into words
  actual_captions = [caption.split() for caption in captions]
  y_pred = y_pred.split()
  # append to the list
  actual.append(actual_captions)
  predicted.append(y_pred)

# calcuate BLEU score
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))      
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))) 



# a white polar bear offering a picture of a polar bear

# 5만장에 50에포.. 강아지두마리 사진
# a black polar bear is looking out the back of a car 
# a penguin is holding a tire on a rock

# 5만장에 50에포 128배치.. 오토바이 사진
# a motorcycle parked on a street next to a street
# a person on a motorcycle with a helmet on the side

# 70에포 64배치
# a man is sitting on a motorcycle with a usa flag

# 20에포 128배치
# a man riding a motorcycle on a leather motorcycle 

# 5에포 50배치
# a man riding a motorcycle on a motorcycle with a helmet

# 40에포 50배치 1만장 vgg16
# a man rides a motor motorcycle on a street
# BLEU-1: 0.978604
# BLEU-2: 0.938410

# 40에포 50배치 1만장 vgg19
# a brightly colored clock stands in front of a building
# BLEU-1: 0.980556

import os
import pickle
import numpy as np
from tqdm.notebook import tqdm

from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.utils import to_categorical, plot_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

BASE_DIR = 'D:\study_data\_data\Flickr8k/'
WORKING_DIR = 'D:\study_data\_data\Flickr8k/working'

'''
# load vgg16 model
model = VGG16()
# restructure the model
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# summarize
# model.summary()

# extract features from image
features = {}
directory = os.path.join(BASE_DIR, 'Images')

for img_name in tqdm(os.listdir(directory)):
    # load the image from file
    img_path = directory + '/' + img_name
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
    # get image ID
    image_id = img_name.split('.')[0]
    # store feature
    features[image_id] = feature
    
# print(features)

# store features in pickle
pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))
print('img processing done.')
'''
# load features from pickle
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)
    
    
with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f) # 첫줄 빼고 읽어오기
    captions_doc = f.read()
# print(captions_doc)

# create mapping of image to captions
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    
    # if len(line) < 2: # 1단어짜리 스킵용도
    #     continue
    
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0] # . 이후 지움
    # convert caption list to string
    caption = "".join(caption)
    '''['A small child is jumping on a bed .\n']
            A small child is jumping on a bed .'''
    
    # create list if needed
    if image_id not in mapping: # key 값으로 mapping 딕셔너리 안에 현재 image_id가 없으면
        mapping[image_id] = []  # image_id:[] 형태로 새로 딕셔너리 자리 하나 만듦
    # store the caption
    mapping[image_id].append(caption) # 만든 자리에 현재 캡션 넣음
                                      # 해당 이미지 id가 이미 있으면 그 자리에 넣음
                                      # 한 이미지당 5개이므로 다음 이미지 아이디가 들어오기 전까지 한자리에 넣음


print(len(mapping))


def clean(mapping): # 맵핑 딕셔너리 안의 caption을 전처리
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc.
            caption = caption.replace('[^A-Za-z]', '')# [A-Z] [a-z] : 각각 대문자 알파벳, 소문자 알파벳 모두를 의미
            # delete additional spaces
            caption = caption.replace('\s+', ' ') # [ \t\n\r\f\v] 가 1번 이상 나오면 공백으로 변경
            # add start and end tags to the caption
            caption = 'start ' + " ".join([word for word in caption.split()]) + ' end'
            # 스페이스 기준 잘라서 넣기
            '''a child is standing on her head .
            start a child is standing on her head end'''
            captions[i] = caption.replace(' .', '')
            

# before preprocess of text
print('bf_text:', mapping['1000268201_693b08cb0e'])

# preprocess the text
clean(mapping)

# after preprocess of text
print('af_text:', mapping['1000268201_693b08cb0e'])


# 딕셔너리에서 캡션만 뽑아오기
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

        
print('all_captions_len:', len(all_captions))

print(all_captions[:3]) # 캡션 아무거나 한개 보기


# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1 # 패딩토큰 포함

print('vacab_size:', vocab_size)

# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)
print('max_len:', max_length)


image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90) # train_test_split
train = image_ids[:] # 안함
# test = image_ids[split:]

# <start> girl going into wooden building end
#        X                   y
# <start>                   girl
# <start> girl              going
# <start> girl going        into
# ...........
# <start> girl going into wooden building      end


# create data generator to get data in batch (avoids session crash)
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # loop over images
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # process each caption
            for caption in captions:
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
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0] # 이해 안감... 아오
                    
                    # store the sequences
                    X1.append(features[key][0])
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

# plot the model
plot_model(model, show_shapes=True)


# train the model
print('start training...')
epochs = 40
batch_size = 32
steps = len(train) // batch_size # 1 batch 당 훈련하는 데이터 수

for i in range(epochs):
    print(f'epoch: {i+1}')
    # create data generator
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    # fit for one epoch
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1) # generator -> [X1, X2], y
print('done training.')      
                
# save the model
model.save(WORKING_DIR+'/best_model.h5')

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'start'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
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

''' bleu score
from nltk.translate.bleu_score import corpus_bleu
# validate with test data
actual, predicted = list(), list()

for key in tqdm(test):
    # get actual caption
    captions = mapping[key]
    # predict the caption for image
    y_pred = predict_caption(model, features[key], tokenizer, max_length) 
    # split into words
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    # append to the list
    actual.append(actual_captions)
    predicted.append(y_pred)
    
# calcuate BLEU score
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))


from PIL import Image
import matplotlib.pyplot as plt
def generate_caption(image_name):
    # load the image
    # image_name = "1001773457_577c3a7d70.jpg"
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    # predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)
    plt.show()
'''

image = load_img('C:\AIA_Team_Project\Project\img_captioning\india/siberian-husky-g84d30ce80_1280.jpg', target_size=(224, 224))
# convert image pixels to numpy array
image = img_to_array(image)
# reshape data for model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

print('extracting features..')
model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
predic_feature = model.predict(image, verbose=1)

print('prediction..')
model = load_model(WORKING_DIR+'/best_model.h5')
y_pred = predict_caption(model, predic_feature, tokenizer, max_length)
y_pred = y_pred.replace('start', '')
y_pred = y_pred.replace('end', '')
print(y_pred)

# generate_caption("1001773457_577c3a7d70.jpg")
# generate_caption("1002674143_1b742ab4b8.jpg")
# generate_caption("101669240_b2d3e7f17b.jpg")
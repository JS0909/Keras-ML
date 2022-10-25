# 가위바위보, hoarse or human
# ImageDataGenerator 사용, 300x300 크기
# 이진분류 혹은 두개의 다중분류 문제로 나옴 sigmoid 혹은 softmax 사용

# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Computer vision with CNNs
#
# Create and train a classifier for horses or humans using the provided data.
# Make sure your final layer is a 1 neuron, activated by sigmoid as shown.
#
# The test will use images that are 300x300 with 3 bytes color depth so be sure to
# design your neural network accordingly

import tensorflow as tf
import urllib
import zipfile
import time
from keras.preprocessing.image import ImageDataGenerator

def solution_model():
    _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')
    local_zip = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/horse-or-human/')
    zip_ref.close()
    urllib.request.urlretrieve(_TEST_URL, 'testdata.zip')
    local_zip = 'testdata.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/testdata/')
    zip_ref.close()

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=5,
        rotation_range=5,
        zoom_range=1.2,
        shear_range=0.7,
        fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1./255)
    start_flow = time.time()
    train_generator = train_datagen.flow_from_directory(
        'tmp/horse-or-human/',
        target_size=(300, 300),
        batch_size=1500,
        class_mode='binary',
        shuffle=True)

    validation_generator = validation_datagen.flow_from_directory(
        'tmp/testdata/',
        target_size=(300, 300),
        batch_size=1500,
        class_mode='binary',
        shuffle=True)
    end_flow = time.time()

    x = train_generator[0][0]
    y = train_generator[0][1]
    
    x_val = validation_generator[0][0]
    y_val = validation_generator[0][1]
    
    model = tf.keras.models.Sequential([
        # Note the input shape specified on your first layer must be (300,300,3)
        # Your Code here
        
        tf.keras.layers.Conv2D(20, (3,3), input_shape=(300,300,3)),
        tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),        
        
        # This is the last layer. You should not change this code.
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    start = time.time()
    model.fit(x, y, epochs=20, batch_size=10)
    end = time.time()
    
    loss, acc = model.evaluate(x_val, y_val)
    print('loss:', loss, 'acc:', acc)
    print(f'img_loading took:{end_flow-start_flow:.3} sec.')
    print(f'training took:{end-start:.3} sec.')
    # loss: 5.86270809173584 acc: 0.71484375
    # img_loading took:0.0527 sec.
    # training took:92.5 sec.
              
    return model

    # NOTE: If training is taking a very long time, you should consider setting the batch size
    # appropriately on the generator, and the steps per epoch in the model.fit() function.

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
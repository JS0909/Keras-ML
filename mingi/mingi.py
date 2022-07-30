from flask import Flask, render_template, request
import os
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten , Dropout,MaxPooling2D,LSTM
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import accuracy_score

# tf.random.set_seed(9) # 하이퍼 파라미터 튜닝 용이하게 하기 위해

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'D:\study_data\_data/test/test'

#업로드 HTML 렌더링
@app.route('/')
def render_file():
   return render_template('start.html')

#파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
        f = request.files['file']
      #저장할 경로 + 파일명
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        
        season = ImageDataGenerator(
        rescale=1./255)

        season1 = season.flow_from_directory(
        'D:\study_data\_data/test/',
        target_size=(150,150),# 크기들을 일정하게 맞춰준다.
        batch_size=4000,
        class_mode='categorical', 
        # color_mode='grayscale', #디폴트값은 컬러
        shuffle=True,
        )
        print(season1[0][0])

        np.save('d:/study_data/_save/_npy/personaltest_test.npy', arr=season1[0][0])


        #1. 데이터
        season = np.load('d:/study_data/_save/_npy/personaltest_test.npy')
        x_train = np.load('d:/study_data/_save/_npy/project_train7_x.npy')
        y_train = np.load('d:/study_data/_save/_npy/project_train7_y.npy')
        x_test = np.load('d:/study_data/_save/_npy/project_test7_x.npy')
        y_test = np.load('d:/study_data/_save/_npy/project_test7_y.npy')

        print(x_train.shape)            # (2000, 150, 150, 3)
        print(y_train.shape)            # (2000,)
        print(x_test.shape)             # (550, 150, 150, 3)
        print(y_test.shape)             # (550,)

        # x_train = x_train.reshape(2000,450,150)
        # x_test = x_test.reshape(550,450,150)


        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.layers import Dense, Conv2D, Flatten , Dropout,MaxPooling2D,LSTM


        #2. 모델 
        model = Sequential()
        model.add(Conv2D(64,(3,3), input_shape = (150,150,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(2,2))
        model.add(Conv2D(128,(3,3), padding='same',activation='relu'))
        model.add(MaxPooling2D(2,2))
        model.add(Conv2D(128,(3,3), padding='same',activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100,activation='relu'))
        model.add(Dense(100,activation='relu'))
        model.add(Dense(7,activation='softmax'))
            

        #3. 컴파일.훈련

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])

        earlystopping =EarlyStopping(monitor='loss', patience=15, mode='auto', 
                    verbose=1, restore_best_weights = True)     

        hist = model.fit(x_train,y_train, epochs=50,validation_split=0.3,verbose=2,batch_size=32,
                        callbacks=[earlystopping]) 

        #4. 예측
        accuracy = hist.history['accuracy']
        val_accuracy = hist.history['val_accuracy']
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']

        print('loss : ',loss[-1])
        print('accuracy : ', accuracy[-1])

        ############################################
        loss = model.evaluate(x_test, y_test)
        y_predict = model.predict(season)

        y_test = np.argmax(y_test, axis= 1)
        y_predict = np.argmax(y_predict, axis=1) 
        print('predict : ',y_predict)

        
        if y_predict[0] == 0:
            print('hail ')
        elif  y_predict[0] ==1 :
            print('lighting')
        elif  y_predict[0] ==2 :
            print('rain')
        elif  y_predict[0] ==3 :
            print('rainbow')
        elif  y_predict[0] ==4 :
            print('sunshine')        
        elif  y_predict[0] ==5 :
            print('smog')        
        else :
            print('snow')   

        wh1 = y_predict[0]

        return render_template('end.html', wh=wh1)

    
if __name__ == '__main__':
    #서버 실행
   app.run(debug = True)

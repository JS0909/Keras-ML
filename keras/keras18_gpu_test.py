import numpy as np
import tensorflow as tf

print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus): # 만약 gpus에 뭔가 있으면 실행해라
    print('GPU 돈다')
else:
    print('GPU 안돈다')
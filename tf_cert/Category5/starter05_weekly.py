# ==============================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative to its
# difficulty. So your Category 1 question will score significantly less than
# your Category 5 question.
#
# WARNING: Do not use lambda layers in your model, they are not supported
# on the grading infrastructure. You do not need them to solve the question.
#
# WARNING: If you are using the GRU layer, it is advised not to use the
# recurrent_dropout argument (you can alternatively set it to 0),
# since it has not been implemented in the cuDNN kernel and may
# result in much longer training times.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ==============================================================================
#
# TIME SERIES QUESTION
#
# Build and train a neural network to predict the time indexed variable of
# the univariate US diesel prices (On - Highway) All types for the period of
# 1994 - 2021.
# Using a window of past 10 observations of 1 feature , train the model
# to predict the next 10 observations of that feature.
#
# ==============================================================================
#
# ABOUT THE DATASET
#
# Original Source:
# https://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_nus_w.htm#
#
# For the purpose of the examination we have used the Diesel (On - Highway) -
# All Types time series data for the period of 1994 - 2021 from the
# aforementioned link. The dataset has 1 time indexed feature.
# We have provided a cleaned version of the data.
#
# ==============================================================================
#
# INSTRUCTIONS
#
# Complete the code in following functions:
# 1. solution_model()
#
# Your code will fail to be graded if the following criteria are not met:
#
# 1. Model input shape must be (BATCH_SIZE, N_PAST = 10, N_FEATURES = 1),
#    since the testing infrastructure expects a window of past N_PAST = 10
#    observations of the 1 feature to predict the next N_FUTURE = 10
#    observations of the same feature.
#
# 2. Model output shape must be (BATCH_SIZE, N_FUTURE = 10, N_FEATURES = 1)
#
# 3. The last layer of your model must be a Dense layer with 1 neuron since
#    the model is expected to predict observations of 1 feature.
#
# 4. Don't change the values of the following constants:
#    SPLIT_TIME, N_FEATURES, BATCH_SIZE, N_PAST, N_FUTURE, SHIFT, in
#    solution_model() (See code for additional note on BATCH_SIZE).
#
# 5. Code for normalizing the data is provided - don't change it.
#    Changing the normalizing code will affect your score.
#
# 6. Code for converting the dataset into windows is provided - don't change it.
#    Changing the windowing code will affect your score.
#
# 7. Code for setting the seed is provided - don't change it.
#
# HINT: If you follow all the rules mentioned above and throughout this
# question while training your neural network, there is a possibility that a
# validation MAE of approximately 0.02 or less on the normalized validation
# dataset may fetch you top marks.


import pandas as pd
import tensorflow as tf


# This function normalizes the dataset using min max scaling.
# DO NOT CHANGE THIS CODE
def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data


# This function is used to map the time series dataset into windows of
# features and respective targets, to prepare it for training and validation.
# The first element of the first window will be the first element of
# the dataset.
#
# Consecutive windows are constructed by shifting the starting position
# of the first window forward, one at a time (indicated by shift=1).
#
# For a window of n_past number of observations of the time
# indexed variable in the dataset, the target for the window is the next
# n_future number of observations of the variable, after the
# end of the window.

# DO NOT CHANGE THIS.
def windowed_dataset(series, batch_size, n_past=10, n_future=10, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)


# This function loads the data from the CSV file, normalizes the data and
# splits the dataset into train and validation data. It also uses
# windowed_dataset() to split the data into windows of observations and
# targets. Finally it defines, compiles and trains a neural network. This
# function returns the final trained model.

# COMPLETE THE CODE IN THIS FUNCTION
def solution_model():
    # DO NOT CHANGE THIS CODE
    # Reads the dataset.
    df = pd.read_csv('Weekly_U.S.Diesel_Retail_Prices.csv',
                     infer_datetime_format=True, index_col='Week of', header=0)

    # Number of features in the dataset. We use all features as predictors to
    # predict all features of future time steps.
    N_FEATURES = len(df.columns) # DO NOT CHANGE THIS

    # Normalizes the data
    data = df.values
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    # Splits the data into training and validation sets.
    SPLIT_TIME = int(len(data) * 0.8) # DO NOT CHANGE THIS
    x_train = data[:SPLIT_TIME]
    x_valid = data[SPLIT_TIME:]

    # DO NOT CHANGE THIS CODE
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    # DO NOT CHANGE BATCH_SIZE IF YOU ARE USING STATEFUL LSTM/RNN/GRU.
    # THE TEST WILL FAIL TO GRADE YOUR SCORE IN SUCH CASES.
    # In other cases, it is advised not to change the batch size since it
    # might affect your final scores. While setting it to a lower size
    # might not do any harm, higher sizes might affect your scores.
    BATCH_SIZE = 32  # ADVISED NOT TO CHANGE THIS

    # DO NOT CHANGE N_PAST, N_FUTURE, SHIFT. The tests will fail to run
    # on the server.
    # Number of past time steps based on which future observations should be
    # predicted
    N_PAST = 10  # DO NOT CHANGE THIS

    # Number of future time steps which are to be predicted.
    N_FUTURE = 10  # DO NOT CHANGE THIS

    # By how many positions the window slides to create a new window
    # of observations.
    SHIFT = 1  # DO NOT CHANGE THIS

    # Code to create windowed train and validation datasets.
    train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)

    # Code to define your model.
    model = tf.keras.models.Sequential([

        # ADD YOUR LAYERS HERE.

        # If you don't follow the instructions in the following comments,
        # tests will fail to grade your code:
        # The input layer of your model must have an input shape of:
        # (BATCH_SIZE, N_PAST = 10, N_FEATURES = 1)
        # The model must have an output shape of:
        # (BATCH_SIZE, N_FUTURE = 10, N_FEATURES = 1).
        # Make sure that there are N_FEATURES = 1 neurons in the final dense
        # layer since the model predicts 1 feature.

        # HINT: Bidirectional LSTMs may help boost your score. This is only a
        # suggestion.

        # WARNING: If you are using the GRU layer, it is advised not to use the
        # recurrent_dropout argument (you can alternatively set it to 0),
        # since it has not been implemented in the cuDNN kernel and may
        # result in much longer training times.
        tf.keras.layers.Dense(N_FEATURES)
    ])

    # Code to train and compile the model
    optimizer =  # YOUR CODE HERE
    model.compile(
        # YOUR CODE HERE
    )
    model.fit(
        # YOUR CODE HERE
    )

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.

if __name__ == '__main__':
    model = solution_model()
    model.save("c5q12.h5")


# THIS CODE IS USED IN THE TESTER FOR FORECASTING. IF YOU WANT TO TEST YOUR MODEL
# BEFORE UPLOADING YOU CAN DO IT WITH THIS

#def model_forecast(model, series, window_size, batch_size):
#    ds = tf.data.Dataset.from_tensor_slices(series)
#    ds = ds.window(window_size, shift=1, drop_remainder=True)
#    ds = ds.flat_map(lambda w: w.batch(window_size))
#    ds = ds.batch(batch_size, drop_remainder=True).prefetch(1)
#    forecast = model.predict(ds)
#    return forecast

# PASS THE NORMALIZED data IN THE FOLLOWING CODE

# rnn_forecast = model_forecast(model, data, N_PAST, BATCH_SIZE)
# rnn_forecast = rnn_forecast[SPLIT_TIME - N_PAST:-1, 0, 0]

# x_valid = np.squeeze(x_valid[:rnn_forecast.shape[0]])
# result = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()

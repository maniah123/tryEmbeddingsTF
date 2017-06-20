# <editor-fold desc="Manage Imports">
import tensorflow as tf
import numpy as np
import os
import math
import matplotlib.pyplot as plt
# </editor-fold>

# <editor-fold desc="Get / Prep Data">
#Get/Create Data
numUniqueStages = 4
numDataPoints = 1000
seq_len = 10
numRows, numCols = int(numDataPoints/(seq_len+1)), seq_len+1
stages = np.random.random_integers(low=1, high=numUniqueStages, size=(numRows, numCols))
print(stages[0:5,:])
print(stages.shape)

batch_size = 20
embedDim = 5
hid_dims = [3]

#Make sure the stages are indexed starting 0
minStageValue = np.min(stages)
if(minStageValue != 0): stages = stages - minStageValue
print(stages[0:5,:])
# </editor-fold>

Raw_In = tf.placeholder(name='raw_input_X', dtype=tf.int32, shape=[None, seq_len+1])
OneHot_X = tf.one_hot(name='one_hot_input_X', indices=Raw_In, depth=numUniqueStages)
Y = tf.placeholder(name='raw_input_Y', dtype=tf.int32, shape=[None])
OneHot_Y = tf.one_hot(name='one_hot_input_Y', indices=Y, depth=numUniqueStages)
W_Emb = tf.Variable(initial_value=tf.truncated_normal(shape=[numUniqueStages, embedDim]), dtype=tf.float32)
W_Ho = tf.Variable(initial_value=tf.truncated_normal(shape=[embedDim, numUniqueStages]), dtype=tf.float32)
B_Ho = tf.Variable(initial_value=0.01, dtype=tf.float32)
X = tf.unstack(OneHot_X, axis=1)
rnn_inputs = [tf.matmul(x, W_Emb) for x in X]
lstmCell_1 = tf.contrib.rnn.LSTMCell(num_units = embedDim, activation=tf.tanh)
temp_outputs, states = tf.contrib.rnn.static_rnn(cell=lstmCell_1, inputs=rnn_inputs, dtype=tf.float32)
final_output = tf.sigmoid(tf.add(tf.matmul(temp_outputs[-1], W_Ho), B_Ho))

loss = tf.losses.softmax_cross_entropy(logits=final_output, onehot_labels=OneHot_Y)

train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

with tf.Session() as S:
    S.run(tf.global_variables_initializer())
    rawinput_X, rawinput_Y = stages[0:batch_size,:], stages[0:batch_size,-1]
    rawIn, onehotIn, xInput, rnnInputs, tempOutputs, finalOuput, finalLoss = S.run([Raw_In, OneHot_X, X, rnn_inputs, temp_outputs,
                                                                         final_output, loss],
                                               feed_dict={Raw_In: rawinput_X, Y:rawinput_Y})
    print('Shape of rawIn =', rawIn.shape)
    print('Shape of onehotIn =', onehotIn.shape)
    # print('Raw In ---->')
    # #print(rawIn[0:2,:])
    # print('')
    # print('One Hot In ---->')
    # #print(onehotIn[0:2,:])
    print('Length and Shape of first xInput =', len(xInput), ': ', xInput[0].shape )
    print('Shape of rnnInputs =', rnnInputs[0].shape)
    print('type of tempOutputs =',type(tempOutputs))
    print('Length and Shape of first tempOutputs =', len(tempOutputs), ': ', tempOutputs[0].shape)
    print('Shape of final Output =', finalOuput.shape)
    print('final loss =', finalLoss)

    feedDict = {Raw_In: stages[0:batch_size,:], Y:stages[0:batch_size,-1]}
    initLoss = S.run(loss, feed_dict=feedDict)
    for i in range(20):
        _ = S.run(train_op, feed_dict=feedDict)
    postLoss = S.run(loss, feed_dict=feedDict)

    print('Init Loss =', initLoss)
    print('Final Loss =', postLoss)
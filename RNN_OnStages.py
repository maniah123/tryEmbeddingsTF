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

#Make sure the stages are indexed starting 0
minStageValue = np.min(stages)
if(minStageValue != 0): stages = stages - minStageValue
print(stages[0:5,:])
# </editor-fold>

Raw_In = tf.placeholder(name='raw_input', dtype=tf.int32, shape=[None, seq_len+1])
OneHot_In = tf.one_hot(name='one_hot_input', indices=Raw_In, depth=numUniqueStages)


with tf.Session() as S:
    rawinput = stages[0:3,:]
    rawIn, onehotIn = S.run([Raw_In, OneHot_In], feed_dict={Raw_In: rawinput})
    print('Shape of rawIn =', rawIn.shape)
    print('Shape of onehotIn =', onehotIn.shape)
    print('Raw In ---->')
    print(rawIn[0:2,:])
    print('')
    print('One Hot In ---->')
    print(onehotIn[0:2,:])
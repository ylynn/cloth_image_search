from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils

import numpy as np
import cPickle

from skimage import transform, io

size_im = 224;
# Preprocessing
f_data = open('E:/zlp/sketch_cnn/a.pickle', 'r')
dic = cPickle.load(f_data)
f_data.close()

llab = dic['llab']

cat = [np_utils.to_categorical(llab[:, k] - 1, max(llab[:, k])) for k in range(llab.shape[1])]
new_llab = np.concatenate(cat, axis=1)
n = new_llab.shape[0]

X = np.zeros((n, 3, size_im, size_im))
for i in range(n):
    im = io.imread("../im_cmap/%d.png" % (i + 1, ), as_grey=False)
    X[i, :, :, :] = im.astype('float64') / 255

# Shuffling
shuffler = np.arange(8989)
np.random.shuffle(shuffler)
X = X[shuffler]
new_llab = new_llab[shuffler]

# 80% training data
X_train, llab_train = X[:7200], new_llab[:7200]
X_test, llab_test = X[7200:], new_llab[7200:]
# Network defination
def train_net():
    
    f_w = file('E:/zlp/TBbenchmark/test.pickle', 'r')
    weights = cPickle.load(f_w)
    f_w.close()
    
    model = Sequential()
    model.add(Convolution2D(64, 11, 11, border_mode="valid", weights=[weights[0][0][0][2][0][0].transpose((3, 2, 0, 1)), weights[0][0][0][2][0][1].ravel()], subsample=(4, 4), input_shape=(3, size_im, size_im)))
    model.add(Activation('relu'))
    # model.add(LRN2D(k=1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same"))

    model.add(Convolution2D(256, 5, 5, border_mode="same", weights=[weights[4][0][0][2][0][0].transpose((3, 2, 0, 1)), weights[4][0][0][2][0][1].ravel()]))
    model.add(Activation('relu'))
    # model.add(LRN2D(k=1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same"))

    model.add(Convolution2D(256, 3, 3, border_mode="same", weights=[weights[8][0][0][2][0][0].transpose((3, 2, 0, 1)), weights[8][0][0][2][0][1].ravel()]))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(256, 3, 3, border_mode="same", weights=[weights[10][0][0][2][0][0].transpose((3, 2, 0, 1)), weights[10][0][0][2][0][1].ravel()]))
    model.add(Activation('relu'))
    
    model.add(Convolution2D(256, 3, 3, border_mode="same", weights=[weights[12][0][0][2][0][0].transpose((3, 2, 0, 1)), weights[12][0][0][2][0][1].ravel()]))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same"))

    model.add(Convolution2D(4096, 6, 6, border_mode="valid", weights=[weights[15][0][0][2][0][0].transpose((3, 2, 0, 1)), weights[15][0][0][2][0][1].ravel()]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(4096, 1, 1, border_mode="valid", weights=[weights[18][0][0][2][0][0].transpose((3, 2, 0, 1)), weights[18][0][0][2][0][1].ravel()]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    
    model.add(Dense(34, activation='sigmoid'))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    return model

model = train_net()

model.fit(X_train, llab_train, batch_size=50, nb_epoch=50, shuffle=True, verbose=1, show_accuracy=True)

model.save_weights('weights.hdf5')

# testing
res_test = model.predict(X_test)
tags = np.max(llab, axis=0)
res_test_llab = None
s = 0

for k in tags:
    if res_test_llab == None:
        res_test_llab = np.argmax(res_test[:, s:s+k], axis=1)[:, np.newaxis]
    else:
        res_test_llab = np.concatenate((res_test_llab, np.argmax(res_test[:, s:s+k], axis=1)[:, np.newaxis]), axis=1)
    s += k
    
res_test_llab += 1
truth = llab[shuffler][7200:]
mat = res_test_llab == truth
sum_mat = np.sum(mat, axis=0)
percentage_mat = sum_mat * 1. / 1789
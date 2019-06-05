import tensorflow as tf
import tensorflow_probability as tfp
from tf.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from tf.keras.models import Model
from tf.keras.callbacks import TensorBoard, Callback
import scipy.io as sio 
import numpy as np
import math
import time
tf.reset_default_graph()

envir = 'indoor' #'indoor' or 'outdoor'
# image params
img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels
# network params
residual_num = 2
encoded_dim = 512  #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32

# Bulid the autoencoder model of CsiNet

# change the position of channel in x_train, to indicate the real/imag

def residual_network(x, residual_num, encoded_dim):
    
    def weight_variable(shape, scale=0.1):
        # first half of the weight is the real part and the second part is the imaginary part
        # original input shape [batch_size, height, width, depth]
        # new input shape [part, batch_size, height, width, depth] where part is used to distinguish real and imaginary part
  
  
        magnitude = tf.convert_to_tensor(tf.cast(tfp.math.random_rayleigh(scale=scale, shape=shape), dtype=tf.float32))
        phase = tf.random_uniform(tf.convert_to_tensor(shape), minval=-pi, maxval=pi)
        #initial is the initial weights, part=0 refers to the real part, part=1 refers to the imaginary part
        out_real = tf.multiply(magnitude, tf.cos(phase))
        out_imag = tf.multiply(magnitude, tf.sin(phase))
  
        initial = tf.stack([out_real, out_imag])
        return initial
    
    def complex_conv(x, out_channel, filter_size, stride=1,name="conv"):
        with tf.variable_scope(name):
            # number of channel in the input x
            in_channel = x.get_shape().as_list()[-1] #get_shape returns a tuple and needed to be converted to a list
            # shape of the weight and bias
            shape = [filter_size, filter_size, in_channel, out_channel]
            # create weight variable
            sigma = 1/np.sqrt(filter_size**2*(in_channel+out_channel))
            initial = weight_variable(shape, scale=sigma)
            w = tf.get_variable('w', initializer = initial)
            # create bias variable
            b = tf.get_variable('biases', [out_channel], initializer=tf.constant_initializer(0.0))
  
            strides = [1, stride, stride, 1]
  
            output_real = tf.nn.conv2d(x[0,:], w[0,:], strides, "SAME") - tf.nn.conv2d(x[1,:], w[1,:], strides, "SAME")
            output_imag = tf.nn.conv2d(x[0,:], w[1,:], strides, "SAME") + tf.nn.conv2d(x[1,:], w[0,:], strides, "SAME")
            conv = tf.stack([output_real, output_imag])
  
        return tf.nn.bias_add(conv,b)
    
    
    def com_full_layer(x, neurons,name="dense"):
        with tf.variable_scope(name):
        
            sigma = 1/np.sqrt(np.prod(x.get_shape().as_list()[2:]))
            initial = weight_variable([x.get_shape().as_list()[2],neurons], scale=sigma)
            w = tf.get_variable('w', initializer = initial)
            b = tf.get_variable('b', [2,x.get_shape().as_list()[1],neurons],initializer=tf.constant_initializer(0.))
  
            out_real = tf.matmul(x[0,:],w[0,:]) - tf.matmul(x[1,:],w[1,:])
            out_imag = tf.matmul(x[0,:],w[1,:]) + tf.matmul(x[1,:],w[0,:])
  
        return tf.stack([out_real, out_imag])+b
    
    def complex_BN(x,name='BN'):
        with tf.variable_scope(name):
            half_channel = x.get_shape()[0]
            
            gamma_rr = tf.get_variable(name='gamma_rr',initializer=tf.convert_to_tensor(1/np.sqrt(2)))
            gamma_ii = tf.get_variable(name='gamma_rr',initializer=tf.convert_to_tensor(1/np.sqrt(2)))
            gamma_ri = tf.get_variable(name='gamma_rr',initializer=tf.convert_to_tensor(0))
            
            x_real = gamma_rr*x[0:half_channel,:]+gamma_ri*x[half_channel:,:]
            x_imag = gamma_ri*x[0:half_channel,:]+gamma_ii*x[half_channel:,:]
            com_x = tf.concat([x_real,x_imag],axis=0)
            b = tf.get_variable('bias',shape=com_x.get_shape(),initializer=tf.constant_initializer(0.))
            return com_x+b
    
    def add_common_layers(y,name='common_layer'):
        y = complex_BN(y,name)
        y = LeakyReLU()(y)
        return y
    def residual_block_decoded(y):
        shortcut = y
        y = Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = add_common_layers(y,'l_1')
        
        y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = add_common_layers(y,'l_2')
        
        y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = complex_BN(y)

        y = add([shortcut, y])
        y = LeakyReLU()(y)

        return y
    
    #x = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(x)
    x = complex_conv(x, 2, 3)
    x = add_common_layers(x,'l_in')
    
    
    x = Reshape((img_total,))(x)
    encoded = Dense(encoded_dim, activation='linear')(x)
    
    x = Dense(img_total, activation='linear')(encoded)
    x = Reshape((img_channels, img_height, img_width,))(x)
    for i in range(residual_num):
        x = residual_block_decoded(x)
    
    x = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)

    return x

image_tensor = Input(shape=(img_channels, img_height, img_width))
network_output = residual_network(image_tensor, residual_num, encoded_dim)
autoencoder = Model(inputs=[image_tensor], outputs=[network_output])
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())

# Data loading
if envir == 'indoor':
    mat = sio.loadmat('data/DATA_Htrainin.mat') 
    x_train = mat['HT'] # array
    mat = sio.loadmat('data/DATA_Hvalin.mat')
    x_val = mat['HT'] # array
    mat = sio.loadmat('data/DATA_Htestin.mat')
    x_test = mat['HT'] # array

elif envir == 'outdoor':
    mat = sio.loadmat('data/DATA_Htrainout.mat') 
    x_train = mat['HT'] # array
    mat = sio.loadmat('data/DATA_Hvalout.mat')
    x_val = mat['HT'] # array
    mat = sio.loadmat('data/DATA_Htestout.mat')
    x_test = mat['HT'] # array

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
x_val = np.reshape(x_val, (len(x_val), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))  # adapt this if using `channels_first` image data format

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        self.losses_train.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses_val.append(logs.get('val_loss'))
        

history = LossHistory()
file = 'CsiNet_'+(envir)+'_dim'+str(encoded_dim)+time.strftime('_%m_%d')
path = 'result/TensorBoard_%s' %file

autoencoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=200,
                shuffle=True,
                validation_data=(x_val, x_val),
                callbacks=[history,
                           TensorBoard(log_dir = path)])

filename = 'result/trainloss_%s.csv'%file
loss_history = np.array(history.losses_train)
np.savetxt(filename, loss_history, delimiter=",")

filename = 'result/valloss_%s.csv'%file
loss_history = np.array(history.losses_val)
np.savetxt(filename, loss_history, delimiter=",")

#Testing data
tStart = time.time()
x_hat = autoencoder.predict(x_test)
tEnd = time.time()
print ("It cost %f sec" % ((tEnd - tStart)/x_test.shape[0]))

# Calcaulating the NMSE and rho
if envir == 'indoor':
    mat = sio.loadmat('data/DATA_HtestFin_all.mat')
    X_test = mat['HF_all']# array

elif envir == 'outdoor':
    mat = sio.loadmat('data/DATA_HtestFout_all.mat')
    X_test = mat['HF_all']# array

X_test = np.reshape(X_test, (len(X_test), img_height, 125))
x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)
x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)
x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257-img_width))), axis=2), axis=2)
X_hat = X_hat[:, :, 0:125]

n1 = np.sqrt(np.sum(np.conj(X_test)*X_test, axis=1))
n1 = n1.astype('float64')
n2 = np.sqrt(np.sum(np.conj(X_hat)*X_hat, axis=1))
n2 = n2.astype('float64')
aa = abs(np.sum(np.conj(X_test)*X_hat, axis=1))
rho = np.mean(aa/(n1*n2), axis=1)
X_hat = np.reshape(X_hat, (len(X_hat), -1))
X_test = np.reshape(X_test, (len(X_test), -1))
power = np.sum(abs(x_test_C)**2, axis=1)
power_d = np.sum(abs(X_hat)**2, axis=1)
mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)

print("In "+envir+" environment")
print("When dimension is", encoded_dim)
print("NMSE is ", 10*math.log10(np.mean(mse/power)))
print("Correlation is ", np.mean(rho))
filename = "result/decoded_%s.csv"%file
x_hat1 = np.reshape(x_hat, (len(x_hat), -1))
np.savetxt(filename, x_hat1, delimiter=",")
filename = "result/rho_%s.csv"%file
np.savetxt(filename, rho, delimiter=",")


import matplotlib.pyplot as plt
'''abs'''
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display origoutal
    ax = plt.subplot(2, n, i + 1 )
    x_testplo = abs(x_test[i, 0, :, :]-0.5 + 1j*(x_test[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    decoded_imgsplo = abs(x_hat[i, 0, :, :]-0.5 
                          + 1j*(x_hat[i, 1, :, :]-0.5))
    plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
plt.show()


# save
# serialize model to JSON
model_json = autoencoder.to_json()
outfile = "result/model_%s.json"%file
with open(outfile, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
outfile = "result/model_%s.h5"%file
autoencoder.save_weights(outfile)

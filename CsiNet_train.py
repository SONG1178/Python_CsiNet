import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.layers import Lambda
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU,multiply, dot, subtract
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.models import Model
from tensorflow.keras.backend import variable
from tensorflow.keras.callbacks import TensorBoard, Callback
import scipy.io as sio 
import numpy as np
import math as m
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
    
    def w_init(shape, scale=0.1):
        # first half of the weight is the real part and the second part is the imaginary part
        # original input shape [batch_size, height, width, depth]
        # new input shape [part, batch_size, height, width, depth] where part is used to distinguish real and imaginary part
        pi = tf.constant(m.pi)
        magnitude =np.random.rayleigh(scale=scale, size=shape)
        phase = np.random.uniform(low=-pi, high=pi, size=shape,dtype=float)
        #initial is the initial weights, part=0 refers to the real part, part=1 refers to the imaginary part   
        out_real = np.multiply(magnitude, m.cos(phase))
        out_imag = np.multiply(magnitude, m.sin(phase))
        w_real = variable(value=out_real)
        w_imag = variable(value=out_imag)
  
        return w_real, w_imag
    
    def complex_conv(xr, xi, out_channel, filter_size, stride=1,name="conv"):
        with tf.variable_scope(name):
            # number of channel in the input x
            in_channel = xr.get_shape().as_list()[1] #get_shape returns a tuple and needed to be converted to a list
            # shape of the weight and bias
            shape = [filter_size, filter_size, in_channel, out_channel]
            # create weight variable
            sigma = 1/np.sqrt(filter_size**2*(in_channel+out_channel))
            w_real,w_imag = w_init(shape, scale=sigma)
            #wr = tf.get_variable('w_real', initializer = w_real)
            #wi = tf.get_variable('w_imag', initializer = w_imag)
            # create bias variable
            
            
  
            output_real = subtract([Conv2d(out_channel, padding='same', data_format='channel_first', kernel_initializer=wr)(xr), Conv2d(out_channel, padding='same', data_format='channel_first', kernel_initializer=wi)(xi)])
            output_imag = add([Conv2d(out_channel, padding='same', data_format='channel_first', kernel_initializer=wi)(xr), Conv2d(out_channel, padding='same', data_format='channel_first', kernel_initializer=wr)(xi)])
            
            #dimension = output_real.get_shape().as_list()[-1]
            #b_real = tf.get_variable('biases_real', [dimension], initializer=tf.constant_initializer(0.0))
            #b_imag = tf.get_variable('biases_imag', [dimension], initializer=tf.constant_initializer(0.0))
        return output_real, output_imag
  
        #return tf.nn.bias_add(output_real,b_real), tf.nn.bias_add(output_imag,b_imag)
    
    
    def com_full_layer(xr, xi, neurons,name="dense"):
        with tf.variable_scope(name):        
            sigma = 1/np.sqrt(np.prod(x.get_shape().as_list()[2:]))
            w_real, w_imag = w_init([xr.get_shape().as_list()[1],neurons], scale=sigma)
            #wr = tf.get_variable('w_real', initializer = w_real)
            #wi = tf.get_variable('w_imag', initializer = w_imag)
            
            out_real = subtract([dot([xr,wr]),dot([xi,wi])])
            out_imag = add([dot([xr,wi]),dot(xi,wr)])
            
            b_real = variable(value=np.zeros(out_real.get_shape()))
            b_imag = variable(value=np.zeros(out_real.get_shape()))
            
  
        return add([out_real,b_real]), add([out_imag,b_imag])
    
    def complex_BN(xr, xi, name='BN'):
        with tf.variable_scope(name):
            half_channel = x.get_shape()[1]
            
            gamma_rr = variable(value=np.full(shape=xr.get_shape(),fill_value=1/np.sqrt(2)))
            gamma_ii = variable(value=np.full(shape=xr.get_shape(),fill_value=1/np.sqrt(2)))
            gamma_ri = variable(value=np.zeros(shape=xr.get_shape()))
            
            x_real = add([multiply([gamma_rr,xr]),multiply([gamma_ri,xi])])
            x_imag = add([multiply([gamma_ri,xr]),multiply([gamma_ii,xi])])
            
            dimension = x_real.get_shape().as_list()[-1]
            b_real = variable(value=np.zeros(out_real.get_shape()))
            b_imag = variable(value=np.zeros(out_real.get_shape()))
            
            return add([x_real,b_real]), add([x_imag,b_imag])
    
    def add_common_layers(yr, yi, name='common_layer'):
        yr,yi = Lambda(complex_BN,arguments={'xi':yi, 'name':name})(yr)
        yr = LeakyReLU()(yr)
        yi = LeakyReLU()(yi)
        return yr, yi
    def residual_block_decoded(y,name='residual_block'):
        with tf.variable_scope(name):
            shortcut = y
            yr = tf.expand_dims(y[:,0,:,:],1)
            yi = tf.expand_dims(y[:,1,:,:],1)
            
            #yr, yi = complex_conv(yr, yi, 4, 3,name='conv_1')
            yr,yi = Lambda(complex_conv,arguments={'xi':yi,'out_channel':4,'filter_size':3,'name':'conv_1'})(yr)
            yr, yi = add_common_layers(yr, yi, 'l_1')
        
            #yr, yi = complex_conv(yr, yi, 8, 3,name='conv_2')
            yr,yi = Lambda(complex_conv,arguments={'xi':yi,'out_channel':8,'filter_size':3,'name':'conv_2'})(yr)
            yr, yi = add_common_layers(yr, yi,'l_2')
        
            #yr, yi = complex_conv(yr, yi, 1, 3,name='conv_3')
            yr,yi = Lambda(complex_conv,arguments={'xi':yi,'out_channel':1,'filter_size':3,'name':'conv_3'})(yr)
            yr, yi = Lambda(complex_BN, arguments={'xi':yi})(yr)
            y = tf.keras.layers.concatenate([yr,yi], axis=1)

            y = add([shortcut, y])
            y = LeakyReLU()(y)

        return y
    
    #x = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(x)
    x_real = tf.expand_dims(x[:,0,:,:],1)
    x_imag = tf.expand_dims(x[:,1,:,:],1)
    x_real,xi_imag = Lambda(complex_conv,arguments={'xi':x_imag,'out_channel':1,'filter_size':3})(x_real)
    #x_real, x_imag = complex_conv(x_real, x_imag, 1, 3)
    xr, xi = add_common_layers(x_real, x_imag,'l_in')
    
    
    xr = Reshape((img_total//2,))(xr)
    xi = Reshape((img_total//2,))(xi)
    encoded_real, encoded_imag = Lambda(com_full_layer,arguments={'xi':xi,'neurons':encoded_dim,'name':'encoder'})(xr)
    
    xr, xi = Lambda(com_full_layer,arguments={'xi':encoded_imag,'neurons': img_total//2,'name':'decoder'})(encoded_real)
    xr = Reshape((img_channels//2, img_height, img_width,))(xr)
    xi = Reshape((img_channels//2, img_height, img_width,))(xi)

    x = residual_block_decoded(x,name='first_decoder')
    x = residual_block_decoded(x,name='second_decoder')
    
    xr = tf.expand_dims(x[:,0,:,:],1)
    xi = tf.expand_dims(x[:,1,:,:],1)
    xr,xi = Lambda(complex_conv,arguments={'xi':xi,'out_channel':1,'filter_size':3,'name':'output'})(xr)
    

    xr = sigmoid(xr)
    xi = sigmoid(xi)

    x = tf.keras.layers.concatenate([xr,xi], axis=1)
    

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

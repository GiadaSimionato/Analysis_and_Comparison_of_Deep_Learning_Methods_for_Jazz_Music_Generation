# Keras implementation of the WaveNet network presented in van den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner, N., Senior, A., Kavukcuoglu, K. (2016) WaveNet: A Generative Model for Raw Audio. Proc. 9th ISCA Speech Synthesis Workshop, 125-125.

from keras.layers import Activation, Add, Conv1D, Dense, Flatten, Input, Multiply
from keras.models import Model
import numpy as np

def ResidualBlock(res_input, n_filters, kernel_size, dilation):

    dconv_sigmoid = Conv1D(n_filters, kernel_size, padding="causal", dilation_rate=dilation, activation="sigmoid") (res_input)
    dconv_tanh = Conv1D(n_filters, kernel_size, padding="causal", dilation_rate=dilation, activation="tanh")(res_input)
    mult = Multiply()([dconv_sigmoid, dconv_tanh])
    skip_conn = Conv1D(1, 1)(mult)
    residual = Add()([res_input, skip_conn])
    return [residual, skip_conn]


def waveNet(input_size, n_filters, kernel_size, n_resBlocks, output_size):
    
    skip_connections = []
    input_l = Input(batch_shape=(None, input_size, 1))
    i_residual = Conv1D(n_filters, kernel_size, padding="causal")(input_l)
    for i in range(n_resBlocks):
        i_residual, i_skip_conn = ResidualBlock(i_residual, n_filters, kernel_size, 2 ** i)
        skip_connections.append(i_skip_conn)
    sum_l = Add()(skip_connections)
    sum_l = Activation("relu")(sum_l)
    conv = Conv1D(1, 1, activation="relu")(sum_l)
    conv = Conv1D(1, 1)(conv)
    flatten = Flatten()(conv)
    output_l = Dense(output_size, activation="softmax")(flatten)

    model = Model(inputs=[input_l], outputs=[output_l])
    model.compile(loss="categorical_crossentropy", optimizer = "adam", metrics=["accuracy"])

    return model

def waveNet(input_size, n_residual_filters, kernel_residual_size, n_resBlocks, output_size, n_conv_filters=None, kernel_conv_size=None, loss="categorical_crossentropy", optimizer="adam", threshold_dilation=512):

    if n_conv_filters == None:
        n_conv_filters = n_residual_filters
    if kernel_conv_size == None:
        kernel_conv_size = kernel_residual_size
    skip_connections = []
    input_l = Input(batch_shape=(None, input_size, 1))
    i_residual = Conv1D(n_conv_filters, kernel_conv_size, padding="causal")(input_l)
    for i in range(n_resBlocks):
        i_residual, i_skip_conn = ResidualBlock(i_residual, n_residual_filters, kernel_residual_size, 2 ** (i%(np.log(threshold_dilation, 2)+1)))
        skip_connections.append(i_skip_conn)
    sum_l = Add()(skip_connections)
    sum_l = Activation("relu")(sum_l)
    conv = Conv1D(1, 1, activation="relu")(sum_l)
    conv = Conv1D(1, 1)(conv)
    flatten = Flatten()(conv)
    output_l = Dense(output_size, activation="softmax")(flatten)

    model = Model(inputs=[input_l], outputs=[output_l])
    model.compile(loss, optimizer, metrics=["accuracy"])

    return model
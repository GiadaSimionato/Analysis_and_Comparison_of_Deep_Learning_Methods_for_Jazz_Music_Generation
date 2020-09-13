from keras.layers import Activation, Add, Conv1D, Dense, Flatten, Input, Multiply, BatchNormalization, LSTM, Dropout
from keras.models import Model
from keras import regularizers
import math


def ResidualBlock(res_input, n_filters, kernel_size, dilation):

    dconv_sigmoid = Conv1D(n_filters, kernel_size, padding="causal", dilation_rate=dilation, kernel_regularizer=regularizers.l2(1e-4)) (res_input)
    dconv_sigmoid = BatchNormalization()(dconv_sigmoid)
    dconv_sigmoid = Activation("sigmoid")(dconv_sigmoid)
    dconv_tanh = Conv1D(n_filters, kernel_size, padding="causal", dilation_rate=dilation, activation="tanh", kernel_regularizer=regularizers.l2(1e-4))(res_input)
    dconv_tanh = BatchNormalization()(dconv_tanh)
    dconv_tanh = Activation("tanh")(dconv_tanh)
    mult = Multiply()([dconv_sigmoid, dconv_tanh])
    skip_conn = Conv1D(1, 1, kernel_regularizer=regularizers.l2(1e-4))(mult)
    residual = Add()([res_input, skip_conn])
    return residual, skip_conn


def waveNet(input_size, n_residual_filters, kernel_residual_size, n_resBlocks, output_size, n_conv_filters=None, kernel_conv_size=None, loss="categorical_crossentropy", optimizer="adam", threshold_dilation=512):

    if n_conv_filters == None:
        n_conv_filters = n_residual_filters
    if kernel_conv_size == None:
        kernel_conv_size = kernel_residual_size
    skip_connections = []
    input_l = Input(batch_shape=(None, input_size, 1))
    i_residual = Conv1D(n_conv_filters, kernel_conv_size, padding="causal", kernel_regularizer=regularizers.l2(1e-4))(input_l)
    i_residual = BatchNormalization()(i_residual)
    i_residual = Activation("relu")(i_residual)
    for i in range(n_resBlocks):
        i_residual, i_skip_conn = ResidualBlock(i_residual, n_residual_filters, kernel_residual_size, int(2 ** (i%(math.log(threshold_dilation, 2)+1))))
        skip_connections.append(i_skip_conn)
    sum_l = Add()(skip_connections)
    sum_l = Activation("relu")(sum_l)
    conv = Conv1D(1, 1, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(sum_l)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    conv = Conv1D(1, 1, kernel_regularizer=regularizers.l2(1e-4))(conv)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)
    flatten = Flatten()(conv)
    output_l = Dense(output_size, activation="softmax", kernel_regularizer=regularizers.l2(1e-4))(flatten)

    model = Model(inputs=[input_l], outputs=[output_l])
    model.compile(optimizer, loss, metrics=["accuracy"])

    return model


def lstmNet(input_size, hidden_units, dropout_rate, dense_units, output_size, optimizer, loss):

    l_input = Input(batch_shape=(None, input_size, 1))
    l_lstm = LSTM(units=hidden_units, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True)(l_input)
    l_lstm = LSTM(units=hidden_units, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True)(l_lstm)
    l_lstm = LSTM(units=hidden_units, activation='tanh')(l_lstm)
    l_flatten = Flatten()(l_lstm)
    l_dense = Dense(units=dense_units, activation='relu')(l_flatten)
    l_dense = Dropout(rate=dropout_rate)(l_dense)
    l_output = Dense(units=output_size, activation='softmax')(l_dense)

    model = Model(inputs=[l_input], outputs=[l_output])
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    return model
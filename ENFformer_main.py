from tensorflow import keras
from keras import layers
from keras.layers import Dense, Dropout, Flatten, Conv2D, Concatenate, Activation, add, LSTM, Bidirectional
import time
from sklearn.metrics import confusion_matrix, classification_report
import os
import tensorflow as tf
import keras
import numpy as np
import shutil
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

def get_data(data_file):
    data_all = []
    with open(data_file, "r") as f:
        for line in f.readlines():
            data = line.split('\n\t')
            for str in data:
                sub_str = str.split(',')
            if sub_str:
                data_all.append(sub_str)
    f.close()
    data_all = np.array(data_all)
    data_all = data_all.astype(np.float64)
    data_all = data_all

    print(data_all.shape)
    return  data_all

# 分类问题的类数，fc层的输出单元个数
NUM_CLASSES = 2
# 更新中心的学习率
ALPHA = 0.2
# center-loss的系数
LAMBDA = 0.0005555

def center_loss1(labels, features, alpha=ALPHA, num_classes=NUM_CLASSES):
    """
    获取center loss及更新样本的center
    :param labels: Tensor,表征样本label,非one-hot编码,shape应为(batch_size,).
    :param features: Tensor,表征样本特征,最后一个fc层的输出,shape应该为(batch_size, num_classes).
    :param alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
    :param num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
    :return: Tensor, center-loss， shape因为(batch_size,)
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，如果labels已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)

    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features

    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    # 更新centers
    centers_update_op = tf.scatter_sub(centers, labels, diff)

    # 这里使用tf.control_dependencies更新centers
    with tf.control_dependencies([centers_update_op]):
        # 计算center-loss
        c_loss = tf.nn.l2_loss(features - centers_batch)

    return c_loss

def center_loss2(labels, features, alpha=ALPHA, num_classes=NUM_CLASSES):
    """
    获取center loss及更新样本的center
    :param labels: Tensor,表征样本label,非one-hot编码,shape应为(batch_size,).
    :param features: Tensor,表征样本特征,最后一个fc层的输出,shape应该为(batch_size, num_classes).
    :param alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
    :param num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
    :return: Tensor, center-loss， shape因为(batch_size,)
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers2 = tf.get_variable('centers2', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，如果labels已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers2, labels)

    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features

    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    # 更新centers
    centers_update_op2 = tf.scatter_sub(centers2, labels, diff)

    # 这里使用tf.control_dependencies更新centers
    with tf.control_dependencies([centers_update_op2]):
        # 计算center-loss
        c_loss2 = tf.nn.l2_loss(features - centers_batch)

    return c_loss2

def center_loss3(labels, features, alpha=ALPHA, num_classes=NUM_CLASSES):
    """
    获取center loss及更新样本的center
    :param labels: Tensor,表征样本label,非one-hot编码,shape应为(batch_size,).
    :param features: Tensor,表征样本特征,最后一个fc层的输出,shape应该为(batch_size, num_classes).
    :param alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
    :param num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
    :return: Tensor, center-loss， shape因为(batch_size,)
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers3 = tf.get_variable('centers3', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，如果labels已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers3, labels)

    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features

    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    # 更新centers
    centers_update_op3 = tf.scatter_sub(centers3, labels, diff)

    # 这里使用tf.control_dependencies更新centers
    with tf.control_dependencies([centers_update_op3]):
        # 计算center-loss
        c_loss3 = tf.nn.l2_loss(features - centers_batch)

    return c_loss3

def test_model_and_find_top_acc(x_test_DFT0,x_test_DFT1,x_test_Hilbert, y_testa,all_modelpath,save_num,pad_mode,fram_num,fram_len,modell="model0"):
    path = all_modelpath
    save_num = save_num
    filename_list = os.listdir(path)

    # 遍历所有保存的模型
    res = []
    moidel_fin_save_dir = modell + pad_mode + "_" + str(fram_num) + "_" + str(fram_len)  # 当前文件夹下一个文件夹
    print("save：", moidel_fin_save_dir)
    if not os.path.exists(moidel_fin_save_dir):  # 若该文件夹不存在 则创建一个
        os.mkdir(moidel_fin_save_dir)
    for i in range(len(filename_list)):
        saved_model = path + "/" + filename_list[i]
        print("test：", saved_model)
        model = keras.models.load_model(saved_model)
        test_acc = model.evaluate([x_test_DFT0,x_test_DFT1,x_test_Hilbert], y_testa, verbose=2)
        test_acc_float = ('%.4f' % test_acc[1])
        # 释放模型
        keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        if len(res) <= save_num:
            res.append(test_acc_float)
            os.remove(saved_model)
        elif len(res) > save_num:
            res.sort(key=None, reverse=True)
            nomber_num_acc = res[0]
            if test_acc_float > nomber_num_acc:
                # 获取模型参数，便于建立新模型名
                name_no_hdf5 = saved_model[0:-5]  # 去除.hdf5
                name_no_hdf5 = name_no_hdf5.split("/")
                name_no_hdf5 = name_no_hdf5[1]
                str_list_name = name_no_hdf5.split("_")  # 分隔方式_
                print(str_list_name)
                epoch_param = str_list_name[1]
                val_acc_param = str_list_name[2]
                new_modelname = modell + pad_mode + "_" + str(fram_num) + "_" + str(
                    fram_len) + "_" + epoch_param + "_" + val_acc_param + "_test_" + str(test_acc_float) + ".hdf5"
                print(new_modelname)
                print("save: ", new_modelname)
                savenew_modelname_path = moidel_fin_save_dir + "/" + new_modelname  # 保存路径
                print(saved_model)
                print(savenew_modelname_path)
                shutil.move(saved_model, savenew_modelname_path)  # 移动并改名
            else:
                os.remove(saved_model)
            res.append(test_acc_float)

    newmodel_list = os.listdir(moidel_fin_save_dir)  # 最后测试模型保存的文件列表
    print(newmodel_list)
    if len(newmodel_list) <= save_num:
        lastmodel_list = os.listdir(moidel_fin_save_dir)
    else:
        lastmodel_list = newmodel_list[-save_num:]
    for _ in lastmodel_list:
        newname_no_hdf5 = _[0:-5]  # 去除.hdf5
        newstr_list_name = newname_no_hdf5.split("_")  # 分隔方式_
        print(newstr_list_name[7])
    return lastmodel_list

def softmax_loss(labels, features):
    """
    计算softmax-loss
    :param labels: 等同于y_true，使用了one_hot编码，shape应为(batch_size, NUM_CLASSES)
    :param features: 等同于y_pred，模型的最后一个FC层(不是softmax层)的输出，shape应为(batch_size, NUM_CLASSES)
    :return: 多云分类的softmax-loss损失，shape为(batch_size, )
    """
    return K.categorical_crossentropy(labels, K.softmax(features))

def loss1(labels, features):
    """
    使用这个函数来作为损失函数，计算softmax-loss加上一定比例的center-loss
    :param labels: Tensor，等同于y_true，使用了one_hot编码，shape应为(batch_size, NUM_CLASSES)
    :param features: Tensor， 等同于y_pred, 模型的最后一个fc层(不是softmax层)的输出，shape应为(batch_size, NUM_CLASSES)
    :return: softmax-loss加上一定比例的center-loss
    """
    labels = K.cast(labels, dtype=tf.float32)
    # 计算softmax-loss
    sf_loss = softmax_loss(labels, features)
    # 计算center-loss，因为labels使用了one_hot来编码，所以这里要使用argmax还原到原来的标签
    c_loss = center_loss1(K.argmax(labels, axis=-1), features)
    return sf_loss + LAMBDA * c_loss

def loss2(labels, features):
    """
    使用这个函数来作为损失函数，计算softmax-loss加上一定比例的center-loss
    :param labels: Tensor，等同于y_true，使用了one_hot编码，shape应为(batch_size, NUM_CLASSES)
    :param features: Tensor， 等同于y_pred, 模型的最后一个fc层(不是softmax层)的输出，shape应为(batch_size, NUM_CLASSES)
    :return: softmax-loss加上一定比例的center-loss
    """
    labels = K.cast(labels, dtype=tf.float32)
    # 计算softmax-loss
    sf_loss = softmax_loss(labels, features)
    # 计算center-loss，因为labels使用了one_hot来编码，所以这里要使用argmax还原到原来的标签
    c_loss = center_loss2(K.argmax(labels, axis=-1), features)
    return sf_loss + LAMBDA * c_loss

def loss3(labels, features):
    """
    使用这个函数来作为损失函数，计算softmax-loss加上一定比例的center-loss
    :param labels: Tensor，等同于y_true，使用了one_hot编码，shape应为(batch_size, NUM_CLASSES)
    :param features: Tensor， 等同于y_pred, 模型的最后一个fc层(不是softmax层)的输出，shape应为(batch_size, NUM_CLASSES)
    :return: softmax-loss加上一定比例的center-loss
    """
    labels = K.cast(labels, dtype=tf.float32)
    # 计算softmax-loss
    sf_loss = softmax_loss(labels, features)
    # 计算center-loss，因为labels使用了one_hot来编码，所以这里要使用argmax还原到原来的标签
    c_loss = center_loss3(K.argmax(labels, axis=-1), features)
    return sf_loss + LAMBDA * c_loss

def categorical_accuracy(y_true, y_pred):
    """
    重写categorical_accuracy函数，以适应去掉softmax层的模型
    :param y_true: 等同于labels，
    :param y_pred: 等同于features。
    :return: 准确率
    """
    # 计算y_pred的softmax值
    sm_y_pred = K.softmax(y_pred)
    # 返回准确率
    return K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(sm_y_pred, axis=-1)), K.floatx())

def mul(x):
    return x[0] * x[1]

num_filters =24
My_wd=1e-3
#strides=[1]
learning_rate = 0.0001
#batch_size = 64
#n_epochs = 50
dropout = 0.25
n_classes=2
steps_per_epoch = 50

input_shape=83*25
input_time = keras.layers.Input(shape=[25, 83])
input_space = keras.layers.Input(shape=[25, 83, 1])

def scheduler(epoch):
    # 每隔30个epoch，学习率减小为原来的1/10
    if epoch % 20 == 0 and epoch != 0:
        lr = K.get_value(m.optimizer.lr)
        K.set_value(m.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(m.optimizer.lr)

def train_test_split_and_scaler(data, label, fram_num, fram_len):
    x_train_all, x_test, y_train_all, y_test = train_test_split(data, label, test_size=0.3, random_state=17)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=18)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)
    x_train = x_train.reshape([x_train.shape[0], fram_num, fram_len])
    x_valid = x_valid.reshape([x_valid.shape[0], fram_num, fram_len])
    x_test = x_test.reshape([x_test.shape[0], fram_num, fram_len])
    print(x_train.shape)
    print(x_valid.shape)
    print(x_test.shape)
    return x_train, y_train, x_valid, y_valid, x_test, y_test

def ResBlock(x,filters,kernel_size,dilation_rate):
    h = layers.Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(x)
    s = layers.Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(h)
    r = layers.Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(s)
    if x.shape[-1]==filters:
        shortcut = x
    else:
        shortcut = layers.Conv1D(filters,kernel_size,padding='same')(x)  #shortcut（捷径）
    d = tf.keras.layers.Concatenate(axis=-1)([h, s, r])
    e = layers.Conv1D(filters, 1, padding='same', dilation_rate=dilation_rate)(d)
    o = add([e,shortcut])
    o = Activation('relu')(o)  #激活函数
    return o

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def transformer_decoder(inputs, enc, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs
    if enc is not None:
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(res,enc)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_model(
    fram_num,
    fram_len,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout,
    mlp_dropout,
):
    input1 = keras.layers.Input(shape=[fram_num_DFT, fram_len_DFT],name='input1')
    #DFT0的输入
    x0_inputs = input1
    x0 = keras.layers.Reshape((25, 83, 1))(x0_inputs)

    x0 = keras.layers.ConvLSTM1D(64, 3, strides=3, data_format='channels_last', return_sequences=True,
                            name='convlstm1_1')(x0)
    x0 = keras.layers.BatchNormalization()(x0)
    x0 = keras.layers.ConvLSTM1D(32, 3, strides=2, data_format='channels_last', return_sequences=True,
                            name='convlstm2_1')(x0)
    x0 = keras.layers.BatchNormalization()(x0)
    x0 = keras.layers.Reshape((x0.shape[1], x0.shape[2]*x0.shape[3]))(x0)

    x0 = Bidirectional(LSTM(128), name='layer_lstm1_1')(x0)   #BiLSTM
    x0 = keras.layers.Reshape((x0.shape[1], 1))(x0)

    #无BiLSTM
    # x0 = keras.layers.Reshape((x0.shape[1] * x0.shape[2], ))(x0)
    # x0 = keras.layers.Dense(256)(x0)
    # x0 = keras.layers.Reshape((x0.shape[1], 1))(x0)


    #DFT1的输入
    input2 = keras.layers.Input(shape=[fram_num_DFT, fram_len_DFT], name='input2')
    x2_inputs = input2
    x2 = keras.layers.Reshape((25, 83, 1))(x2_inputs)

    x2 = keras.layers.ConvLSTM1D(64, 3, strides=3, data_format='channels_last', return_sequences=True,
                                name='convlstm1_2')(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.ConvLSTM1D(32, 3, strides=2, data_format='channels_last', return_sequences=True,
                                name='convlstm2_2')(x2)
    x2 = keras.layers.BatchNormalization()(x2)
    x2 = keras.layers.Reshape((x2.shape[1], x2.shape[2] * x2.shape[3]))(x2)
    x2 = Bidirectional(LSTM(128), name='layer_lstm1_2')(x2)
    x2 = keras.layers.Reshape((x2.shape[1], 1))(x2)

    # 无BiLSTM
    # x2 = keras.layers.Reshape((x2.shape[1] * x2.shape[2],))(x2)
    # x2 = keras.layers.Dense(256)(x2)
    # x2 = keras.layers.Reshape((x2.shape[1], 1))(x2)

    # Hilbert的输入
    input3 = keras.layers.Input(shape=[fram_num_Hilbert, fram_len_Hilbert], name='input3')
    x3_inputs = input3
    x3 = keras.layers.Reshape((256, 148, 1))(x3_inputs)

    x3 = keras.layers.ConvLSTM1D(64, 3, strides=3, data_format='channels_last', return_sequences=True,
                                name='convlstm1_3')(x3)
    x3 = keras.layers.BatchNormalization()(x3)
    x3 = keras.layers.ConvLSTM1D(32, 3, strides=2, data_format='channels_last', return_sequences=True,
                                name='convlstm2_3')(x3)
    x3 = keras.layers.BatchNormalization()(x3)
    x3 = keras.layers.Reshape((x3.shape[1], x3.shape[2] * x3.shape[3]))(x3)
    x3 = Bidirectional(LSTM(128), name='layer_lstm1_3')(x3)
    x3 = keras.layers.Reshape((x3.shape[1], 1))(x3)

    # 无BiLSTM
    # # x3 = keras.layers.Reshape((x3.shape[1] * x3.shape[2],))(x3)
    # # x3 = keras.layers.Dense(256)(x3)
    # # x3 = keras.layers.Reshape((x3.shape[1], 1))(x3)


    # 拼接特征融合
    # x = keras.layers.concatenate([x0,x2,x3])

    # 分支注意力特征融合
    hidden17 = keras.layers.concatenate([x0,x2,x3])
    hidden18 = keras.layers.Reshape([768])(hidden17)
    hidden100 = keras.layers.Reshape((2, 384))(hidden18)
    hidden18_1 = keras.layers.Reshape((2, 384,1))(hidden18)
    hidden19 = keras.layers.Conv2D(16, (3, 3), strides=(1, 2), activation='relu', padding='same',data_format='channels_last', name='layer2_con8')(hidden18_1)
    hidden20 = keras.layers.Conv2D(32, (5, 5), strides=(1, 3), activation='relu', padding='same',data_format='channels_last', name='layer2_con9')(hidden19)
    hiddenx = keras.layers.MaxPooling2D(pool_size=(1, 5), strides=(1, 5), padding='valid')(hidden20)
    hidden21 = keras.layers.MaxPooling2D(pool_size=(1, 7), strides=(1, 7), padding='valid')(hiddenx)
    hidden22 = keras.layers.Reshape([64])(hidden21)
    hidden23 = keras.layers.Dense(2, activation='sigmoid')(hidden22)
    hidden24 = keras.layers.Reshape((2, 1))(hidden23)
    x = keras.layers.multiply([hidden24, hidden100])

    # Transformer分类决策
    for _ in range(num_transformer_blocks):
        enc = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        x = transformer_decoder(x, enc, head_size, num_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs=[input1,input2,input3], outputs=outputs)

if __name__ == '__main__':

    #数据准备
    save_e = 5
    data_file = "Feature_data/F01H500next_fram_len_148_256_148.txt"
    # data_file = "Feature_data/F01H2000next_fram_len_148_256_148.txt"
    # data_file = "Feature_data/F01H5168next_fram_len_148_256_148.txt"
    data_time_all = get_data(data_file)
    data_time_lable = data_time_all[:, -1]#标签
    data_time = data_time_all[:, 0:-1]#获得除最后一列标签的数据
    fram_num=25
    fram_len=83
    data_time_lable = to_categorical(data_time_lable, num_classes=2)

    x_train_all, x_test, y_train_all, y_test = train_test_split(data_time, data_time_lable, test_size=0.3,random_state=17)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.2,random_state=18)

    fram_num_DFT=25
    fram_len_DFT=83
    phase_len_DFT = 83 * 25
    fram_num_Hilbert = 256
    fram_len_Hilbert = 148
    phase_len_Hilbert = 148 * 256
    LEN = 2*phase_len_DFT
    train_data_time_DFT0 = x_train[:, 0:phase_len_DFT]
    train_data_time_DFT1 = x_train[:, phase_len_DFT:2*phase_len_DFT]
    train_data_time_Hilbert = x_train[:, 2*phase_len_DFT:2*phase_len_DFT+phase_len_Hilbert]
    valid_data_time_DFT0 = x_valid[:, 0:phase_len_DFT]
    valid_data_time_DFT1 = x_valid[:, phase_len_DFT:2*phase_len_DFT]
    valid_data_time_Hilbert = x_valid[:, 2*phase_len_DFT:2*phase_len_DFT+phase_len_Hilbert]
    test_data_time_DFT0 = x_test[:, 0:phase_len_DFT]
    test_data_time_DFT1 = x_test[:, phase_len_DFT:2*phase_len_DFT]
    test_data_time_Hilbert = x_test[:, 2*phase_len_DFT:2*phase_len_DFT+phase_len_Hilbert]

    scaler = MinMaxScaler()
    train_data_time_DFT0 = scaler.fit_transform(train_data_time_DFT0)
    train_data_time_DFT1 = scaler.fit_transform(train_data_time_DFT1)
    valid_data_time_DFT0 = scaler.transform(valid_data_time_DFT0)
    valid_data_time_DFT1 = scaler.transform(valid_data_time_DFT1)
    test_data_time_DFT0 = scaler.transform(test_data_time_DFT0)
    test_data_time_DFT1 = scaler.transform(test_data_time_DFT1)
    train_data_time_Hilbert = scaler.fit_transform(train_data_time_Hilbert)
    valid_data_time_Hilbert = scaler.transform(valid_data_time_Hilbert)
    test_data_time_Hilbert = scaler.transform(test_data_time_Hilbert)
    print(train_data_time_DFT0.shape)
    print(valid_data_time_DFT0.shape)
    print(test_data_time_DFT0.shape)

    #数据划分
    ###时间
    train_time_DFT0 = train_data_time_DFT0.reshape(train_data_time_DFT0.shape[0], fram_num_DFT, fram_len_DFT)
    test_time_DFT0 = test_data_time_DFT0.reshape(test_data_time_DFT0.shape[0], fram_num_DFT, fram_len_DFT)
    valid_time_DFT0 = valid_data_time_DFT0.reshape(valid_data_time_DFT0.shape[0], fram_num_DFT, fram_len_DFT)
    train_time_DFT1 = train_data_time_DFT1.reshape(train_data_time_DFT1.shape[0], fram_num_DFT, fram_len_DFT)
    test_time_DFT1 = test_data_time_DFT1.reshape(test_data_time_DFT1.shape[0], fram_num_DFT, fram_len_DFT)
    valid_time_DFT1 = valid_data_time_DFT1.reshape(valid_data_time_DFT1.shape[0], fram_num_DFT, fram_len_DFT)
    train_time_Hilbert = train_data_time_Hilbert.reshape(train_data_time_Hilbert.shape[0], fram_num_Hilbert, fram_len_Hilbert)
    test_time_Hilbert = test_data_time_Hilbert.reshape(test_data_time_Hilbert.shape[0], fram_num_Hilbert, fram_len_Hilbert)
    valid_time_Hilbert = valid_data_time_Hilbert.reshape(valid_data_time_Hilbert.shape[0], fram_num_Hilbert, fram_len_Hilbert)


###  模型训练
    m = build_model(
        fram_num,
        fram_len,
        head_size=64,
        num_heads=8,
        ff_dim=8,
        num_transformer_blocks=2,
        mlp_units=[128],
        mlp_dropout=0.25,
        dropout=0.25,
    )
    m.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"],
    )
    m.summary()

    #训练方式
    logdir = "logdir0" + "_" + str(fram_len) + "_" + str(fram_num)
    modella = "model0"
    if not os.path.exists(logdir):  # 若该文件夹不存在 则创建一个
        os.mkdir(logdir)
    checkpointer = keras.callbacks.ModelCheckpoint(
        os.path.join(logdir, 'model_epoch{epoch:02d}_valacc{val_accuracy:.2f}.hdf5'),
        verbose=0, save_weights_only=False)
    m.fit([train_time_DFT0,train_time_DFT1,train_time_Hilbert], y_train, epochs=200, batch_size=8,
          validation_data=([valid_time_DFT0,valid_time_DFT1,valid_time_Hilbert], y_valid),  verbose=2,callbacks=[checkpointer])

#模型保存
    logdir = "logdir0" + "_" + str(fram_len) + "_" + str(fram_num)
    modella = "model0"

    lastmodel_list = test_model_and_find_top_acc(test_time_DFT0,test_time_DFT1,test_time_Hilbert, y_test, logdir, save_e, "pad_mode", fram_num, fram_len,
                                                 modella)
    print(lastmodel_list)
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    # model_fin_save_dir = "model0" + "pad_mode" + "_" + str(fram_num) + "_" + str(fram_len)  # 保存测试后模型的文件夹名
    # lastmodel_list = os.listdir(model_fin_save_dir)  # 最后测试模型保存的文件列表
    # top_model_path = model_fin_save_dir + "/" + lastmodel_list[-1]

    model_fin_save_dir = modella + "pad_mode" + "_" + str(fram_num) + "_" + str(fram_len)  # 保存测试后模型的文件夹名
    top_model_path = model_fin_save_dir + "/" + lastmodel_list[-1]
    top_model = keras.models.load_model(top_model_path)
    y_pre = top_model.predict([test_time_DFT0,test_time_DFT1,test_time_Hilbert])
    true = np.argmax(y_test, 1)
    pre = np.argmax(y_pre, 1)
    ##绘制混淆矩阵
    matrix = confusion_matrix(true, pre)
    plt.imshow(matrix, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len([0, 1]))
    plt.xticks(tick_marks, [0, 1])
    plt.yticks(tick_marks, [0, 1])
    mid = matrix.max() / 2
    for i in np.arange(2):
        for j in np.arange(2):
            plt.text(j, i, matrix[i, j], horizontalalignment='center', color='white' if matrix[i, j] > mid else 'black')
    plt.xlabel('True Lable')
    plt.ylabel('Predict Result')
    plt.show()

    # 用于打印分类报告（classification report），其中包含了模型的准确率、召回率、F1值等指标
    print('classification report', classification_report(true, pre, digits=5))
    # 评估模型性能得到损失值和准确率
    loss, categorical_accuracy = top_model.evaluate([test_time_DFT0,test_time_DFT1,test_time_Hilbert],
                                                    y_test)  # 该函数返回值是一个包含两个元素的列表，第一个元素是测试集上的损失值（test loss），第二个元素是测试集上的准确率（test accuracy）
    print('loss:', loss)
    print('accuracy:', categorical_accuracy)
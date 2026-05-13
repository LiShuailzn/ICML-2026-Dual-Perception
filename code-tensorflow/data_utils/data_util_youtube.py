
#encoding=utf-8
#author: liang xinyan
#email: liangxinyan48@163.com
import tensorflow as tf
import numpy as np
import os
import config
opt = os.path
paras = config.get_configs()
nb_view = paras['nb_view']
image_size = paras['image_size']
w, h, c = image_size['w'], image_size['h'], image_size['c']

def get_data(data_base_dir='..'):
    print('Data loading ......')
    train_x = np.load(os.path.join(data_base_dir, 'train_X.npy'))
    test_x = np.load(os.path.join(data_base_dir, 'test_X.npy'))
    if c == 1:
        train_x = np.expand_dims(train_x, axis=-1)
        test_x = np.expand_dims(test_x, axis=-1)
    train_x = (train_x / 127.5) - 1.
    test_x = (test_x / 127.5) - 1.
    train_y = np.load(os.path.join(data_base_dir, 'train_Y.npy'))
    test_y = np.load(os.path.join(data_base_dir, 'test_Y.npy'))
    train_y = tf.keras.utils.to_categorical(train_y)
    test_y = tf.keras.utils.to_categorical(test_y)
    print('Data loading finished！！！')
    return train_x, train_y, test_x, test_y



# ③ youtube
def get_views(view_data_dir='/home/lishuai_lxy/fph/YoutubeFace'):
    num_views = 5
    view_train_x = []
    view_test_x = []
    for i in range(1, num_views + 1):
        train_file = os.path.join(view_data_dir, f'train_{i}.npy')
        test_file = os.path.join(view_data_dir, f'test_{i}.npy')
        view_train_x.append(np.load(train_file))
        view_test_x.append(np.load(test_file))
        print(f"Loading view_train and view_test for view {i}")
    train_y_file = os.path.join(view_data_dir, 'train_y.npy')
    test_y_file = os.path.join(view_data_dir, 'test_y.npy')
    train_y = np.load(train_y_file)
    test_y = np.load(test_y_file)
    train_y = tf.keras.utils.to_categorical(train_y)
    test_y = tf.keras.utils.to_categorical(test_y)
    # 打印 view_train_x 中每个元素的维度
    print(f"Dimensions of view_train_x: {[arr.shape for arr in view_train_x]}")
    # 打印标签的维度
    print(f"Dimensions of train_y: {train_y.shape}")
    print(f"Dimensions of test_y: {test_y.shape}")

    return view_train_x, train_y, view_test_x, test_y

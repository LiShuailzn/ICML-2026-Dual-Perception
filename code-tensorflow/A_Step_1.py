import os
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
import random
from sklearn import metrics
from code2net_tree import code2net_tree
import config

SEED = 42
def set_random_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

set_random_seeds(SEED)

paras = config.get_configs()
fusion_ways = paras['fusion_ways']
fused_nb_feats = paras['fused_nb_feats']
classes = paras['classes']
batch_size = paras['batch_size']
epochs = paras['epochs']
pop_size = paras['pop_size']
nb_iters = paras['nb_iters']
data_name = paras['data_name']

data_base_dir = os.path.join('..', data_name)
view_data_dir = os.path.join(data_base_dir, 'view')

from data_utils.data_util_r2 import get_views
view_train_x, train_y, view_test_x, test_y = get_views(
    view_data_dir='/mnt/disk1/lishuai/code-tensorflow/R5/test_1/noisy dataset')

def train_individual(individual_code, result_save_dir='.', gpu='0', seed=SEED):
    set_random_seeds(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    print(individual_code)
    print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Using random seed: {seed}")
    view_train_xx, view_test_xx = [], []
    for code in individual_code:
        if code[0] != '-':
            index = int(code[0])
            view_train_xx.append(view_train_x[index])
            view_test_xx.append(view_test_x[index])
    individual_code_str = '+'.join([str(ind) for ind in individual_code])
    nb_feats = [i.shape[1] for i in view_train_x]
    tf.keras.utils.set_random_seed(seed)
    model = code2net_tree(individual_code=[], nb_feats=nb_feats, listtree=individual_code)
    num_views = len(view_train_xx)
    adam = tf.keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    checkpoint_filepath = os.path.join(result_save_dir, individual_code_str + f'_seed{seed}' + '.h5')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=paras['patience'])
    csv_filepath = os.path.join(result_save_dir, individual_code_str + f'_seed{seed}' + '.csv')
    csv_logger = tf.keras.callbacks.CSVLogger(csv_filepath)
    model.fit(view_train_xx, train_y, batch_size=batch_size, epochs=epochs,
              verbose=0, validation_data=(view_test_xx, test_y),
              callbacks=[csv_logger, early_stop, checkpoint])
    model_best = tf.keras.models.load_model(checkpoint_filepath)
    pre_y = model_best.predict(view_test_xx)
    pre_y = np.argmax(pre_y, axis=1)
    true_y = np.argmax(test_y, axis=1)
    acc = metrics.accuracy_score(true_y, pre_y)
    print(f"Model accuracy: {acc:.4f}")
    return individual_code_str + ',' + str(acc)

individual_code = ['0a', '1a', '2a', '3a', '4a',
                   '-0', '-0', '-0', '-0',]
result_save_dir = '/mnt/disk1/lishuai/code-tensorflow/step-1'

gpu = '0'  # 指定 GPU
train_individual(individual_code, result_save_dir, gpu, seed=SEED)

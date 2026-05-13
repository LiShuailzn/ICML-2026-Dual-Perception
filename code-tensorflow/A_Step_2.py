import os
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
import random
from sklearn.ensemble import IsolationForest
from sklearn.random_projection import SparseRandomProjection
from code2net_tree import code2net_tree
import config
import time

NOISE_RATE = 0.2
Dataname = 'R5'


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
step_1_MODEL_DIR = '/mnt/disk1/lishuai/code-tensorflow/step-1/0a+1a+2a+3a+4a+-0+-0+-0+-0_seed42.h5'


def train_individual_with_transfer(individual_code, result_save_dir='.', gpu='0', seed=SEED):
    set_random_seeds(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
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
    phase2_checkpoint = step_1_MODEL_DIR
    if os.path.exists(phase2_checkpoint):
        model.load_weights(phase2_checkpoint)
        print(f"[INFO] phase 2 best model: {phase2_checkpoint}")
    else:
        print("[WARN] Not found phase 2 best model")
    os.makedirs(result_save_dir, exist_ok=True)
    num_views = len(view_train_xx)
    view_train_tensors = [
        tf.convert_to_tensor(x, dtype=tf.float32) for x in view_train_xx
    ]
    train_y_tensor = tf.convert_to_tensor(train_y, dtype=tf.float32)
    num_samples = train_y_tensor.shape[0]
    num_classes = train_y_tensor.shape[1]
    print(f">>> Train samples: {num_samples}, num_classes: {num_classes}")
    print(f">>> Using {num_views} views for this individual: {individual_code_str}")
    last_fc = model.get_layer('fusion_fc_2')
    kernel = last_fc.kernel  # [feat_dim, num_classes]
    grad_dim = int(np.prod(kernel.shape))
    print(f">>> Last linear layer: fusion_fc_2, kernel shape = {kernel.shape}")
    print(f">>> Each gradient vector dim = {grad_dim}")
    grads_all = np.zeros((num_samples, grad_dim), dtype=np.float32)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    gradient_start_time = time.perf_counter()
    for i in range(num_samples):
        inputs = [
            tf.expand_dims(view_train_tensors[v][i], axis=0)
            for v in range(num_views)
        ]
        y_true = tf.expand_dims(train_y_tensor[i], axis=0)
        with tf.GradientTape() as tape:
            logits = model(inputs, training=False)
            loss = loss_fn(y_true, logits)
        grad_kernel = tape.gradient(loss, kernel)  # [feat_dim, num_classes]
        if grad_kernel is None:
            raise RuntimeError("gradient of fusion_fc_2.kernel is None, "
                               "please check model/trainable_variables.")
        grads_all[i] = tf.reshape(grad_kernel, [-1]).numpy()
        if (i + 1) % 1000 == 0 or i == num_samples - 1:
            print(f"    computed gradients for {i + 1}/{num_samples} samples")
    gradient_end_time = time.perf_counter()
    gradient_extraction_time = gradient_end_time - gradient_start_time
    print(">>> Finished computing per-sample gradients.")
    print(f">>> 梯度提取时间消耗: {gradient_extraction_time:.4f} 秒")
    print(">>> Running IsolationForest on gradient vectors ...")
    X_for_iforest = grads_all
    if Dataname in ['VOX', 'CB']:
        print(f">>> Detected Dataname = {Dataname}, applying SparseRandomProjection before IsolationForest ...")
        n_samples, n_features = grads_all.shape
        print(f"    Original gradient dimension: {grads_all.shape}")
        projector = SparseRandomProjection(
            n_components='auto',
            eps=0.1,
            dense_output=True,
            random_state=seed,
        )
        X_for_iforest = projector.fit_transform(grads_all).astype(np.float32)
        print(f"    Projected gradient dimension: {grads_all.shape} -> {X_for_iforest.shape}")
    else:
        print(f">>> Dataname = {Dataname}, skip dimensionality reduction and use raw gradients.")
    outlier_start_time = time.perf_counter()
    clf = IsolationForest(
        contamination=NOISE_RATE,
        random_state=seed,
        n_jobs=-1
    )
    clf.fit(X_for_iforest)
    scores = clf.predict(X_for_iforest)  # 1 = inlier, -1 = outlier

    outlier_end_time = time.perf_counter()
    outlier_detection_time = outlier_end_time - outlier_start_time
    outlier_indices = np.where(scores == -1)[0]
    inlier_indices = np.where(scores == 1)[0]
    print(f">>> IForest finished: total {len(grads_all)}, "
          f"outliers {len(outlier_indices)}, inliers {len(inlier_indices)}")
    print(f">>> 离群检测时间消耗: {outlier_detection_time:.4f} 秒")
    harmful_indices = outlier_indices
    print(">>> Sample pruning summary:")
    print(f"    Original number of training samples: {num_samples}")
    print(f"    Pruning ratio (contamination): {NOISE_RATE:.2%}")
    print(f"    Number of pruned samples: {len(harmful_indices)}")

    save_path = os.path.join(result_save_dir, 'harmful_outlier_indices.npy')
    np.save(save_path, harmful_indices)
    print(">>> Harmful (outlier) sample indices saved to:")
    print(f"    {save_path}")

    print("\n================ 时间开销统计 ================")
    print(f"梯度提取时间消耗   = {gradient_extraction_time:.4f} 秒")
    print(f"离群检测时间消耗   = {outlier_detection_time:.4f} 秒")
    print(f"两阶段总时间消耗   = {gradient_extraction_time + outlier_detection_time:.4f} 秒")
    print("============================================")

    return individual_code_str

individual_code = ['0a', '1a', '2a', '3a', '4a',
                   '-0', '-0', '-0', '-0',]
result_save_dir = '/mnt/disk1/lishuai/code-tensorflow/step-2'

gpu = '0'
train_individual_with_transfer(individual_code, result_save_dir, gpu, seed=SEED)

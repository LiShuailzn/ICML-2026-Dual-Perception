import os
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
import random
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
CLEAN_MODEL_PATH = '/mnt/disk1/lishuai/code-tensorflow/step-3/0a+1a+2a+3a+4a+-0+-0+-0+-0_seed42.h5'
HARMFUL_INDICES_PATH = '/mnt/disk1/lishuai/code-tensorflow/step-2/harmful_outlier_indices.npy'
LABEL_SAVE_DIR = '/mnt/disk1/lishuai/code-tensorflow/R5/test_1/purified dataset'



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
    if os.path.exists(CLEAN_MODEL_PATH):
        model.load_weights(CLEAN_MODEL_PATH)
        print(f"[INFO] Loaded clean best model weights from: {CLEAN_MODEL_PATH}")
    else:
        raise FileNotFoundError(f"[ERROR] Clean best model not found: {CLEAN_MODEL_PATH}")
    model.trainable = False
    try:
        feature_layer = model.get_layer('fusion_features').output
    except ValueError as e:
        raise ValueError("找不到名为 'fusion_features' 的层，请检查 code2net_tree 中的层命名。") from e
    feature_model = tf.keras.Model(inputs=model.inputs, outputs=feature_layer)
    if not os.path.exists(HARMFUL_INDICES_PATH):
        raise FileNotFoundError(f"Harmful indices file not found: {HARMFUL_INDICES_PATH}")
    harmful_indices = np.load(HARMFUL_INDICES_PATH).astype(int)
    harmful_indices = np.unique(harmful_indices)
    print(f">>> Loaded {len(harmful_indices)} harmful samples from {HARMFUL_INDICES_PATH}")
    num_total = train_y.shape[0]
    all_indices = np.arange(num_total)
    mask = np.ones(num_total, dtype=bool)
    mask[harmful_indices] = False
    clean_indices = all_indices[mask]
    print(f">>> Total train samples: {num_total}")
    print(f">>> Clean (purified) train samples: {len(clean_indices)}")
    print(f">>> Harmful samples: {len(harmful_indices)}")
    clean_view_train_xx = [x[clean_indices] for x in view_train_xx]
    clean_train_y = train_y[clean_indices]          # one-hot
    clean_labels_int = np.argmax(clean_train_y, axis=1)
    num_classes = clean_train_y.shape[1]
    print(">>> Extracting fusion_features for purified (clean) dataset ...")
    clean_feats = feature_model.predict(
        clean_view_train_xx, batch_size=256, verbose=1
    )  # [N_clean, feat_dim]
    feat_dim = clean_feats.shape[1]
    print(f">>> clean_feats shape: {clean_feats.shape}")
    prototypes = np.zeros((num_classes, feat_dim), dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.int32)
    for c in range(num_classes):
        idx_c = np.where(clean_labels_int == c)[0]
        if len(idx_c) == 0:
            print(f"[WARN] class {c} has 0 samples in purified dataset, prototype stays zero.")
            continue
        prototypes[c] = clean_feats[idx_c].mean(axis=0)
        counts[c] = len(idx_c)
    print(">>> Prototype counts per class:", counts.tolist())
    proto_norms = np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-12
    prototypes = prototypes / proto_norms
    print(">>> Extracting fusion_features for harmful samples ...")
    harmful_view_train_xx = [x[harmful_indices] for x in view_train_xx]
    harmful_feats = feature_model.predict(
        harmful_view_train_xx, batch_size=256, verbose=1
    )  # [N_harm, feat_dim]
    print(f">>> harmful_feats shape: {harmful_feats.shape}")
    harmful_norms = np.linalg.norm(harmful_feats, axis=1, keepdims=True) + 1e-12
    harmful_feats_norm = harmful_feats / harmful_norms
    cos_sims = np.matmul(harmful_feats_norm, prototypes.T)
    new_labels_int = np.argmax(cos_sims, axis=1)
    original_labels_int = np.argmax(train_y, axis=1)
    corrected_labels_int = original_labels_int.copy()
    for idx, new_y in zip(harmful_indices, new_labels_int):
        corrected_labels_int[idx] = int(new_y)
    os.makedirs(LABEL_SAVE_DIR, exist_ok=True)
    corrected_labels_path = os.path.join(LABEL_SAVE_DIR, 'corrected_labels.npy')
    np.save(corrected_labels_path, corrected_labels_int)
    print(">>> Saved corrected integer labels to:", corrected_labels_path)
    os.makedirs(result_save_dir, exist_ok=True)
    return individual_code_str

if __name__ == "__main__":
    individual_code = ['0a', '1a', '2a', '3a', '4a',
                       '-0', '-0', '-0', '-0', ]
    result_save_dir = '/mnt/disk1/lishuai/code-tensorflow/R5/test_1/purified dataset'
    gpu = '0'
    _ = train_individual_with_transfer(individual_code, result_save_dir, gpu, seed=SEED)

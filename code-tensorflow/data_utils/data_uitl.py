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

# # ① CB
# def get_views(view_data_dir='views'):
#     models_ls = ['resnet50', 'desnet121', 'MobileNetV2', 'Xception', 'InceptionV3','resnet18', 'resnet34', 'desnet169', 'desnet201', 'NASNetMobile']
#     view_train_x = []
#     view_test_x = []
#     for model in models_ls:
#         view_train_x.append(np.load(os.path.join(view_data_dir, model+'train_X.npy')))
#         view_test_x.append(np.load(os.path.join(view_data_dir, model+'test_X.npy')))
#         print(f"Loading view_train_x and view_test_x for model {model}")
#     train_y = np.load(os.path.join(view_data_dir, 'train_Y.npy'))
#     test_y = np.load(os.path.join(view_data_dir, 'test_Y.npy'))
#     train_y = tf.keras.utils.to_categorical(train_y)
#     test_y = tf.keras.utils.to_categorical(test_y)
#     # 打印 view_train_x 中每个元素的维度
#     print(f"Dimensions of view_train_x: {[arr.shape for arr in view_train_x]}")
#
#     return view_train_x, train_y, view_test_x, test_y
#
# def load_teacher_logits(view_data_dir='views', models_ls=None):
#     models_ls = ['resnet50', 'desnet121', 'MobileNetV2', 'Xception', 'InceptionV3','resnet18', 'resnet34', 'desnet169', 'desnet201', 'NASNetMobile']
#
#     teacher_labels = []  # 用来存储软标签
#
#     for i, model in enumerate(models_ls):
#         # 软标签文件路径
#         soft_label_path = os.path.join(view_data_dir, f'view_{i}_train_Y.npy')
#
#         # 检查软标签文件是否存在
#         if os.path.exists(soft_label_path):
#             print(f"Loading soft label for model {model} from {soft_label_path}")
#             # 加载软标签
#             soft_label = np.load(soft_label_path)
#             teacher_labels.append(soft_label)
#         else:
#             print(f"Warning: Soft label file for model {model} (view {i}) not found at {soft_label_path}.")
#             teacher_labels.append(None)  # 如果找不到软标签文件，设置为 None
#
#     return teacher_labels
#
#
# def load_cost_matrices(view_data_dir='views', models_ls=None):
#     models_ls = ['resnet50', 'desnet121', 'MobileNetV2', 'Xception', 'InceptionV3','resnet18', 'resnet34', 'desnet169', 'desnet201', 'NASNetMobile']
#
#     cost_matrices = []  # 用来存储成本矩阵
#
#     for i, model in enumerate(models_ls):
#         # 成本矩阵文件路径
#         cost_matrix_path = os.path.join(view_data_dir, f'view_{i}_kernel', 'cost_matrix.npy')
#
#         # 检查成本矩阵文件是否存在
#         if os.path.exists(cost_matrix_path):
#             print(f"Loading cost matrix for model {model} from {cost_matrix_path}")
#             # 加载成本矩阵
#             cost_matrix = np.load(cost_matrix_path)
#             cost_matrices.append(cost_matrix)
#         else:
#             print(f"Warning: Cost matrix file for model {model} (view {i}) not found at {cost_matrix_path}.")
#             cost_matrices.append(None)  # 如果找不到成本矩阵文件，设置为 None
#
#     return cost_matrices



def get_views(view_data_dir='views'):
    models_ls = ['resnet50', 'desnet121', 'MobileNetV2', 'Xception', 'InceptionV3','resnet18', 'resnet34', 'desnet169', 'desnet201', 'NASNetMobile']
    view_train_x = []
    view_test_x = []
    for model in models_ls:
        view_train_x.append(np.load(os.path.join(view_data_dir, model+'train_X.npy')))
        view_test_x.append(np.load(os.path.join(view_data_dir, model+'test_X.npy')))
        print(f"Loading view_train_x and view_test_x for model {model}")
    train_y = np.load(os.path.join(view_data_dir, 'train_Y.npy'))
    test_y = np.load(os.path.join(view_data_dir, 'test_Y.npy'))
    train_y = tf.keras.utils.to_categorical(train_y)
    test_y = tf.keras.utils.to_categorical(test_y)
    # 打印 view_train_x 中每个元素的维度
    print(f"Dimensions of view_train_x: {[arr.shape for arr in view_train_x]}")

    return view_train_x, train_y, view_test_x, test_y
def load_teacher_labels(view_data_dir='views', models_ls=None):
    models_ls = ['resnet50', 'desnet121', 'MobileNetV2', 'Xception', 'InceptionV3','resnet18', 'resnet34', 'desnet169', 'desnet201', 'NASNetMobile']

    teacher_labels = []  # 用来存储软标签

    for i, model in enumerate(models_ls):
        # 软标签文件路径
        soft_label_path = os.path.join(view_data_dir, f'view_{i}_train_Y.npy')

        # 检查软标签文件是否存在
        if os.path.exists(soft_label_path):
            print(f"Loading soft label for model {model} from {soft_label_path}")
            # 加载软标签
            soft_label = np.load(soft_label_path)
            teacher_labels.append(soft_label)
        else:
            print(f"Warning: Soft label file for model {model} (view {i}) not found at {soft_label_path}.")
            teacher_labels.append(None)  # 如果找不到软标签文件，设置为 None

    return teacher_labels







#
# # # # ② NTU
# def get_views(view_data_dir='views'):
#     models_ls = ['Skeleton1', 'Skeleton2', 'Skeleton3', 'Skeleton4', 'Video1', 'Video2', 'Video3', 'Video4']
#     view_train_x = []
#     view_test_x = []
#     for model in models_ls:
#         view_train_x.append(np.load(os.path.join(view_data_dir, model+'train_X.npy')))
#         view_test_x.append(np.load(os.path.join(view_data_dir, model+'test_X.npy')))
#         print(f"Loading view_train_x and view_test_x for model {model}")
#     train_y = np.load(os.path.join(view_data_dir, 'train_Y.npy'))
#     test_y = np.load(os.path.join(view_data_dir, 'test_Y.npy'))
#     train_y = tf.keras.utils.to_categorical(train_y)
#     test_y = tf.keras.utils.to_categorical(test_y)
#     # 打印 view_train_x 中每个元素的维度
#     print(f"Dimensions of view_train_x: {[arr.shape for arr in view_train_x]}")
#     # 打印标签的维度
#     print(f"Dimensions of train_y: {train_y.shape}")
#     print(f"Dimensions of test_y: {test_y.shape}")
#
#
#     return view_train_x, train_y, view_test_x, test_y
#
# def load_teacher_logits(view_data_dir='views', models_ls=None):
#     models_ls = ['Skeleton1', 'Skeleton2', 'Skeleton3', 'Skeleton4', 'Video1', 'Video2', 'Video3', 'Video4']
#
#     teacher_labels = []  # 用来存储软标签
#
#     for i, model in enumerate(models_ls):
#         # 软标签文件路径
#         soft_label_path = os.path.join(view_data_dir, f'view_{i}_logits.npy')
#
#         # 检查软标签文件是否存在
#         if os.path.exists(soft_label_path):
#             print(f"Loading soft label for model {model} from {soft_label_path}")
#             # 加载软标签
#             soft_label = np.load(soft_label_path)
#             teacher_labels.append(soft_label)
#         else:
#             print(f"Warning: Soft label file for model {model} (view {i}) not found at {soft_label_path}.")
#             teacher_labels.append(None)  # 如果找不到软标签文件，设置为 None
#
#     return teacher_labels
#
#
# def load_cost_matrices(view_data_dir='views', models_ls=None):
#     models_ls = ['Skeleton1', 'Skeleton2', 'Skeleton3', 'Skeleton4', 'Video1', 'Video2', 'Video3', 'Video4']
#
#     cost_matrices = []  # 用来存储成本矩阵
#
#     for i, model in enumerate(models_ls):
#         # 成本矩阵文件路径
#         cost_matrix_path = os.path.join(view_data_dir, f'view_{i}_kernel', 'cost_matrix.npy')
#
#         # 检查成本矩阵文件是否存在
#         if os.path.exists(cost_matrix_path):
#             print(f"Loading cost matrix for model {model} from {cost_matrix_path}")
#             # 加载成本矩阵
#             cost_matrix = np.load(cost_matrix_path)
#             cost_matrices.append(cost_matrix)
#         else:
#             print(f"Warning: Cost matrix file for model {model} (view {i}) not found at {cost_matrix_path}.")
#             cost_matrices.append(None)  # 如果找不到成本矩阵文件，设置为 None
#
#     return cost_matrices
#
# # ③ EGO
# def get_views(view_data_dir='views'):
#     models_ls = ['Depth1', 'Depth2', 'Depth3', 'Depth4', 'RGB1', 'RGB2', 'RGB3', 'RGB4']
#     view_train_x = []
#     view_test_x = []
#     for model in models_ls:
#         view_train_x.append(np.load(os.path.join(view_data_dir, model+'train.npy')))
#         view_test_x.append(np.load(os.path.join(view_data_dir, model+'test.npy')))
#         print(f"Loading view_train and view_test for model {model}")
#     train_y = np.load(os.path.join(view_data_dir, 'train_YY.npy'))
#     test_y = np.load(os.path.join(view_data_dir, 'test_YY.npy'))
#     train_y = tf.keras.utils.to_categorical(train_y)
#     test_y = tf.keras.utils.to_categorical(test_y)
#     # 打印 view_train_x 中每个元素的维度
#     print(f"Dimensions of view_train_x: {[arr.shape for arr in view_train_x]}")
#     # 打印标签的维度
#     print(f"Dimensions of train_y: {train_y.shape}")
#     print(f"Dimensions of test_y: {test_y.shape}")
#
#
#     return view_train_x, train_y, view_test_x, test_y
#
# def load_teacher_logits(view_data_dir='views', models_ls=None):
#     models_ls = ['Depth1', 'Depth2', 'Depth3', 'Depth4', 'RGB1', 'RGB2', 'RGB3', 'RGB4']
#
#     teacher_labels = []  # 用来存储软标签
#
#     for i, model in enumerate(models_ls):
#         # 软标签文件路径
#         soft_label_path = os.path.join(view_data_dir, f'view_{i}_logits.npy')
#
#         # 检查软标签文件是否存在
#         if os.path.exists(soft_label_path):
#             print(f"Loading soft label for model {model} from {soft_label_path}")
#             # 加载软标签
#             soft_label = np.load(soft_label_path)
#             teacher_labels.append(soft_label)
#         else:
#             print(f"Warning: Soft label file for model {model} (view {i}) not found at {soft_label_path}.")
#             teacher_labels.append(None)  # 如果找不到软标签文件，设置为 None
#
#     return teacher_labels
#
#
# def load_cost_matrices(view_data_dir='views', models_ls=None):
#     models_ls = ['Depth1', 'Depth2', 'Depth3', 'Depth4', 'RGB1', 'RGB2', 'RGB3', 'RGB4']
#
#     cost_matrices = []  # 用来存储成本矩阵
#
#     for i, model in enumerate(models_ls):
#         # 成本矩阵文件路径
#         cost_matrix_path = os.path.join(view_data_dir, f'view_{i}_kernel', 'cost_matrix.npy')
#
#         # 检查成本矩阵文件是否存在
#         if os.path.exists(cost_matrix_path):
#             print(f"Loading cost matrix for model {model} from {cost_matrix_path}")
#             # 加载成本矩阵
#             cost_matrix = np.load(cost_matrix_path)
#             cost_matrices.append(cost_matrix)
#         else:
#             print(f"Warning: Cost matrix file for model {model} (view {i}) not found at {cost_matrix_path}.")
#             cost_matrices.append(None)  # 如果找不到成本矩阵文件，设置为 None
#
#     return cost_matrices

#



# # # ④ MMIM
# def get_views(view_data_dir='views'):
#     models_ls = ['Image1', 'Image2', 'Image3', 'Image4', 'Text1', 'Text2']
#     view_train_x = []
#     view_test_x = []
#     for model in models_ls:
#         view_train_x.append(np.load(os.path.join(view_data_dir, model+'train_X.npy')))
#         view_test_x.append(np.load(os.path.join(view_data_dir, model+'test_X.npy')))
#         print(f"Loading view_train_x and view_test_x for model {model}")
#     train_y = np.load(os.path.join(view_data_dir, 'train_Y.npy'))
#     test_y = np.load(os.path.join(view_data_dir, 'test_Y.npy'))
#     # 打印 view_train_x 中每个元素的维度
#     print(f"Dimensions of view_train_x: {[arr.shape for arr in view_train_x]}")
#     # 打印标签的维度
#     print(f"Dimensions of train_y: {train_y.shape}")
#     print(f"Dimensions of test_y: {test_y.shape}")
#
#
#     return view_train_x, train_y, view_test_x, test_y
#
# def load_teacher_labels(view_data_dir='views', models_ls=None):
#
#     teacher_labels = []  # 用来存储软标签
#     models_ls = ['Image1', 'Image2', 'Image3', 'Image4', 'Text1', 'Text2']
#
#     for i, model in enumerate(models_ls):
#         # 软标签文件路径
#         soft_label_path = os.path.join(view_data_dir, f'view_{i}_train_Y.npy')
#
#         # 检查软标签文件是否存在
#         if os.path.exists(soft_label_path):
#             print(f"Loading soft label for model {model} from {soft_label_path}")
#             # 加载软标签
#             soft_label = np.load(soft_label_path)
#             teacher_labels.append(soft_label)
#         else:
#             print(f"Warning: Soft label file for model {model} (view {i}) not found at {soft_label_path}.")
#             teacher_labels.append(None)  # 如果找不到软标签文件，设置为 None
#
#     return teacher_labels








if __name__ == '__main__':
    base_dir = opt.join('fn')
    train_fns, train_y, test_fns, test_y = get_image_paths(base_dir=base_dir)
    # train_fns = [opt.join('data', v) for v in train_fns]
    # train_fns = [v.split('_')[0] for v in train_fns]
    # print(len(set(train_fns)))
    print(train_fns)

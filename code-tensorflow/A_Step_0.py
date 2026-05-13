import os
import numpy as np
from data_utils.data_util_r2 import get_views

def add_label_noise_youtube(data_dir, noise_rate, seed, save_suffix):
    rng = np.random.RandomState(seed)
    view_train_x, train_y_onehot, view_test_x, test_y_onehot = get_views(data_dir)
    y = np.argmax(train_y_onehot, axis=1)
    num_samples = y.shape[0]
    num_classes = train_y_onehot.shape[1]
    print(f"总样本数 N = {num_samples}, 类别数 C = {num_classes}")
    target_noisy = int(round(noise_rate * num_samples))
    print(f"目标噪声样本数 = {target_noisy} ({noise_rate * 100:.1f}% 噪声)")
    base_per_class = target_noisy // num_classes
    extra = target_noisy % num_classes
    print(f"每类基础噪声数 = {base_per_class}, 其中前 {extra} 个类别多 1 个")
    noisy_indices = []
    for c in range(num_classes):
        class_indices = np.where(y == c)[0]
        if len(class_indices) == 0:
            print(f"警告：类别 {c} 没有样本，跳过。")
            continue
        k = base_per_class + (1 if c < extra else 0)
        k = min(k, len(class_indices))
        if k == 0:
            continue
        chosen = rng.choice(class_indices, size=k, replace=False)
        noisy_indices.extend(chosen)
    noisy_indices = np.array(sorted(noisy_indices))
    print(f"实际加噪样本数 = {len(noisy_indices)}")
    y_noisy = y.copy()
    for idx in noisy_indices:
        old_label = y_noisy[idx]
        r = rng.randint(0, num_classes - 1)
        new_label = r if r < old_label else r + 1
        y_noisy[idx] = new_label
    noisy_label_path = os.path.join(data_dir, f"train_y_{save_suffix}.npy")
    np.save(noisy_label_path, y_noisy)
    print(f"已保存含噪声整型标签到: {noisy_label_path}")
    return noisy_indices

# -----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    add_label_noise_youtube(
        data_dir="/mnt/disk1/lishuai/code-tensorflow/R5/test_1/noisy dataset",
        noise_rate=0.4,
        seed=42,
        save_suffix="noisy_40")
# -----------------------------------------------------------------------------------------------------
import os
import shutil
import random
from collections import defaultdict

# ====================== 只需要修改这里的路径 ======================
# 原始数据集根目录（里面必须包含images和labels两个文件夹）
DATASET_ROOT = "D:/ruod_1"
# 输出划分后的数据集目录（自动创建）
OUTPUT_ROOT = os.path.join(os.path.dirname(DATASET_ROOT), "ruod_split_811")
# 随机种子（保证每次划分结果完全一致）
RANDOM_SEED = 42
# 划分比例 train:val:test = 8:1:1
SPLIT_RATIO = [0.8, 0.1, 0.1]
# 支持的图片格式
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
# =================================================================

def main():
    random.seed(RANDOM_SEED)
    
    # 1. 创建输出目录结构
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_ROOT, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, 'labels', split), exist_ok=True)
    
    # 2. 收集所有标签文件并按类别分组
    label_dir = os.path.join(DATASET_ROOT, 'labels')
    image_dir = os.path.join(DATASET_ROOT, 'images')
    
    class_samples = defaultdict(list)
    all_labels = []
    
    print("正在扫描数据集...")
    for filename in os.listdir(label_dir):
        if not filename.endswith('.txt'):
            continue
        
        label_path = os.path.join(label_dir, filename)
        # 跳过空文件
        if os.path.getsize(label_path) == 0:
            continue
        
        # 读取标签文件，获取第一个目标的类别（按主类别分组）
        with open(label_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if not first_line:
                continue
            class_id = int(first_line.split()[0])
            class_samples[class_id].append(filename)
            all_labels.append(filename)
    
    # 3. 统计原始数据集信息
    total_samples = len(all_labels)
    print(f"\n原始数据集统计：")
    print(f"总样本数：{total_samples} 张")
    print(f"类别数：{len(class_samples)} 类")
    for cls_id, samples in sorted(class_samples.items()):
        print(f"  类别 {cls_id}: {len(samples)} 张")
    
    # 4. 按类别进行8:1:1划分
    train_files = []
    val_files = []
    test_files = []
    
    print("\n正在按类别均衡划分...")
    for cls_id, samples in sorted(class_samples.items()):
        random.shuffle(samples)
        n = len(samples)
        
        # 计算每个划分的样本数
        n_train = int(n * SPLIT_RATIO[0])
        n_val = int(n * SPLIT_RATIO[1])
        n_test = n - n_train - n_val
        
        # 处理边界情况（样本数太少时）
        if n == 1:
            n_train, n_val, n_test = 1, 0, 0
        elif n == 2:
            n_train, n_val, n_test = 1, 1, 0
        elif n == 3:
            n_train, n_val, n_test = 2, 1, 0
        
        # 分配样本
        train_files.extend(samples[:n_train])
        val_files.extend(samples[n_train:n_train+n_val])
        test_files.extend(samples[n_train+n_val:])
        
        print(f"  类别 {cls_id}: train={n_train}, val={n_val}, test={n_test}")
    
    # 5. 打乱最终划分结果（避免同类别集中）
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)
    
    # 6. 复制文件到输出目录
    def copy_files(file_list, split):
        for filename in file_list:
            # 复制标签文件
            src_label = os.path.join(label_dir, filename)
            dst_label = os.path.join(OUTPUT_ROOT, 'labels', split, filename)
            shutil.copy2(src_label, dst_label)
            
            # 复制对应的图片文件
            base_name = os.path.splitext(filename)[0]
            for ext in IMAGE_EXTENSIONS:
                src_image = os.path.join(image_dir, base_name + ext)
                if os.path.exists(src_image):
                    dst_image = os.path.join(OUTPUT_ROOT, 'images', split, base_name + ext)
                    shutil.copy2(src_image, dst_image)
                    break
            else:
                print(f"警告：未找到图片文件 {base_name}")
    
    print("\n正在复制文件...")
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    
    # 7. 打印最终划分结果
    print("\n" + "="*50)
    print("数据集划分完成！")
    print("="*50)
    print(f"输出目录：{OUTPUT_ROOT}")
    print(f"训练集：{len(train_files)} 张 ({len(train_files)/total_samples*100:.1f}%)")
    print(f"验证集：{len(val_files)} 张 ({len(val_files)/total_samples*100:.1f}%)")
    print(f"测试集：{len(test_files)} 张 ({len(test_files)/total_samples*100:.1f}%)")
    print("="*50)
    print("✅ 可以直接使用以下路径进行训练：")
    print(f"   train: {os.path.join(OUTPUT_ROOT, 'images/train')}")
    print(f"   val: {os.path.join(OUTPUT_ROOT, 'images/val')}")
    print(f"   test: {os.path.join(OUTPUT_ROOT, 'images/test')}")

if __name__ == "__main__":
    main()
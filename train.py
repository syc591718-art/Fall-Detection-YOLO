# train.py - YOLOv8姿态检测训练脚本（修复路径嵌套问题）
from ultralytics import YOLO
import torch
import os
import shutil
from datetime import datetime
from pathlib import Path


def ensure_correct_detect_dir(project_root):
    """确保 runs/detect 目录正确，避免嵌套"""
    detect_dir = project_root / 'runs' / 'detect'

    print("\n🔍 检查目录结构...")

    # 情况1：检查是否有嵌套的 runs/detect/runs/detect
    nested_dir = detect_dir / 'runs' / 'detect'
    if nested_dir.exists():
        print("⚠️  发现嵌套目录，正在修复...")

        # 将嵌套目录中的训练结果移动到正确位置
        moved_count = 0
        for train_dir in nested_dir.iterdir():
            if train_dir.is_dir() and train_dir.name.startswith('train_'):
                target_dir = detect_dir / train_dir.name
                if not target_dir.exists():
                    print(f"  移动 {train_dir.name} 到正确位置")
                    shutil.move(str(train_dir), str(detect_dir))
                    moved_count += 1

        # 删除空的嵌套目录
        if (detect_dir / 'runs').exists():
            shutil.rmtree(str(detect_dir / 'runs'))
            print(f"✅ 已清理嵌套目录，移动了 {moved_count} 个模型")

    # 情况2：检查是否有模型直接放在 detect 下（正确位置）
    correct_models = []
    if detect_dir.exists():
        for item in detect_dir.iterdir():
            if item.is_dir() and item.name.startswith('train_'):
                weights_path = item / 'weights' / 'best.pt'
                if weights_path.exists():
                    correct_models.append(item.name)

    if correct_models:
        print(f"✅ 找到 {len(correct_models)} 个正确位置的模型:")
        for model_name in correct_models[-5:]:  # 显示最近5个
            print(f"   • {model_name}")
    else:
        print("📁 没有找到已训练的模型，将创建新目录")

    # 确保目录存在
    detect_dir.mkdir(parents=True, exist_ok=True)
    return detect_dir


def check_gpu():
    """检查GPU可用性"""
    print("🔍 检查硬件环境...")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✅ 检测到 {device_count} 个GPU:")

        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

            # 显示当前显存使用情况
            allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
            cached = torch.cuda.memory_reserved(i) / 1024 ** 3
            print(f"      已分配: {allocated:.2f}GB, 已缓存: {cached:.2f}GB")

        return 0  # 使用第一个GPU
    else:
        print("⚠  未检测到GPU，使用CPU训练（速度较慢）")
        return "cpu"


def check_dataset():
    """检查数据集完整性"""
    print("\n📂 检查数据集...")

    yaml_path = "data/dataset/dataset.yaml"
    if not os.path.exists(yaml_path):
        print(f"❌ 配置文件不存在: {yaml_path}")
        return False

    # 检查数据集目录
    required_dirs = [
        "data/dataset/images/train",
        "data/dataset/images/val",
        "data/dataset/labels/train",
        "data/dataset/labels/val"
    ]

    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ 目录不存在: {dir_path}")
            return False

    # 统计样本数量
    train_imgs = len(os.listdir("data/dataset/images/train"))
    val_imgs = len(os.listdir("data/dataset/images/val"))

    print(f"  训练集: {train_imgs} 张图片")
    print(f"  验证集: {val_imgs} 张图片")

    if train_imgs == 0 or val_imgs == 0:
        print("❌ 数据集为空，请先运行数据转换脚本")
        return False

    return True


def get_class_distribution():
    """获取类别分布"""
    import yaml

    yaml_path = "data/dataset/dataset.yaml"
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    class_names = data.get('names', [])

    # 统计训练集标签
    label_dir = "data/dataset/labels/train"
    class_counts = {}

    if os.path.exists(label_dir):
        for label_file in os.listdir(label_dir):
            if label_file.endswith('.txt'):
                with open(os.path.join(label_dir, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1

    return class_counts, class_names


def main():
    """主训练函数"""
    print("=" * 60)
    print("YOLOv8 姿态检测 - 训练脚本")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 获取项目根目录
    project_root = Path(__file__).parent.absolute()

    # === 新增：确保目录正确，避免嵌套 ===
    detect_dir = ensure_correct_detect_dir(project_root)
    print(f"📁 模型将保存到: {detect_dir}")
    # ================================

    # 1. 检查数据集
    if not check_dataset():
        print("\n❌ 数据集检查失败，请先完成数据准备")
        return

    # 2. 获取类别分布
    class_counts, class_names = get_class_distribution()
    if class_counts:
        print("\n📊 类别分布:")
        total = sum(class_counts.values())
        for class_id, count in sorted(class_counts.items()):
            if class_id < len(class_names):
                name = class_names[class_id]
                percentage = count / total * 100
                print(f"   {name}: {count} 个目标 ({percentage:.1f}%)")

    # 3. 检查GPU并清理内存
    device = check_gpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU内存已清理")

    # 4. 选择训练模式
    print("\n📦 选择训练模式:")
    print("  1. 从头开始训练（使用yolov8n.pt）")
    print("  2. 继续训练（使用已有的best.pt）")

    choice = input("请选择 (1/2，默认2): ").strip()

    if choice == "1":
        model_path = "yolov8n.pt"
        mode = "从头训练"
    else:
        # 自动查找最新的best.pt（只搜索正确位置）
        best_models = []
        if detect_dir.exists():
            for model_dir in detect_dir.iterdir():
                if model_dir.is_dir() and model_dir.name.startswith('train_'):
                    weights_path = model_dir / 'weights' / 'best.pt'
                    if weights_path.exists():
                        best_models.append(weights_path)

        if best_models:
            # 按修改时间排序，取最新的
            best_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            model_path = str(best_models[0])
            model_name = best_models[0].parent.parent.name
            mode = f"继续训练 (使用 {model_name})"
        else:
            model_path = "yolov8n.pt"
            mode = "从头训练（未找到已有模型）"

    print(f"\n🎯 训练模式: {mode}")
    print(f"   模型文件: {os.path.basename(model_path)}")

    # 5. 加载模型
    try:
        model = YOLO(model_path)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 6. 训练参数配置
    timestamp = datetime.now().strftime("%m%d_%H%M")

    train_args = {
        # ===== 基础配置 =====
        'data': 'data/dataset/dataset.yaml',
        'epochs': 120,
        'patience': 30,
        'batch': 4,
        'imgsz': 640,
        'device': device,

        # ===== 优化器配置 =====
        'optimizer': 'auto',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,

        # ===== 数据增强（优化版）=====
        'augment': True,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 15.0,  # 旋转增强
        'translate': 0.2,  # 平移增强
        'scale': 0.5,  # 缩放增强
        'shear': 5.0,  # 剪切增强
        'perspective': 0.001,  # 透视变换
        'flipud': 0.1,  # 上下翻转
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.2,  # 混合增强
        'copy_paste': 0.2,  # 复制粘贴

        # ===== 训练策略 =====
        'cos_lr': True,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'close_mosaic': 10,

        # ===== 显存优化 =====
        'workers': 4,
        'cache': False,
        'amp': True,
        'fraction': 1.0,

        # ===== 损失函数 =====
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'label_smoothing': 0.1,

        # ===== 输出配置 =====
        'save': True,
        'save_period': 20,
        'exist_ok': True,
        'pretrained': True,
        'val': True,
        'plots': True,
        'verbose': True,

        # ===== 项目命名（确保正确路径）=====
        'project': 'runs/detect',
        'name': f'train_{timestamp}',
    }

    # 7. 显示训练配置摘要
    print(f"\n⚙️  训练配置:")
    print(f"  训练轮数: {train_args['epochs']}")
    print(f"  批量大小: {train_args['batch']}")
    print(f"  图像尺寸: {train_args['imgsz']}")
    print(f"  工作线程: {train_args['workers']}")
    print(f"  输出目录: {train_args['project']}/{train_args['name']}")
    print(f"  ✅ 路径正确，不会产生嵌套")

    # 8. 开始训练
    print("\n" + "=" * 60)
    print("🚀 开始训练...")
    print("=" * 60)
    print("-" * 60)

    try:
        results = model.train(**train_args)

        # 9. 训练完成提示
        print("\n" + "=" * 60)
        print("✅ 训练完成！")
        print("=" * 60)

        # 显示最佳模型路径
        best_model_path = f"{train_args['project']}/{train_args['name']}/weights/best.pt"
        if os.path.exists(best_model_path):
            print(f"\n📁 最佳模型已保存:")
            print(f"   {best_model_path}")
            print(f"   ✅ 路径正确，无嵌套")

            # 计算模型大小
            model_size = os.path.getsize(best_model_path) / 1024 ** 2
            print(f"   模型大小: {model_size:.1f} MB")

            print(f"\n🎯 下一步建议:")
            print(f"  1. 测试模型: python detect.py")
            print(f"  2. 查看训练曲线: tensorboard --logdir=runs/detect")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n❌ 显存不足！建议:")
            print("  1. 将 batch 改为 2")
            print("  2. 将 workers 改为 2")
            print("  3. 关闭其他占用显存的程序")
        else:
            print(f"\n❌ 训练过程出错: {e}")

    except Exception as e:
        print(f"\n❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU内存已清理")

        print(f"\n🕒 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)


if __name__ == "__main__":
    main()
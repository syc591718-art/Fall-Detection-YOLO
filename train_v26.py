# train_v26.py - 使用YOLOv26训练自定义姿态检测模型（彻底修复嵌套问题）
from ultralytics import YOLO
import torch
import os
import shutil
from datetime import datetime
from pathlib import Path


def check_environment():
    """检查环境和YOLO版本"""
    print("=" * 60)
    print("🔍 检查YOLOv26环境")
    print("=" * 60)

    # 检查ultralytics版本
    import ultralytics
    print(f"📦 ultralytics版本: {ultralytics.__version__}")

    # 检查GPU
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✅ GPU可用！检测到 {device_count} 个设备")
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

        # 清理显存
        torch.cuda.empty_cache()
        print("🧹 GPU内存已清理")
        return 0
    else:
        print("⚠️  GPU不可用，使用CPU训练（速度较慢）")
        return "cpu"


def check_dataset():
    """检查数据集"""
    print("\n📂 检查数据集...")

    yaml_path = "data/dataset/dataset.yaml"
    if not os.path.exists(yaml_path):
        print(f"❌ 配置文件不存在: {yaml_path}")
        return False

    # 读取数据集信息
    import yaml
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    print(f"📊 数据集信息:")
    print(f"   类别: {data.get('names', {})}")
    print(f"   训练集: {data.get('train', 'N/A')}")
    print(f"   验证集: {data.get('val', 'N/A')}")

    # 检查目录
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

    # 统计样本
    train_imgs = len(os.listdir("data/dataset/images/train"))
    val_imgs = len(os.listdir("data/dataset/images/val"))

    print(f"\n📈 数据统计:")
    print(f"   训练集图片: {train_imgs} 张")
    print(f"   验证集图片: {val_imgs} 张")

    return True


def get_class_distribution():
    """获取类别分布"""
    import yaml
    from collections import Counter

    yaml_path = "data/dataset/dataset.yaml"
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    class_names = data.get('names', [])

    # 统计训练集
    label_dir = "data/dataset/labels/train"
    class_counts = Counter()

    if os.path.exists(label_dir):
        for label_file in os.listdir(label_dir):
            if label_file.endswith('.txt'):
                with open(os.path.join(label_dir, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1

    return class_counts, class_names


def ensure_clean_output_dir(project_root):
    """确保输出目录干净，没有嵌套"""
    output_dir = project_root / 'runs' / 'detect_v26'

    # 检查是否有嵌套的旧目录
    nested_dirs = [
        project_root / 'runs' / 'detect' / 'runs',
        project_root / 'runs' / 'detect_v26' / 'runs',
    ]

    for nested in nested_dirs:
        if nested.exists():
            print(f"⚠️  发现嵌套目录: {nested}")
            # 将嵌套中的模型移到正确位置
            if nested / 'detect_v26' in nested.iterdir():
                for item in (nested / 'detect_v26').iterdir():
                    if item.is_dir() and item.name.startswith('train_'):
                        target = output_dir / item.name
                        if not target.exists():
                            print(f"  移动 {item.name} 到正确位置")
                            shutil.move(str(item), str(output_dir))
            # 删除嵌套目录
            shutil.rmtree(str(nested.parent))
            print("✅ 嵌套目录已清理")

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    """主训练函数"""
    print("=" * 60)
    print("🎯 YOLOv26 姿态检测训练脚本")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 获取项目根目录
    project_root = Path(__file__).parent.absolute()

    # 1. 清理可能存在的嵌套目录
    output_dir = ensure_clean_output_dir(project_root)
    print(f"📁 输出目录: {output_dir}")

    # 2. 检查环境
    device = check_environment()

    # 3. 检查数据集
    if not check_dataset():
        print("\n❌ 数据集检查失败，请先准备数据")
        return

    # 4. 获取类别分布
    class_counts, class_names = get_class_distribution()
    if class_counts:
        print("\n📊 训练集类别分布:")
        total = sum(class_counts.values())
        for class_id, count in sorted(class_counts.items()):
            if class_id < len(class_names):
                name = class_names[class_id]
                percentage = count / total * 100
                bar = "█" * int(percentage / 2)
                print(f"   {name}: {count} 个 ({percentage:.1f}%) {bar}")

    # 5. 检查yolo26n.pt是否存在
    if not os.path.exists('yolo26n.pt'):
        print("\n⚠️  未找到 yolo26n.pt")
        print("   请确保 yolo26n.pt 在当前目录")
        response = input("是否继续使用其他模型？(y/n): ").strip().lower()
        if response != 'y':
            return
        model_path = input("请输入模型路径 (默认: yolov8n.pt): ").strip() or 'yolov8n.pt'
    else:
        model_path = 'yolo26n.pt'
        print(f"\n✅ 找到 YOLOv26 模型: {model_path}")

    # 6. 加载模型
    try:
        print(f"\n🔄 加载模型 {model_path}...")
        model = YOLO(model_path)
        print(f"✅ 模型加载成功")
        print(f"   原始类别数: {len(model.names)}")
        print(f"   将微调为你的3个类别: up, bending, down")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 7. 定义timestamp
    timestamp = datetime.now().strftime("%m%d_%H%M")

    # 8. 训练参数配置（彻底修复嵌套问题）
    train_args = {
        # ===== 基础配置 =====
        'data': 'data/dataset/dataset.yaml',
        'epochs': 150,
        'patience': 30,
        'batch': 4,
        'imgsz': 640,
        'device': device,

        # ===== 优化器配置 =====
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,

        # ===== 数据增强 =====
        'augment': True,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 15.0,
        'translate': 0.2,
        'scale': 0.5,
        'shear': 5.0,
        'perspective': 0.001,
        'flipud': 0.1,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.2,
        'copy_paste': 0.2,

        # ===== 训练策略 =====
        'cos_lr': True,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'close_mosaic': 10,

        # ===== 显存优化 =====
        'workers': 2,  # Windows用2更稳定
        'cache': False,
        'amp': True,

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

        # ===== 项目命名（彻底修复嵌套）=====
        'project': 'runs',  # 只写到 runs
        'name': f'detect_v26/train_{timestamp}',  # 直接 detect_v26/train_时间戳
    }

    # 9. 显示训练配置
    print("\n" + "=" * 60)
    print("⚙️  训练配置:")
    print("=" * 60)
    print(f"   基础模型: {os.path.basename(model_path)}")
    print(f"   训练轮数: {train_args['epochs']}")
    print(f"   批次大小: {train_args['batch']}")
    print(f"   图像尺寸: {train_args['imgsz']}")
    print(f"   输出目录: runs/detect_v26/train_{timestamp}")
    print(f"   ✅ 路径正确，绝对不会产生嵌套")
    print("=" * 60)

    # 10. 确认开始
    print("\n🚀 准备开始训练...")
    print("训练后模型将只输出 up/bending/down 三个类别")
    choice = input("是否开始训练？(y/n, 默认y): ").strip().lower()
    if choice == 'n':
        print("训练已取消")
        return

    # 11. 开始训练
    print("\n" + "=" * 60)
    print("🔥 开始训练...")
    print("=" * 60)

    try:
        results = model.train(**train_args)

        # 12. 训练完成
        print("\n" + "=" * 60)
        print("✅ 训练完成！")
        print("=" * 60)

        # 显示最佳模型路径
        best_model_path = f"runs/detect_v26/train_{timestamp}/weights/best.pt"
        if os.path.exists(best_model_path):
            print(f"\n📁 最佳模型已保存:")
            print(f"   {best_model_path}")
            print(f"   ✅ 路径正确，无嵌套")

            # 验证训练后的类别
            trained_model = YOLO(best_model_path)
            print(f"\n🎯 训练后的类别:")
            for cls_id, cls_name in trained_model.names.items():
                print(f"   {cls_id}: {cls_name}")

            # 模型大小
            model_size = os.path.getsize(best_model_path) / 1024 ** 2
            print(f"\n📦 模型大小: {model_size:.1f} MB")

            print(f"\n🎉 现在可以用这个模型检测 up/bending/down 了!")
            print(f"   测试命令: python detect.py --weights {best_model_path}")

    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\n🧹 GPU内存已清理")

        print(f"\n🕒 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)


if __name__ == "__main__":
    main()
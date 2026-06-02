"""
跌倒检测数据集处理工具
功能：将原始数据（图片+XML标签）转换为YOLOv8格式
"""

import os
import shutil
import random
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml


class FallDatasetProcessor:
    """跌倒检测数据集处理器"""

    def __init__(self):
        # 定义三个跌倒检测类别
        self.classes = ['up', 'bending', 'down']
        print(f"初始化处理器，检测类别: {self.classes}")

    def convert_voc_to_yolo(self, xml_file, img_file, output_dir):
        """
        将单个VOC格式的XML文件转换为YOLO格式的TXT文件

        参数:
            xml_file: VOC格式的XML标签文件路径
            img_file: 对应的图片文件路径
            output_dir: 输出TXT文件的目录
        """
        try:
            # 检查文件是否存在
            if not Path(xml_file).exists():
                print(f"  警告: XML文件不存在 {xml_file}")
                return False

            # 读取图片尺寸（用于坐标归一化）
            # 注意：这里简化处理，实际可能需要用PIL或OpenCV读取
            # 我们假设图片存在且可读
            img_width, img_height = 640, 480  # 默认值，实际应该从图片读取

            # 解析XML文件
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # 获取图片尺寸（从XML中）
            size = root.find('size')
            if size is not None:
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)

            yolo_lines = []  # 存储YOLO格式的行

            # 遍历所有目标对象
            for obj in root.findall('object'):
                # 获取类别名称
                cls_name = obj.find('name').text

                # 检查是否是我们定义的类别
                if cls_name not in self.classes:
                    continue

                # 获取类别ID
                cls_id = self.classes.index(cls_name)

                # 获取边界框坐标
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)

                # 转换为YOLO格式（归一化坐标）
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                # 确保坐标在合理范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                # 添加到列表
                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # 如果有检测目标，保存TXT文件
            if yolo_lines:
                # 生成输出文件名（与图片同名，扩展名为.txt）
                txt_filename = Path(xml_file).stem + '.txt'
                txt_path = Path(output_dir) / txt_filename

                # 写入文件
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_lines))

                return True
            else:
                # 如果没有目标，创建空文件
                txt_filename = Path(xml_file).stem + '.txt'
                txt_path = Path(output_dir) / txt_filename
                open(txt_path, 'w').close()  # 创建空文件
                return True

        except Exception as e:
            print(f"  错误: 转换 {xml_file} 时出错: {e}")
            return False

    def prepare_dataset(self, source_dir='data/raw', output_dir='data/dataset', train_ratio=0.8):
        """
        准备YOLOv8格式的数据集

        参数:
            source_dir: 原始数据目录（包含图片和XML文件）
            output_dir: 输出数据集目录
            train_ratio: 训练集比例（0-1之间）
        """
        print("=" * 60)
        print("开始准备跌倒检测数据集")
        print(f"原始数据目录: {source_dir}")
        print(f"输出目录: {output_dir}")
        print("=" * 60)

        # 创建输出目录结构
        dirs_to_create = [
            f"{output_dir}/images/train",
            f"{output_dir}/images/val",
            f"{output_dir}/labels/train",
            f"{output_dir}/labels/val"
        ]

        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"创建目录: {dir_path}")

        # 查找所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']
        all_images = []

        for ext in image_extensions:
            all_images.extend(list(Path(source_dir).glob(f'*{ext}')))

        if not all_images:
            print(f"❌ 错误: 在 {source_dir} 中没有找到图片文件")
            print("   支持的格式: .jpg, .jpeg, .png, .bmp")
            return None

        print(f"找到 {len(all_images)} 张图片")

        # 随机打乱并划分数据集
        random.shuffle(all_images)
        split_idx = int(len(all_images) * train_ratio)
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]

        print(f"训练集: {len(train_images)} 张图片")
        print(f"验证集: {len(val_images)} 张图片")

        # 处理训练集
        train_success = 0
        print("\n处理训练集...")
        for img_path in train_images:
            # 复制图片到训练集目录
            dst_img = Path(output_dir) / 'images' / 'train' / img_path.name
            shutil.copy2(img_path, dst_img)

            # 查找对应的XML文件
            xml_path = img_path.with_suffix('.xml')

            # 转换标签文件
            if xml_path.exists():
                if self.convert_voc_to_yolo(xml_path, img_path,
                                            Path(output_dir) / 'labels' / 'train'):
                    train_success += 1
            else:
                # 如果没有XML文件，创建空的标签文件
                txt_path = Path(output_dir) / 'labels' / 'train' / img_path.with_suffix('.txt').name
                open(txt_path, 'w').close()

        # 处理验证集
        val_success = 0
        print("\n处理验证集...")
        for img_path in val_images:
            # 复制图片到验证集目录
            dst_img = Path(output_dir) / 'images' / 'val' / img_path.name
            shutil.copy2(img_path, dst_img)

            # 查找对应的XML文件
            xml_path = img_path.with_suffix('.xml')

            # 转换标签文件
            if xml_path.exists():
                if self.convert_voc_to_yolo(xml_path, img_path,
                                            Path(output_dir) / 'labels' / 'val'):
                    val_success += 1
            else:
                # 如果没有XML文件，创建空的标签文件
                txt_path = Path(output_dir) / 'labels' / 'val' / img_path.with_suffix('.txt').name
                open(txt_path, 'w').close()

        print(f"\n处理完成:")
        print(f"  训练集: {train_success}/{len(train_images)} 个标签文件")
        print(f"  验证集: {val_success}/{len(val_images)} 个标签文件")

        # 创建或更新 dataset.yaml 文件
        self.update_dataset_yaml(output_dir)

        print("=" * 60)
        print("✅ 数据集准备完成!")
        print(f"数据集已保存到: {output_dir}")
        print(f"配置文件: {output_dir}/dataset.yaml")
        print("=" * 60)

        return f"{output_dir}/dataset.yaml"

    def update_dataset_yaml(self, dataset_dir):
        """创建或更新数据集配置文件"""
        yaml_content = {
            'path': str(Path(dataset_dir).absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.classes),
            'names': {i: name for i, name in enumerate(self.classes)}
        }

        yaml_path = Path(dataset_dir) / 'dataset.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        print(f"更新配置文件: {yaml_path}")


# 使用示例
if __name__ == "__main__":
    print("跌倒检测数据集处理工具")
    print("=" * 50)

    # 创建处理器实例
    processor = FallDatasetProcessor()

    # 检查原始数据目录
    raw_dir = 'data/raw'
    if not Path(raw_dir).exists():
        print(f"❌ 原始数据目录不存在: {raw_dir}")
        print(f"请创建目录并将你的图片和XML文件放入: {raw_dir}")
        print("\n目录结构示例:")
        print(f"{raw_dir}/")
        print("  ├── image1.jpg")
        print("  ├── image1.xml")
        print("  ├── image2.jpg")
        print("  └── image2.xml")
    else:
        # 开始处理数据
        print(f"找到原始数据目录: {raw_dir}")
        response = input("是否开始处理数据？(y/n): ")

        if response.lower() == 'y':
            processor.prepare_dataset(source_dir=raw_dir, output_dir='data/dataset')
        else:
            print("取消处理。")

    print("\n手动运行方式:")
    print("  from utils.data_utils import FallDatasetProcessor")
    print("  processor = FallDatasetProcessor()")
    print("  processor.prepare_dataset('data/raw', 'data/dataset')")
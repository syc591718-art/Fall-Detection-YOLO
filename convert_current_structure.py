# convert_current_structure.py
import os
import shutil
import random
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

# 固定随机种子，确保每次分割结果一致
random.seed(42)
np.random.seed(42)


class DataConverter:
    def __init__(self):
        self.base_dir = "data"
        self.raw_dir = os.path.join(self.base_dir, "raw")
        self.dataset_dir = os.path.join(self.base_dir, "dataset")

        # 检查目录是否存在
        print("检查目录结构...")
        print(f"原始数据目录: {self.raw_dir} - {os.path.exists(self.raw_dir)}")
        print(f"数据集目录: {self.dataset_dir} - {os.path.exists(self.dataset_dir)}")

        # 类别映射
        self.class_mapping = {
            'up': 0,
            'standing': 0,
            'bending': 1,
            'down': 2
        }

    def check_raw_files(self):
        """检查原始文件"""
        print("\n检查原始文件...")
        files = list(Path(self.raw_dir).glob("*"))
        file_types = {}

        for file in files:
            ext = file.suffix.lower()
            if ext not in file_types:
                file_types[ext] = []
            file_types[ext].append(file.name)

        for ext, names in file_types.items():
            print(f"  {ext}: {len(names)} 个文件 - {', '.join(names[:3])}{'...' if len(names) > 3 else ''}")

        return files

    def convert_xml_to_yolo(self, xml_path):
        """转换单个XML文件为YOLO格式"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 获取图片尺寸
            size_elem = root.find('size')
            if size_elem is not None:
                img_width = float(size_elem.find('width').text)
                img_height = float(size_elem.find('height').text)
            else:
                # 如果没有尺寸信息，尝试从图片获取
                img_filename = root.find('filename').text or Path(xml_path).stem
                img_found = False

                # 支持多种图片格式
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']:
                    img_path = os.path.join(self.raw_dir, img_filename + ext)
                    if os.path.exists(img_path):
                        try:
                            from PIL import Image
                            with Image.open(img_path) as img:
                                img_width, img_height = img.size
                            img_found = True
                            break
                        except:
                            continue

                if not img_found:
                    img_width, img_height = 640, 480
                    print(f"  警告: {xml_path.name} 使用默认尺寸 640x480")

            # 获取对象信息
            objects = []
            for obj in root.findall('object'):
                cls_name = obj.find('name').text.lower()

                # 标准化类别名称
                if cls_name not in self.class_mapping:
                    for key in self.class_mapping:
                        if key in cls_name:
                            cls_name = key
                            break

                if cls_name in self.class_mapping:
                    bbox = obj.find('bndbox')
                    if bbox is not None:
                        xmin = float(bbox.find('xmin').text)
                        ymin = float(bbox.find('ymin').text)
                        xmax = float(bbox.find('xmax').text)
                        ymax = float(bbox.find('ymax').text)

                        # 转换为YOLO格式（归一化坐标）
                        x_center = (xmin + xmax) / 2 / img_width
                        y_center = (ymin + ymax) / 2 / img_height
                        width = (xmax - xmin) / img_width
                        height = (ymax - ymin) / img_height

                        # 限制在0-1范围内
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))

                        objects.append({
                            'class_id': self.class_mapping[cls_name],
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })

            return objects, img_width, img_height

        except Exception as e:
            print(f"  错误: 转换 {xml_path.name} 时出错: {str(e)}")
            return None, None, None

    def prepare_dataset_structure(self):
        """准备数据集目录结构"""
        print("\n准备数据集目录结构...")

        # 创建子目录
        subdirs = [
            'images/train', 'images/val', 'images/test',
            'labels/train', 'labels/val', 'labels/test'
        ]

        for subdir in subdirs:
            full_path = os.path.join(self.dataset_dir, subdir)
            os.makedirs(full_path, exist_ok=True)
            print(f"  创建: {subdir}")

    def process_files(self):
        """处理所有文件（支持多种图片格式）"""
        print("\n开始处理文件...")

        # 获取所有XML文件
        xml_files = list(Path(self.raw_dir).glob("*.xml"))
        print(f"找到 {len(xml_files)} 个XML文件")

        # 收集所有有效的数据对
        data_pairs = []

        for xml_file in xml_files:
            print(f"处理: {xml_file.name}", end=" ")

            # 转换XML
            objects, img_width, img_height = self.convert_xml_to_yolo(xml_file)

            if objects is not None and len(objects) > 0:
                # 查找对应的图片文件（支持多种格式）
                img_stem = xml_file.stem
                img_found = False

                # 支持的图片格式
                img_extensions = ['.jpg', '.jpeg', '.png', '.bmp',
                                  '.JPG', '.JPEG', '.PNG', '.BMP']

                for ext in img_extensions:
                    img_path = os.path.join(self.raw_dir, img_stem + ext)
                    if os.path.exists(img_path):
                        # 保存YOLO格式标签
                        label_filename = img_stem + '.txt'
                        label_path = os.path.join(self.dataset_dir, 'labels', label_filename)

                        with open(label_path, 'w') as f:
                            for obj in objects:
                                f.write(f"{obj['class_id']} {obj['x_center']:.6f} {obj['y_center']:.6f} "
                                        f"{obj['width']:.6f} {obj['height']:.6f}\n")

                        data_pairs.append({
                            'image': img_path,
                            'label': label_path,
                            'name': img_stem,
                            'class_count': len(objects)
                        })

                        print(f"✓ 找到图片: {ext}")
                        img_found = True
                        break

                if not img_found:
                    print(f"✗ 未找到图片文件")
            else:
                print(f"✗ 转换失败或无对象")

        print(f"\n总计: 成功处理 {len(data_pairs)} 个数据对")

        # 统计类别分布
        if data_pairs:
            class_dist = {}
            for pair in data_pairs:
                label_path = pair['label']
                with open(label_path, 'r') as f:
                    for line in f:
                        cls_id = int(line.split()[0])
                        class_name = {0: 'up', 1: 'bending', 2: 'down'}.get(cls_id, f'cls{cls_id}')
                        class_dist[class_name] = class_dist.get(class_name, 0) + 1

            print("类别分布:")
            for cls_name, count in class_dist.items():
                print(f"  {cls_name}: {count} 个对象")

        return data_pairs

    def split_dataset(self, data_pairs):
        """分割数据集（固定随机种子确保一致性）"""
        print("\n分割数据集（随机种子固定为42）...")

        if len(data_pairs) == 0:
            print("错误: 没有可用的数据对")
            return

        # 固定顺序：先按文件名排序，再随机打乱（但随机种子固定）
        data_pairs.sort(key=lambda x: x['name'])
        random.shuffle(data_pairs)  # 由于种子固定，每次shuffle结果相同

        # 分割比例
        n_total = len(data_pairs)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.2)

        # 确保每个集合至少有样本
        if n_train < 1:
            n_train = max(1, n_total - 2)
        if n_val < 1:
            n_val = 1 if n_total > 1 else 0

        splits = {
            'train': data_pairs[:n_train],
            'val': data_pairs[n_train:n_train + n_val],
            'test': data_pairs[n_train + n_val:]
        }

        print(f"数据分割: 总共 {n_total} 个样本")
        print(f"  训练集: {len(splits['train'])} 个样本")
        print(f"  验证集: {len(splits['val'])} 个样本")
        print(f"  测试集: {len(splits['test'])} 个样本")

        # 复制文件到对应目录
        for split_name, pairs in splits.items():
            print(f"\n{split_name} 集 ({len(pairs)} 个样本):")

            for pair in pairs:
                # 复制图片
                img_src = pair['image']
                img_dst = os.path.join(self.dataset_dir, 'images', split_name, Path(img_src).name)
                shutil.copy2(img_src, img_dst)

                # 复制标签
                label_src = pair['label']
                label_dst = os.path.join(self.dataset_dir, 'labels', split_name, Path(label_src).name)
                shutil.copy2(label_src, label_dst)

                print(f"  ✓ {pair['name']} ({pair['class_count']}个对象)")

    def update_dataset_yaml(self):
        """更新dataset.yaml文件"""
        print("\n更新配置文件...")

        yaml_path = os.path.join(self.dataset_dir, "dataset.yaml")

        # 读取现有内容或创建新内容
        yaml_content = f"""# YOLOv8跌倒检测数据集
# 自动生成于: {Path(yaml_path).parent}
# 随机种子: 42 (确保分割一致性)

path: {os.path.abspath(self.dataset_dir)}
train: images/train
val: images/val
test: images/test

# 类别数量
nc: 3

# 类别名称
names:
  0: up
  1: bending
  2: down
"""

        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)

        print(f"✅ 配置文件已更新: {yaml_path}")
        print(f"   类别数量: {len(self.class_mapping)}")
        print(f"   类别: {list(self.class_mapping.keys())[:len(self.class_mapping)]}")

    def cleanup_temp_files(self):
        """清理临时文件"""
        print("\n清理临时文件...")

        # 删除根labels文件夹中的临时文件
        labels_root = os.path.join(self.dataset_dir, 'labels')
        temp_files = list(Path(labels_root).glob("*.txt"))

        for file in temp_files:
            if file.is_file():
                os.remove(file)
                print(f"  删除临时文件: {file.name}")

    def run(self):
        """运行完整转换流程"""
        print("=" * 60)
        print("YOLOv8数据转换工具 (支持PNG/JPG多种格式)")
        print("=" * 60)

        # 1. 检查文件
        self.check_raw_files()

        # 2. 准备目录结构
        self.prepare_dataset_structure()

        # 3. 处理文件
        data_pairs = self.process_files()

        if len(data_pairs) == 0:
            print("\n错误: 没有成功处理任何文件，请检查原始数据！")
            print("可能问题:")
            print("  1. XML文件格式错误")
            print("  2. 图片和XML文件名不匹配")
            print("  3. XML中缺少必要的标签")
            return False

        # 4. 分割数据集
        self.split_dataset(data_pairs)

        # 5. 更新配置文件
        self.update_dataset_yaml()

        # 6. 清理临时文件
        self.cleanup_temp_files()

        print("\n" + "=" * 60)
        print("✅ 数据转换完成！")
        print(f"数据集位置: {os.path.abspath(self.dataset_dir)}")
        print(f"训练样本: {len(list(Path(self.dataset_dir).glob('images/train/*')))}")
        print(f"验证样本: {len(list(Path(self.dataset_dir).glob('images/val/*')))}")
        print(f"测试样本: {len(list(Path(self.dataset_dir).glob('images/test/*')))}")
        print("\n重要: 由于随机种子固定为42，每次运行的分割结果相同")
        print("=" * 60)

        return True


if __name__ == "__main__":
    converter = DataConverter()
    success = converter.run()

    if success:
        print("\n🎉 下一步: 运行训练脚本")
        print("命令: python train.py")
    else:
        print("\n❌ 转换失败，请检查错误信息")
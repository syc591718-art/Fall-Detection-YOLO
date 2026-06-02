# 基于 YOLO 的摔倒行为检测系统

基于 YOLOv8 / YOLO26 的摔倒行为检测系统，支持视频流实时检测、GUI 拖放检测。

## 技术栈

| 类别 | 技术 |
| :--- | :--- |
| 深度学习框架 | PyTorch |
| 目标检测 | YOLOv8、YOLO26 |
| GUI 界面 | PyQt5（detect_gui.py） |
| 数据处理 | OpenCV、NumPy |
| 编程语言 | Python |

## 功能

- ✅ YOLOv8 / YOLO26 模型训练与验证
- ✅ 命令行视频/图片检测
- ✅ GUI 拖放检测（支持图片、视频）
- ✅ VOC 格式转 YOLO 格式
- ✅ 检测结果可视化（边框、置信度）

## 工程结构
Fall-Detection-YOLOv8/
├── data/ # 数据集
│ ├── dataset/ # YOLO 格式数据集
│ │ ├── images/ # 图片（train/val/test）
│ │ ├── labels/ # 标注（train/val/test）
│ │ └── dataset.yaml # 数据集配置
│ └── raw/ # 原始 VOC 格式数据
├── runs/ # 训练结果（检测结果）
├── utils/ # 工具函数
├── convert_current_structure.py # XML → YOLO 格式转换
├── train.py # YOLOv8 训练脚本
├── train_v26.py # YOLO26 训练脚本
├── detect.py # 命令行检测
├── detect_gui.py # GUI 拖放检测
└── requirements.txt # 依赖包

## 使用说明

## 训练
# YOLOv8 训练
python train.py

# YOLO26 训练
python train_v26.py

## 检测
# 命令行检测
python detect.py --source 视频/图片路径

# GUI 拖放检测
python detect_gui.py

## 数据集
- 支持 VOC 格式（XML）自动转换为 YOLO 格式（TXT）
- 运行 convert_current_structure.py 完成转换

## 模型
yolov8n.pt：YOLOv8 预训练模型
yolo26n.pt：YOLO26 预训练模型

## 待改进
优化小目标检测精度
增加实时摄像头检测

## 联系方式
GitHub：syc591718-art
邮箱：1165254194@qq.com

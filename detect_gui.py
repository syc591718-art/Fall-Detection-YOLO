# detect_gui.py - YOLOv8拖放检测GUI
# 基于你的detect.py风格编写，兼容现有项目结构

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel,
                             QVBoxLayout, QWidget, QTextEdit,
                             QHBoxLayout, QPushButton, QSlider,
                             QFileDialog, QMessageBox, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QImage, QFont
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime


class DetectionWorker(QThread):
    """后台检测工作线程"""
    progress = pyqtSignal(str)  # 进度信号
    result_ready = pyqtSignal(np.ndarray, str, dict)  # 结果信号
    error = pyqtSignal(str)  # 错误信号

    def __init__(self, model_path, image_path, conf_threshold=0.5):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.conf_threshold = conf_threshold
        self.model = None

    def run(self):
        try:
            self.progress.emit("正在加载模型...")

            # 加载模型（兼容你的detect.py方式）
            self.model = YOLO(self.model_path)

            self.progress.emit(f"模型加载成功，类别: {list(self.model.names.values())}")
            self.progress.emit(f"开始检测: {os.path.basename(self.image_path)}...")

            # 执行检测
            results = self.model.predict(
                source=self.image_path,
                conf=self.conf_threshold,
                save=False,
                show=False,
                verbose=False
            )

            # 处理结果
            result_img = results[0].plot()  # 带检测框的图像

            # 统计信息（保持与detect.py一致）
            detection_summary = {}
            total_objects = 0
            info_text = ""

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for i, box in enumerate(results[0].boxes):
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    conf = float(box.conf[0])

                    detection_summary[cls_name] = detection_summary.get(cls_name, 0) + 1
                    total_objects += 1

                    info_text += f"目标 {i + 1}: {cls_name} | 置信度: {conf:.3f}\n"

            # 添加统计摘要
            if detection_summary:
                info_text += f"\n📊 检测统计:\n"
                info_text += f"   总检测对象: {total_objects}\n"
                for cls_name, count in sorted(detection_summary.items()):
                    percentage = count / total_objects * 100 if total_objects > 0 else 0
                    info_text += f"   {cls_name}: {count} 次 ({percentage:.1f}%)\n"
            else:
                info_text += "⚠ 未检测到任何对象"

            # 发送结果
            self.result_ready.emit(result_img, info_text, detection_summary)

        except Exception as e:
            self.error.emit(f"检测失败: {str(e)}")


class DragDropDetector(QMainWindow):
    """主窗口类"""

    def __init__(self):
        super().__init__()
        # 项目路径（与detect.py保持一致）
        self.project_root = Path(__file__).parent.absolute()

        # 默认模型路径
        self.default_model = self.project_root / 'runs/detect/fall_detection_gpu/weights/best.pt'
        self.current_model = str(self.default_model)
        self.conf_threshold = 0.5

        self.init_ui()
        self.load_available_models()

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle('YOLOv8姿态检测 - 拖放测试工具')
        self.setGeometry(200, 100, 1200, 700)

        # 设置字体
        font = QFont("Microsoft YaHei", 10)
        self.setFont(font)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout()

        # 左侧控制面板
        control_panel = QWidget()
        control_panel.setFixedWidth(350)
        control_layout = QVBoxLayout()

        # 标题
        title = QLabel("YOLOv8 拖放检测器")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)

        # 模型选择
        model_group = QWidget()
        model_layout = QVBoxLayout()
        model_label = QLabel("选择检测模型:")
        model_label.setStyleSheet("font-weight: bold;")

        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet("padding: 5px;")

        self.model_path_label = QLabel(f"路径: {os.path.basename(self.current_model)}")
        self.model_path_label.setWordWrap(True)
        self.model_path_label.setStyleSheet("color: #666; font-size: 9px;")

        model_btn_layout = QHBoxLayout()
        self.load_model_btn = QPushButton("重新选择模型")
        self.load_model_btn.clicked.connect(self.select_model_file)
        self.load_model_btn.setStyleSheet("padding: 5px;")

        self.refresh_btn = QPushButton("刷新列表")
        self.refresh_btn.clicked.connect(self.load_available_models)
        self.refresh_btn.setStyleSheet("padding: 5px;")

        model_btn_layout.addWidget(self.load_model_btn)
        model_btn_layout.addWidget(self.refresh_btn)

        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.model_path_label)
        model_layout.addLayout(model_btn_layout)
        model_group.setLayout(model_layout)

        # 置信度阈值
        conf_group = QWidget()
        conf_layout = QVBoxLayout()
        conf_label = QLabel(f"置信度阈值: {self.conf_threshold:.2f}")
        conf_label.setStyleSheet("font-weight: bold;")

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 95)  # 0.1 to 0.95
        self.conf_slider.setValue(int(self.conf_threshold * 100))
        self.conf_slider.valueChanged.connect(
            lambda v: conf_label.setText(f"置信度阈值: {v / 100:.2f}")
        )
        self.conf_slider.valueChanged.connect(
            lambda v: setattr(self, 'conf_threshold', v / 100)
        )

        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_group.setLayout(conf_layout)

        # 状态显示
        self.status_label = QLabel("状态: 准备就绪")
        self.status_label.setStyleSheet("padding: 10px; background-color: #f8f9fa; border-radius: 5px;")

        # 添加到控制面板
        control_layout.addWidget(title)
        control_layout.addWidget(model_group)
        control_layout.addWidget(conf_group)
        control_layout.addSpacing(20)

        # 使用说明
        instructions = QLabel(
            "使用说明:\n"
            "1. 拖放图片到右侧区域\n"
            "2. 自动检测并显示结果\n"
            "3. 可调整置信度阈值\n"
            "4. 支持格式: JPG, PNG, BMP"
        )
        instructions.setStyleSheet("padding: 10px; background-color: #e9f7fe; border-radius: 5px;")
        instructions.setWordWrap(True)

        control_layout.addWidget(instructions)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()

        # 保存按钮
        save_layout = QHBoxLayout()
        self.save_btn = QPushButton("💾 保存结果")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setStyleSheet("padding: 8px; font-weight: bold;")

        self.clear_btn = QPushButton("🗑️ 清除")
        self.clear_btn.clicked.connect(self.clear_results)
        self.clear_btn.setStyleSheet("padding: 8px;")

        save_layout.addWidget(self.save_btn)
        save_layout.addWidget(self.clear_btn)
        control_layout.addLayout(save_layout)

        control_panel.setLayout(control_layout)

        # 右侧显示区域
        display_panel = QWidget()
        display_layout = QVBoxLayout()

        # 拖放区域
        self.drop_label = QLabel("拖放图片到此区域进行检测")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 3px dashed #aaa;
                border-radius: 10px;
                padding: 40px;
                font-size: 14px;
                color: #666;
                background-color: #f8f9fa;
                min-height: 250px;
            }
            QLabel:hover {
                border-color: #007bff;
                background-color: #e9f7fe;
            }
        """)

        # 结果显示区域
        result_group = QWidget()
        result_layout = QVBoxLayout()
        result_title = QLabel("检测结果")
        result_title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")

        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumHeight(250)
        self.result_label.setStyleSheet("border: 1px solid #ddd; background-color: white;")

        result_layout.addWidget(result_title)
        result_layout.addWidget(self.result_label)
        result_group.setLayout(result_layout)

        # 检测信息
        info_group = QWidget()
        info_layout = QVBoxLayout()
        info_title = QLabel("检测信息")
        info_title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")

        self.info_text = QTextEdit()
        self.info_text.setPlaceholderText("检测结果将显示在这里...")
        self.info_text.setMaximumHeight(150)
        self.info_text.setReadOnly(True)
        self.info_text.setStyleSheet("font-family: 'Consolas', monospace; font-size: 10px;")

        info_layout.addWidget(info_title)
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)

        # 添加到显示面板
        display_layout.addWidget(self.drop_label)
        display_layout.addWidget(result_group)
        display_layout.addWidget(info_group)

        display_panel.setLayout(display_layout)

        # 添加到主布局
        main_layout.addWidget(control_panel)
        main_layout.addWidget(display_panel)

        central_widget.setLayout(main_layout)

        # 启用拖放
        self.setAcceptDrops(True)

        # 当前检测结果
        self.current_result = None
        self.current_image_path = None

    def load_available_models(self):
        """加载可用的模型文件（同时搜索 v8 和 v26 目录）"""
        self.model_combo.clear()

        # 同时搜索两个目录
        search_dirs = [
            self.project_root / 'runs' / 'detect',
            self.project_root / 'runs' / 'detect_v26'
        ]

        models_found = []

        for detect_dir in search_dirs:
            if detect_dir.exists():
                for model_dir in detect_dir.iterdir():
                    if model_dir.is_dir():
                        weights_dir = model_dir / 'weights'
                        if weights_dir.exists():
                            for weight_file in weights_dir.glob('*.pt'):
                                # 添加版本标识
                                version = "YOLOv26" if "v26" in str(detect_dir) else "YOLOv8"
                                display_name = f"[{version}] {model_dir.name}/{weight_file.name}"
                                models_found.append((display_name, str(weight_file)))

        # 按文件名排序
        models_found.sort(key=lambda x: x[0])

        # 添加到下拉框
        for display_name, file_path in models_found:
            self.model_combo.addItem(display_name, file_path)

        # 设置当前选择的模型
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == self.current_model:
                self.model_combo.setCurrentIndex(i)
                break

        # 如果没有找到，添加默认路径
        if self.model_combo.count() == 0:
            rel_path = os.path.relpath(self.current_model, self.project_root)
            self.model_combo.addItem(rel_path, self.current_model)
            self.model_combo.setCurrentIndex(0)

        # 更新状态
        self.status_label.setText(f"找到 {self.model_combo.count()} 个模型")

    def select_model_file(self):
        """手动选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择YOLOv8/YOLOv26模型文件",
            str(self.project_root),
            "PyTorch模型 (*.pt);;所有文件 (*.*)"
        )

        if file_path:
            self.current_model = file_path
            self.model_path_label.setText(f"路径: {os.path.basename(file_path)}")

            # 添加到下拉列表，自动判断版本
            if "v26" in file_path.lower():
                version = "YOLOv26"
            else:
                version = "YOLOv8"

            display_name = f"[{version}] {os.path.basename(os.path.dirname(os.path.dirname(file_path)))}/{os.path.basename(file_path)}"
            self.model_combo.addItem(display_name, file_path)
            self.model_combo.setCurrentIndex(self.model_combo.count() - 1)

            self.status_label.setText(f"已选择模型: {os.path.basename(file_path)}")

    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                event.accept()
                self.drop_label.setText("松开鼠标进行检测...")
                self.drop_label.setStyleSheet("""
                    QLabel {
                        border: 3px dashed #28a745;
                        border-radius: 10px;
                        padding: 40px;
                        font-size: 14px;
                        color: #28a745;
                        background-color: #e9f7fe;
                        min-height: 250px;
                    }
                """)
            else:
                event.ignore()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """拖拽离开事件"""
        self.drop_label.setText("拖放图片到此区域进行检测")
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 3px dashed #aaa;
                border-radius: 10px;
                padding: 40px;
                font-size: 14px;
                color: #666;
                background-color: #f8f9fa;
                min-height: 250px;
            }
            QLabel:hover {
                border-color: #007bff;
                background-color: #e9f7fe;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        """松开鼠标事件"""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.current_image_path = file_path

            # 显示原始图片
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(self.drop_label.size(),
                                              Qt.KeepAspectRatio,
                                              Qt.SmoothTransformation)
                self.drop_label.setPixmap(scaled_pixmap)

            # 获取选择的模型
            model_index = self.model_combo.currentIndex()
            if model_index >= 0:
                self.current_model = self.model_combo.itemData(model_index)

            # 更新状态
            self.status_label.setText(f"正在检测: {os.path.basename(file_path)}...")
            self.info_text.clear()

            # 禁用保存按钮直到新结果
            self.save_btn.setEnabled(False)

            # 启动检测线程
            self.detection_worker = DetectionWorker(
                self.current_model,
                file_path,
                self.conf_threshold
            )
            self.detection_worker.progress.connect(self.update_status)
            self.detection_worker.result_ready.connect(self.show_detection_results)
            self.detection_worker.error.connect(self.show_error)
            self.detection_worker.start()

            event.accept()
        else:
            event.ignore()

    def update_status(self, message):
        """更新状态信息"""
        self.status_label.setText(f"状态: {message}")

    def show_detection_results(self, result_img, info_text, detection_summary):
        """显示检测结果"""
        # 将numpy数组转换为QPixmap
        height, width, channel = result_img.shape
        bytes_per_line = 3 * width

        # 确保是RGB格式
        if result_img.shape[2] == 3:
            q_img = QImage(result_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:
            # 如果是BGR，转换为RGB
            rgb_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)

        # 调整大小
        scaled_pixmap = pixmap.scaled(self.result_label.size(),
                                      Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
        self.result_label.setPixmap(scaled_pixmap)

        # 保存结果
        self.current_result = result_img

        # 显示检测信息
        self.info_text.setText(info_text)

        # 更新状态
        self.status_label.setText("检测完成 ✓")
        self.save_btn.setEnabled(True)

        # 恢复拖放区域样式
        self.drop_label.setText("拖放图片到此区域进行检测")
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 3px dashed #aaa;
                border-radius: 10px;
                padding: 40px;
                font-size: 14px;
                color: #666;
                background-color: #f8f9fa;
                min-height: 250px;
            }
            QLabel:hover {
                border-color: #007bff;
                background-color: #e9f7fe;
            }
        """)

    def show_error(self, error_msg):
        """显示错误信息"""
        self.status_label.setText(f"错误: {error_msg}")
        QMessageBox.critical(self, "检测错误", error_msg)

        # 恢复界面
        self.drop_label.setText("拖放图片到此区域进行检测")
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 3px dashed #aaa;
                border-radius: 10px;
                padding: 40px;
                font-size: 14px;
                color: #666;
                background-color: #f8f9fa;
                min-height: 250px;
            }
            QLabel:hover {
                border-color: #007bff;
                background-color: #e9f7fe;
            }
        """)

    def save_result(self):
        """保存检测结果"""
        if self.current_result is None or self.current_image_path is None:
            return

        # 生成保存路径（与detect.py保持一致）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.project_root / 'runs/detect' / f'gui_predict_{timestamp}'
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存图片
        original_name = os.path.basename(self.current_image_path)
        save_name = f"detected_{original_name}"
        save_path = save_dir / save_name

        # 保存图像
        cv2.imwrite(str(save_path), cv2.cvtColor(self.current_result, cv2.COLOR_RGB2BGR))

        # 保存检测信息
        info_path = save_dir / "detection_info.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"原始图片: {self.current_image_path}\n")
            f.write(f"模型文件: {self.current_model}\n")
            f.write(f"置信度阈值: {self.conf_threshold}\n")
            f.write("\n检测结果:\n")
            f.write(self.info_text.toPlainText())

        QMessageBox.information(self, "保存成功",
                                f"结果已保存到:\n{save_dir}")

    def clear_results(self):
        """清除当前结果"""
        self.result_label.clear()
        self.info_text.clear()
        self.drop_label.clear()
        self.drop_label.setText("拖放图片到此区域进行检测")
        self.current_result = None
        self.current_image_path = None
        self.save_btn.setEnabled(False)
        self.status_label.setText("状态: 准备就绪")


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle('Fusion')

    # 创建并显示窗口
    window = DragDropDetector()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
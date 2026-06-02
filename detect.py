# detect.py - 交互式YOLOv8/v26检测脚本（修复中文显示问题）
from ultralytics import YOLO
import os
import shutil
from datetime import datetime
from pathlib import Path
import sys
import cv2
import time
from collections import defaultdict

# 全局变量用于控制摄像头线程
camera_running = False


def select_mode():
    print("\n" + "=" * 60)
    print("🎯 YOLOv8/v26 交互式检测系统")
    print("=" * 60)
    print("请选择检测模式:")
    print("  1. 📷 测试单张图片")
    print("  2. 📁 测试整个文件夹")
    print("  3. 🎥 测试视频文件")
    print("  4. 📹 实时摄像头检测")
    print("  5. 🚪 退出程序")
    print("=" * 60)
    while True:
        choice = input("请输入选择 (1-5): ").strip()
        if choice == '1':
            return 'single_image'
        elif choice == '2':
            return 'folder'
        elif choice == '3':
            return 'video'
        elif choice == '4':
            return 'camera'
        elif choice == '5':
            print("👋 退出程序")
            sys.exit(0)
        else:
            print("❌ 无效选择，请重新输入")


def get_input_source(mode, project_root):
    common_dirs = ['data/raw', 'data/test_images', 'data/dataset/images/val']

    print("\n" + "-" * 40)
    print("🔄 操作提示: 输入 'q' 可返回主菜单, 输入 'exit' 直接退出程序")
    print("-" * 40)

    if mode == 'single_image':
        print("\n📷 请选择图片:")
        print("  1. 浏览文件")
        print("  2. 使用常见目录")
        print("  q. 返回主菜单")
        print("  exit. 退出程序")

        choice = input("请选择 (1/2/q/exit): ").strip().lower()

        if choice == 'q':
            return 'back_to_menu'
        elif choice == 'exit':
            print("👋 退出程序")
            sys.exit(0)
        elif choice == '1':
            while True:
                path = input("请输入图片完整路径 (或拖拽图片到这里，输入q返回，exit退出): ").strip().strip('"')
                if path.lower() == 'q':
                    return 'back_to_menu'
                elif path.lower() == 'exit':
                    print("👋 退出程序")
                    sys.exit(0)
                elif os.path.exists(path) and path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    return path
                else:
                    print("❌ 文件不存在或不是图片文件，请重新输入")
        elif choice == '2':
            print("\n常见目录:")
            valid_dirs = []
            for i, dir_path in enumerate(common_dirs, 1):
                full_path = project_root / dir_path
                if full_path.exists():
                    valid_dirs.append(dir_path)
                    print(f"  {i}. {dir_path}")

            print("  q. 返回主菜单")
            print("  exit. 退出程序")

            if not valid_dirs:
                print("❌ 没有找到常见目录")
                return 'back_to_menu'

            dir_choice = input(f"选择目录 (1-{len(valid_dirs)}/q/exit): ").strip().lower()

            if dir_choice == 'q':
                return 'back_to_menu'
            elif dir_choice == 'exit':
                print("👋 退出程序")
                sys.exit(0)
            elif dir_choice.isdigit() and 1 <= int(dir_choice) <= len(valid_dirs):
                dir_path = valid_dirs[int(dir_choice) - 1]
                img_dir = project_root / dir_path
                images = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
                    images.extend(list(img_dir.glob(ext)))

                if images:
                    print(f"\n找到 {len(images)} 张图片:")
                    for i, img in enumerate(images[:10], 1):
                        print(f"  {i}. {img.name}")
                    if len(images) > 10: print(f"  ... 等 {len(images)} 张图片")

                    print("  q. 返回主菜单")
                    print("  exit. 退出程序")

                    img_choice = input(f"选择图片 (1-{min(10, len(images))} 或输入文件名/q/exit): ").strip().lower()

                    if img_choice == 'q':
                        return 'back_to_menu'
                    elif img_choice == 'exit':
                        print("👋 退出程序")
                        sys.exit(0)
                    elif img_choice.isdigit() and 1 <= int(img_choice) <= len(images):
                        return str(images[int(img_choice) - 1])
                    else:
                        test_path = img_dir / img_choice
                        if test_path.exists():
                            return str(test_path)
                        else:
                            print("❌ 文件不存在")
                            return 'back_to_menu'
                else:
                    print(f"❌ {dir_path} 中没有图片文件")
                    return 'back_to_menu'
            return 'back_to_menu'
        else:
            print("❌ 无效选择")
            return 'back_to_menu'

    elif mode == 'folder':
        print("\n📁 请选择文件夹:")
        print("  1. 浏览文件夹")
        print("  2. 使用常见目录")
        print("  q. 返回主菜单")
        print("  exit. 退出程序")

        choice = input("请选择 (1/2/q/exit): ").strip().lower()

        if choice == 'q':
            return 'back_to_menu'
        elif choice == 'exit':
            print("👋 退出程序")
            sys.exit(0)
        elif choice == '1':
            while True:
                path = input("请输入文件夹完整路径 (输入q返回，exit退出): ").strip().strip('"')
                if path.lower() == 'q':
                    return 'back_to_menu'
                elif path.lower() == 'exit':
                    print("👋 退出程序")
                    sys.exit(0)
                elif os.path.exists(path) and os.path.isdir(path):
                    return path
                else:
                    print("❌ 文件夹不存在，请重新输入")
        elif choice == '2':
            print("\n常见目录:")
            valid_dirs = []
            for i, dir_path in enumerate(common_dirs, 1):
                full_path = project_root / dir_path
                if full_path.exists():
                    valid_dirs.append(dir_path)
                    print(f"  {i}. {dir_path}")

            print("  q. 返回主菜单")
            print("  exit. 退出程序")

            if not valid_dirs:
                print("❌ 没有找到常见目录")
                return 'back_to_menu'

            dir_choice = input(f"选择目录 (1-{len(valid_dirs)}/q/exit): ").strip().lower()

            if dir_choice == 'q':
                return 'back_to_menu'
            elif dir_choice == 'exit':
                print("👋 退出程序")
                sys.exit(0)
            elif dir_choice.isdigit() and 1 <= int(dir_choice) <= len(valid_dirs):
                return str(project_root / valid_dirs[int(dir_choice) - 1])
            else:
                print("❌ 无效选择")
                return 'back_to_menu'
        else:
            print("❌ 无效选择")
            return 'back_to_menu'

    elif mode == 'video':
        print("\n🎥 请选择视频文件:")
        print("  1. 浏览文件")
        print("  2. 使用示例视频")
        print("  q. 返回主菜单")
        print("  exit. 退出程序")

        choice = input("请选择 (1/2/q/exit): ").strip().lower()

        if choice == 'q':
            return 'back_to_menu'
        elif choice == 'exit':
            print("👋 退出程序")
            sys.exit(0)
        elif choice == '1':
            while True:
                path = input("请输入视频完整路径 (输入q返回，exit退出): ").strip().strip('"')
                if path.lower() == 'q':
                    return 'back_to_menu'
                elif path.lower() == 'exit':
                    print("👋 退出程序")
                    sys.exit(0)
                elif os.path.exists(path) and path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    return path
                else:
                    print("❌ 文件不存在或不是视频文件，请重新输入")
        elif choice == '2':
            video_dir = project_root / 'data/videos'
            if video_dir.exists():
                videos = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
                if videos:
                    print(f"\n找到 {len(videos)} 个视频:")
                    for i, video in enumerate(videos, 1):
                        print(f"  {i}. {video.name}")

                    print("  q. 返回主菜单")
                    print("  exit. 退出程序")

                    vid_choice = input(f"选择视频 (1-{len(videos)}/q/exit): ").strip().lower()

                    if vid_choice == 'q':
                        return 'back_to_menu'
                    elif vid_choice == 'exit':
                        print("👋 退出程序")
                        sys.exit(0)
                    elif vid_choice.isdigit() and 1 <= int(vid_choice) <= len(videos):
                        return str(videos[int(vid_choice) - 1])
                    else:
                        print("❌ 无效选择")
                        return 'back_to_menu'
                else:
                    print("❌ 没有找到示例视频")
                    return 'back_to_menu'
            else:
                print("❌ 视频目录不存在")
                return 'back_to_menu'
        else:
            print("❌ 无效选择")
            return 'back_to_menu'

    elif mode == 'camera':
        print("\n📹 摄像头检测模式")
        print("  1. 默认摄像头 (0)")
        print("  2. 其他摄像头ID")
        print("  q. 返回主菜单")
        print("  exit. 退出程序")

        choice = input("请选择 (1/2/q/exit): ").strip().lower()

        if choice == 'q':
            return 'back_to_menu'
        elif choice == 'exit':
            print("👋 退出程序")
            sys.exit(0)
        elif choice == '1':
            return '0'
        elif choice == '2':
            cam_id = input("请输入摄像头ID (通常是0, 1, 2...，输入q返回，exit退出): ").strip().lower()
            if cam_id == 'q':
                return 'back_to_menu'
            elif cam_id == 'exit':
                print("👋 退出程序")
                sys.exit(0)
            else:
                return cam_id
        else:
            print("❌ 无效选择")
            return 'back_to_menu'

    return 'back_to_menu'


def select_model(project_root):
    print("\n" + "-" * 40)
    print("🔄 操作提示: 输入 'q' 可返回主菜单, 输入 'exit' 直接退出程序")
    print("-" * 40)

    # 搜索多个目录
    search_dirs = [
        project_root / 'runs' / 'detect',
        project_root / 'runs' / 'detect_v26'
    ]

    models = []

    for detect_dir in search_dirs:
        if detect_dir.exists():
            for model_dir in detect_dir.iterdir():
                if model_dir.is_dir():
                    weights_path = model_dir / 'weights' / 'best.pt'
                    if weights_path.exists():
                        version = "v26" if "v26" in str(detect_dir) else "v8"
                        models.append({
                            'name': f"{version}/{model_dir.name}",
                            'path': str(weights_path),
                            'time': weights_path.stat().st_mtime
                        })

    if models:
        print("\n🤖 可用的模型文件:")
        models.sort(key=lambda x: x['time'], reverse=True)

        for i, model in enumerate(models[:15], 1):
            time_str = datetime.fromtimestamp(model['time']).strftime('%m-%d %H:%M')
            print(f"  {i}. {model['name']} ({time_str})")

        print(f"  {len(models) + 1}. 使用预训练模型 (yolov8n.pt)")
        print("  q. 返回主菜单")
        print("  exit. 退出程序")

        choice = input(f"选择模型 (1-{len(models) + 1}/q/exit, 默认1): ").strip().lower()

        if choice == 'q':
            return 'back_to_menu'
        elif choice == 'exit':
            print("👋 退出程序")
            sys.exit(0)
        elif choice == '' or choice == '1':
            return models[0]['path']
        elif choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]['path']
        elif choice.isdigit() and int(choice) == len(models) + 1:
            return 'yolov8n.pt'
        else:
            print(f"⚠  使用默认模型: {models[0]['path']}")
            return models[0]['path']
    else:
        print("⚠  未找到训练模型，使用预训练模型")
        print("  q. 返回主菜单")
        print("  exit. 退出程序")

        choice = input("请选择 (q/exit, 默认使用预训练模型): ").strip().lower()
        if choice == 'q':
            return 'back_to_menu'
        elif choice == 'exit':
            print("👋 退出程序")
            sys.exit(0)
        else:
            return 'yolov8n.pt'


def select_confidence():
    print("\n" + "-" * 40)
    print("🔄 操作提示: 输入 'q' 可返回主菜单, 输入 'exit' 直接退出程序")
    print("-" * 40)

    print("\n🎚️  置信度阈值设置:")
    print("  1. 高敏感度 (0.3) - 更多检测，可能包含误检")
    print("  2. 标准 (0.5) - 平衡模式")
    print("  3. 高精度 (0.7) - 更少检测，更准确")
    print("  4. 自定义")
    print("  q. 返回主菜单")
    print("  exit. 退出程序")

    choice = input("请选择 (1-4/q/exit, 默认2): ").strip().lower()

    if choice == 'q':
        return 'back_to_menu'
    elif choice == 'exit':
        print("👋 退出程序")
        sys.exit(0)
    elif choice == '1':
        return 0.3
    elif choice == '' or choice == '2':
        return 0.5
    elif choice == '3':
        return 0.7
    elif choice == '4':
        while True:
            try:
                custom_conf = float(input("请输入置信度 (0.1-0.9，输入q返回，exit退出): "))
                if 0.1 <= custom_conf <= 0.9:
                    return custom_conf
                else:
                    print("❌ 请输入0.1-0.9之间的值")
            except ValueError:
                val = input("请输入有效的数字 (或输入q返回，exit退出): ").strip().lower()
                if val == 'q':
                    return 'back_to_menu'
                elif val == 'exit':
                    print("👋 退出程序")
                    sys.exit(0)
    else:
        return 0.5


def camera_detection(source, model, conf, save_dir):
    """摄像头检测函数 - 只使用英文显示，避免中文乱码"""
    global camera_running

    print(f"📹 正在启动摄像头 {source}...")
    cap = cv2.VideoCapture(int(source))

    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {source}")
        return 0, {}

    # 摄像头参数设置
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"✅ 摄像头已连接 ({width}x{height} @ 20FPS)")
    print("\n📋 操作指南:")
    print("  按 'q' 键 - 退出检测")
    print("  按 's' 键 - 保存当前帧")
    print("  按 'd' 键 - 切换检测模式")
    print("  按 '+' 键 - 提高检测灵敏度")
    print("  按 '-' 键 - 降低检测灵敏度")
    print("  按 'p' 键 - 暂停/继续检测")
    print("\n检测将在3秒后开始...")
    time.sleep(3)

    detection_summary = defaultdict(int)
    frame_count = 0
    last_print_time = time.time()
    print_interval = 2.0

    # 可调参数
    current_conf = conf
    detection_mode = "normal"
    paused = False

    # 用于存储上一帧的检测结果，实现平滑显示
    last_detection_results = None
    last_detection_frame = 0
    display_persistence = 3

    # 检测缓冲和稳定化
    detection_buffer = []
    buffer_size = 5
    stable_detection_count = 0

    print(f"\n⚙️  当前设置:")
    print(f"  置信度: {current_conf:.2f}")
    print(f"  检测模式: {detection_mode}")
    print(f"  检测间隔: 每3帧检测一次")

    camera_running = True

    while camera_running:
        if paused:
            key = cv2.waitKey(300) & 0xFF
            if key == ord('p'):
                paused = False
                print("▶️  继续检测")
            elif key == ord('q'):
                break
            continue

        ret, frame = cap.read()
        if not ret:
            print("❌ 无法读取摄像头画面")
            break

        frame_count += 1
        display_frame = frame.copy()

        # 每3帧检测一次
        should_detect = (frame_count % 3 == 0)

        detection_info = "No target"

        if should_detect:
            # 根据模式调整参数
            if detection_mode == "fast":
                results = model(frame, conf=current_conf, iou=0.4, verbose=False, max_det=30)
            elif detection_mode == "sensitive":
                results = model(frame, conf=0.2, iou=0.3, verbose=False, max_det=50)
            else:  # normal
                results = model(frame, conf=current_conf, iou=0.45, verbose=False, max_det=30)

            # 保存检测结果用于后续帧显示
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                last_detection_results = results
                last_detection_frame = frame_count

                # 添加到检测缓冲
                detection_buffer.append(results)
                if len(detection_buffer) > buffer_size:
                    detection_buffer.pop(0)
            else:
                last_detection_results = None

                # 无检测时也添加空结果到缓冲
                detection_buffer.append(None)
                if len(detection_buffer) > buffer_size:
                    detection_buffer.pop(0)
        else:
            # 使用上一帧的检测结果保持显示
            if last_detection_results is not None and frame_count - last_detection_frame <= display_persistence:
                results = last_detection_results
            else:
                results = None

        # 使用缓冲进行稳定化处理
        if len(detection_buffer) >= 3:
            valid_detections = [r for r in detection_buffer[-3:] if r is not None and r[0].boxes is not None]
            if len(valid_detections) >= 2:  # 最近3帧中有2帧检测到
                stable_detection_count = len(valid_detections[-1][0].boxes) if valid_detections[-1] else 0
            else:
                stable_detection_count = 0

        # 处理检测结果并绘制
        if results is not None and results[0].boxes is not None and len(results[0].boxes) > 0:
            annotated_frame = results[0].plot(line_width=2, font_size=1)
            display_frame = annotated_frame

            objects_detected = defaultdict(int)
            persons_detected = 0

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf_score = float(box.conf[0])

                objects_detected[cls_name] += 1
                detection_summary[cls_name] += 1

                if cls_name in ['up', 'bending', 'down']:
                    persons_detected += 1

            # 生成检测信息（英文）
            detection_info = f"People: {persons_detected}"
            if objects_detected:
                det_str = ", ".join([f"{k}({v})" for k, v in objects_detected.items()])
                detection_info = f"{detection_info} - {det_str}"

            # 控制打印频率
            current_time = time.time()
            if current_time - last_print_time >= print_interval:
                print(f"  帧 {frame_count}: {detection_info}")
                last_print_time = current_time

        # 在画面上显示状态信息（全部用英文，避免中文乱码）
        mode_text = f"Mode: {detection_mode}"
        conf_text = f"Conf: {current_conf:.2f}"
        stable_text = f"Stable: {stable_detection_count}"
        frame_text = f"Frame: {frame_count}"

        cv2.putText(display_frame, mode_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, conf_text, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, stable_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display_frame, detection_info, (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display_frame, frame_text, (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 显示帮助信息（英文，避免乱码）
        help_text = "q:quit s:save d:mode +/-:sens p:pause"
        cv2.putText(display_frame, help_text, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('YOLO Detection', display_frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 正在退出摄像头...")
            camera_running = False
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = save_dir / f'frame_{timestamp}.jpg'
            cv2.imwrite(str(save_path), display_frame)
            print(f"📸 已保存当前帧: {save_path.name}")
        elif key == ord('d'):
            if detection_mode == "normal":
                detection_mode = "fast"
                print("🔍 切换到快速模式")
            elif detection_mode == "fast":
                detection_mode = "sensitive"
                print("🔍 切换到敏感模式")
            else:
                detection_mode = "normal"
                print("🔍 切换到普通模式")
        elif key == ord('+'):
            current_conf = max(0.1, current_conf - 0.05)
            print(f"📈 提高灵敏度: 置信度 -> {current_conf:.2f}")
        elif key == ord('-'):
            current_conf = min(0.9, current_conf + 0.05)
            print(f"📉 降低灵敏度: 置信度 -> {current_conf:.2f}")
        elif key == ord('p'):
            paused = True
            print("⏸️  暂停检测")

    # 清理
    print("🛑 正在关闭摄像头...")
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)

    print(f"\n📊 摄像头检测统计:")
    print(f"   总处理帧数: {frame_count}")
    print(f"   实际检测帧数: {frame_count // 3}")

    camera_running = False
    return frame_count, dict(detection_summary)


def main():
    project_root = Path(__file__).parent.absolute()

    while True:
        mode = select_mode()
        source = get_input_source(mode, project_root)

        if source == 'back_to_menu':
            print("\n🔄 返回主菜单...")
            continue

        weights = select_model(project_root)

        if weights == 'back_to_menu':
            print("\n🔄 返回主菜单...")
            continue

        conf = select_confidence()

        if conf == 'back_to_menu':
            print("\n🔄 返回主菜单...")
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建自定义结果文件夹
        results_dir = project_root / 'runs' / 'my_results'
        results_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.now().strftime("%Y%m%d")
        daily_dir = results_dir / today
        daily_dir.mkdir(parents=True, exist_ok=True)

        save_dir = daily_dir / f'predict_{timestamp}'
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("⚙️  检测配置:")
        print("=" * 60)
        print(f"📁 项目目录: {project_root}")
        print(f"🤖 模型文件: {weights}")
        print(f"📷 检测源: {source}")
        print(f"🎚️  初始置信度: {conf}")
        print(f"💾 保存位置: {save_dir}")
        print("=" * 60)

        if mode == 'camera':
            print("💡 多人检测优化提示:")
            print("  1. 现在支持同时识别多人")
            print("  2. 检测框会持续显示，不会一闪而过")
            print("  3. 按 'q' 键可正常退出")
            print("  4. 按 'p' 键暂停/继续检测")

        if source != '0' and not source.isdigit() and not os.path.exists(source):
            print(f"❌ 错误: 检测源不存在")
            print(f"   路径: {source}")
            continue

        if not os.path.exists(weights):
            print(f"❌ 错误: 模型文件不存在")
            print(f"   路径: {weights}")
            continue

        try:
            print("🔄 加载模型...")
            model = YOLO(weights)
            print(f"✅ 模型加载成功")
            print(f"   类别: {list(model.names.values())}")

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            continue

        print(f"\n🚀 开始检测...")
        print(f"📂 结果将保存到: {save_dir}")
        print("-" * 60)

        try:
            if mode == 'camera':
                frame_count, detection_summary = camera_detection(source, model, conf, save_dir)

                if detection_summary:
                    print(f"\n📈 检测统计:")
                    total_objects = sum(detection_summary.values())
                    print(f"   总检测次数: {total_objects}")
                    for cls_name, count in sorted(detection_summary.items()):
                        percentage = count / total_objects * 100 if total_objects > 0 else 0
                        print(f"   {cls_name}: {count} 次 ({percentage:.1f}%)")

            elif mode == 'video':
                print("🎥 视频检测模式")
                results = model.predict(
                    source=source,
                    conf=conf,
                    save=True,
                    show=False,
                    save_txt=False,
                    project=str(results_dir),
                    name=f'{today}/predict_{timestamp}',
                    exist_ok=True,
                    verbose=False
                )
                print(f"\n✅ 视频检测完成！")

            else:
                results = model.predict(
                    source=source,
                    conf=conf,
                    save=True,
                    show=False,
                    save_txt=False,
                    project=str(results_dir),
                    name=f'{today}/predict_{timestamp}',
                    exist_ok=True,
                    verbose=False
                )

                print(f"\n✅ 检测完成！")
                print(f"📊 处理数量: {len(results)} 张图片/帧")

                detection_summary = {}
                total_objects = 0
                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            cls_name = model.names[cls_id]
                            detection_summary[cls_name] = detection_summary.get(cls_name, 0) + 1
                            total_objects += 1

                if detection_summary:
                    print(f"\n📈 检测统计:")
                    print(f"   总检测对象: {total_objects}")
                    for cls_name, count in sorted(detection_summary.items()):
                        percentage = count / total_objects * 100 if total_objects > 0 else 0
                        print(f"   {cls_name}: {count} 次 ({percentage:.1f}%)")
                else:
                    print("⚠  未检测到任何对象")

            # 显示保存的文件
            print(f"\n💾 保存的文件:")
            image_files = list(save_dir.glob('*.jpg')) + list(save_dir.glob('*.png'))

            if image_files:
                print(f"   图片文件: {len(image_files)} 张")
                if len(image_files) <= 5:
                    for img in image_files: print(f"     • {img.name}")

            print(f"\n📁 完整结果路径: {save_dir.resolve()}")

            try:
                os.startfile(save_dir)
                print(f"📂 已自动打开结果目录")
            except:
                pass

            print(f"\n🔄 是否再次检测?")
            print("  1. 是 (返回主菜单)")
            print("  2. 否 (退出程序)")
            again = input("请选择 (1/2, 默认2): ").strip()
            if again == '1':
                continue
            else:
                print("👋 程序结束")
                break

        except Exception as e:
            print(f"❌ 检测过程中出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n" + "=" * 60)
            print(f"🕒 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)


if __name__ == '__main__':
    main()
import os
import cv2


def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.flv', '.mov', '.wmv']  # 添加其他视频文件扩展名
    return any(filename.endswith(ext) for ext in video_extensions)


def video_to_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"fps{frame_rate}_frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()


def frames_to_video(frames_folder, output_video_path):
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, layers = frame.shape
    frame_rate = int(frame_files[0][3:frame_files[0].index("_frame")])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()

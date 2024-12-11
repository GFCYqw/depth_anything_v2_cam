import cv2
import torch
import matplotlib
import numpy as np
import time

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# 模型选择
encoders = ['vits', 'vitb', 'vitl', 'vitg']
encoder = encoders[0]
# encoder = encoders[2]

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu', weights_only=True))
model = model.to(DEVICE).eval()

t_last = time.time()
pos = [320, 240]    # 标记点初始坐标
TextColor = (0, 255, 0)  # 文字颜色
TextFont = cv2.FONT_HERSHEY_TRIPLEX  # 文字字体

# 定义鼠标回调函数


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        # print('鼠标左键点击坐标：', x % 640, y)
        global pos
        pos = [x % 640, y]


# 创建窗口，并绑定鼠标回调函数
cv2.namedWindow('RESULT')
cv2.setMouseCallback('RESULT', mouse_callback)

cmap = matplotlib.colormaps.get_cmap('Spectral_r')  # colorful颜色映射
cap = cv2.VideoCapture(0)  # 获取摄像头视频流

while cap.isOpened():
    ret, raw_img = cap.read()
    if ret == True:
        raw_depth = model.infer_image(raw_img)  # 进行编码
        depth = raw_depth  # 保留原始深度数据
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0  # 重映射到 0-255
        depth = depth.astype(np.uint8)

        grayscale = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        colorful = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # 左上角显示各图像名称
        cv2.putText(raw_img, 'raw_img', (20, 30), TextFont, 1, TextColor, 2)
        cv2.putText(grayscale, 'grayscale', (20, 30), TextFont, 1, TextColor, 2)
        cv2.putText(colorful, 'colorful', (20, 30), TextFont, 1, TextColor, 2)

        # 标记鼠标点击位置
        cv2.drawMarker(raw_img, pos, TextColor, cv2.MARKER_CROSS, 40, 2)
        cv2.drawMarker(grayscale, pos, TextColor, cv2.MARKER_CROSS, 40, 2)
        cv2.drawMarker(colorful, pos, TextColor, cv2.MARKER_CROSS, 40, 2)

        # 显示标记点信息
        cv2.putText(grayscale, f'P O S: {str(pos)}',
                    (20, 400), TextFont, 0.6, TextColor, 2)
        cv2.putText(grayscale, f'DEPTH: {str(raw_depth[pos[1]][pos[0]])}',
                    (20, 420), TextFont, 0.6, TextColor, 2)
        cv2.putText(grayscale, f'M A X: {str(raw_depth.max())}',
                    (20, 440), TextFont, 0.6, TextColor, 2)
        cv2.putText(grayscale, f'M I N: {str(raw_depth.min())}',
                    (20, 460), TextFont, 0.6, TextColor, 2)

        # 显示图像
        res = np.hstack([raw_img, grayscale, colorful])  # 连接三个图像

        # 计算 FPS
        dt = time.time() - t_last
        fps = 1.0 / dt
        t_last = time.time()
        cv2.putText(res, f"FPS: {int(fps)}", (20, 460), TextFont, 1, TextColor, 2)

        cv2.imshow('RESULT', res)

        # 按键检测
        key = cv2.waitKey(1)
        if key & 0xFF == ord(' '):  # SPACE 暂停
            cv2.waitKey(0)
        elif key & 0xFF == 27:  # ESC 退出
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()


# raw_img = cv2.imread("E:\\Project\\Depth-Anything-V2-main\\assets\\examples\\demo06.jpg")
# depth = model.infer_image(raw_img) # HxW raw depth map in numpy
# cv2.imshow('imshow', depth)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

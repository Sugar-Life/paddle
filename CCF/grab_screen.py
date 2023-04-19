# import mss
# import numpy as np
# import cv2
#
# sct = mss.mss()
# monitor={
#     'left':0,
#     'top':0,
#     'width':1920,
#     'height':1080
# }
# window_name = 'test'
# while True:
#     img = sct.grab(monitor=monitor)
#     img = np.array(img)
#     cv2.imshow(window_name,img)
#     cv2.waitKey(0)
import numpy as np
import cv2
from PIL import ImageGrab

# 创建窗口并设置标志
cv2.namedWindow("Screen Capture", cv2.WINDOW_GUI_NORMAL)

# 定义全局变量
refPt = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    global refPt, cropping

    # 鼠标左键按下，开始截屏
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # 鼠标左键松开，结束截屏
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False

        # 绘制截屏区域
        cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("Screen Capture", frame)


while True:
    # 获取屏幕截图
    img = ImageGrab.grab()

    # 转换为OpenCV格式
    img_np = np.array(img)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # 显示截图并等待鼠标事件
    cv2.imshow("Screen Capture", frame)
    cv2.setMouseCallback("Screen Capture", click_and_crop)

    # 按下q键退出循环
    if cv2.waitKey(0) == ord("ESC"):
        break

    # 如果已经截屏，进行裁剪并显示
    if len(refPt) == 2:
        roi = img.crop((refPt[0][0], refPt[0][1], refPt[1][0], refPt[1][1]))
        roi.show()

# 释放资源
cv2.destroyAllWindows()


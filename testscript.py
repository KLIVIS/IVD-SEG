from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def resize_and_analyze_image(image_path):
    # 打开图像
    img = Image.open(image_path)

    # 将图像调整为 64x64
    img_resized = img.resize((64, 64),Image.NEAREST)

    # 将图像转换为 NumPy 数组
    img_array = np.array(img_resized)

    # 获取通道数
    channels = img_array.shape[2] if len(img_array.shape) == 3 else 1

    # 输出通道数
    print(f"通道数：{channels}")

    # 输出所有像素点的值
    print("像素点的值：")

    for row in img_array:
        for pixel_value in row:
            print(pixel_value, end=' ')
        print()  # 换行

def AUC(output, label):
    # 1. 整理数据
    b, c, h, w = output.shape
    reshaped_output = output.reshape((b * h * w, c))
    # 2. 计算真实标签和预测值
    reshaped_labels = label.reshape((b * h * w, 1))

    auc = roc_auc_score(reshaped_labels, reshaped_output)
    return auc

if __name__ == "__main__":
    torch.cuda.set_device(1)



from PIL import Image
import os
import random
import numpy as np
import math

import shutil

def resize_images(input_folder):
    output_folder = input_folder
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图像
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 打开图像
            img = Image.open(input_path)

            # 使用 NEAREST 模式进行 resize
            img_resized = img.resize((2048,256), Image.NEAREST)

            # 保存 resize 后的图像
            img_resized.save(output_path)

# 将多分类的ann转换成单分类，所有类别都统一为缺陷，然后reszie为256
def process_annotation(input_folder, class_number):
    output_folder = input_folder
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图像
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 打开图像
            img = Image.open(input_path)
            # 保证图片都是三通道
            img = img.convert('RGB')

            # 将图像转换为 NumPy 数组
            img_array = np.array(img)

            # 对所有非零像素点更改为 255
            img_array[img_array > 0] = class_number

            # 将 NumPy 数组转换回图像
            img_processed = Image.fromarray(img_array)

            # 使用 NEAREST 模式进行 resize
            img_resized = img_processed.resize((256,256), Image.NEAREST)

            # 保存处理后的图像
            img_resized.save(output_path)
            print(f"成功处理{filename}")


def split_images_to_files(source_folder,  train_ratio=0.7, val_ratio=0.1):
    
    source_folder_a = os.path.join(source_folder, 'all')
    source_folder_a = os.path.join(source_folder_a, 'annotationsAll')
    
    output_folder = os.path.join(source_folder, 'list')
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = [f for f in os.listdir(source_folder_a) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    random.shuffle(image_files)

    total_images = len(image_files)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)

    train_images = image_files[:train_count]
    val_images = image_files[train_count:train_count + val_count]
    test_images = image_files[train_count + val_count:]

    def write_to_file(image_list, file_name):
        with open(os.path.join(output_folder, file_name), 'w') as f:
            for image_name in image_list:
                if image_name.endswith('.png'):
                    # 如果以 .png 结尾，替换掉 .png 后缀
                    image_name = image_name.replace('.png', '') 
                f.write(image_name + '\n')
                print(f"完成{image_name}的写入")

    write_to_file(train_images, 'train.txt')
    write_to_file(val_images, 'val.txt')
    write_to_file(test_images, 'test.txt')

## 下面两个函数一起使用，根据IVD_data文件夹中的某个类别，已经含有了all子文件夹的所有内容，并创建了annotations和imgs，根据已有的list进行图片分类
def move_img_annotationV2(target_folder):
    ps_folder = target_folder  # 请替换成实际的文件夹路径
    splits = ["train", "test", "val"]
    for split in splits:
        split_file_path = os.path.join(ps_folder, 'list', f'{split}.txt')
        imgs_dest_folder = os.path.join(ps_folder, 'imgs', split)
        annotations_dest_folder = os.path.join(ps_folder, 'annotations', split)

        os.makedirs(imgs_dest_folder, exist_ok=True)
        os.makedirs(annotations_dest_folder, exist_ok=True)

        move_images_and_annotationsV1(ps_folder,imgs_dest_folder, annotations_dest_folder, split_file_path)

def move_images_and_annotationsV1(src_folder,imgs_dest_folder,annotations_dest_folder, split_file):
    with open(split_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            file_name = line.strip()
            
            # 构建源文件和目标文件的完整路径
            src_image_path = os.path.join(src_folder, 'all', 'imagesAll', file_name + '.png')
            src_annotation_path = os.path.join(src_folder, 'all', 'annotationsAll', file_name + '.png')

            dest_image_path = os.path.join(imgs_dest_folder, file_name + '.png')
            dest_annotation_path = os.path.join(annotations_dest_folder,  file_name + '.png')

            # 移动文件
            shutil.copy(src_image_path, dest_image_path)
            shutil.copy(src_annotation_path, dest_annotation_path)
            print(f"图片{file_name}移动完成")

# 将某个文件夹内的一级子文件夹内的所有图片的非零像素点改为class_number
def process_annotations(folder_path,class_number):
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        image_names = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]
        
        for image_name in image_names:
            image_path = os.path.join(subfolder_path, image_name)
            
            # Open the image
            image = Image.open(image_path)
            image = image.convert('L')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Set non-zero pixels to 1
            image_array[image_array != 0] = class_number
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(image_array)
            
            # Save the processed image
            processed_image.save(os.path.join(subfolder_path,  image_name))
            print(f"成功处理图片{image_name}")

#获得通道数和所有图像点的信息
def image_info(image_path):
    try:
        # 打开图像
        with Image.open(image_path) as img:
            # 获取图像的通道数
            channels = img.getbands()
            
            # 获取图像的大小
            width, height = img.size

            # 获取图像的每个像素点信息
            pixel_data = list(img.getdata())

            res =  {
                'channels': channels,
                'width': width,
                'height': height,
                #'pixel_data':pixel_data,
                'max':max(pixel_data)
            }
            print(res)
    except Exception as e:
        return {'error': str(e)}

# 将文件夹A内所有的子文件夹里面的.png图片，从三通道图片改为单通道图片
def convert_to_grayscale(folder_path):
    # 获取文件夹内所有子文件夹的路径
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    for subfolder in subfolders:
        # 获取子文件夹内所有的.png图片
        png_files = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.lower().endswith('.png')]

        for png_file in png_files:
            # 打开图像
            with Image.open(png_file) as img:
                # 将图像转为灰度图
                grayscale_img = img.convert('L')

                # 保存单通道图像
                grayscale_path = os.path.join(subfolder, os.path.basename(png_file))
                grayscale_img.save(grayscale_path)
                print(f"Converted {png_file} to {grayscale_path}")

        
def move_and_rename_images(folder_path, number):
    # 获取文件夹内所有的.png图片
    png_files = [f.path for f in os.scandir(folder_path) if f.is_file() and f.name.lower().endswith('.png')]

    # 创建目标文件夹
    annotation_folder = 'raw_data/Magnetic-tile-defect-datasets.-master/annotations'
    img_folder = 'raw_data/Magnetic-tile-defect-datasets.-master/imgs'

    # 遍历.png图片并移动到annotation，并以1.png, 2.png的格式进行命名
    for i, png_file in enumerate(png_files, start=number):
        new_name = f"{i}.png"
        destination_path_annotation = os.path.join(annotation_folder, new_name)
        shutil.move(png_file, destination_path_annotation)
        print(f"Moved {png_file} to {destination_path_annotation}")

        # 在原文件夹中查找相同名称的.jpg图片，并移动到img，并改变格式为.png
        jpg_file = os.path.splitext(png_file)[0] + '.jpg'
        if os.path.exists(jpg_file):
            destination_path_img = os.path.join(img_folder, new_name)
            shutil.move(jpg_file, destination_path_img)
            print(f"Moved {jpg_file} to {destination_path_img}")


if __name__ == "__main__":
    convert_to_grayscale('IVD_data/IVD_noPavement/annotations')
    

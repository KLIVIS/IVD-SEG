from PIL import Image
import os
import random
import numpy as np
import math
import cv2
import shutil



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

# 将某个文件夹内的一级子文件夹内的所有图片的非零像素点改为1
def process_annotations(folder_path):
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        image_names = [f for f in os.listdir(subfolder_path) if f.endswith('.png') or f.endswith('.jpg')]
        
        for image_name in image_names:
            image_path = os.path.join(subfolder_path, image_name)
            
            # Open the image
            image = Image.open(image_path)
            image = image.convert('L')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Set non-zero pixels to 1
            image_array[image_array != 0] = 1
            
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
                'pixel_data': pixel_data,
                'max':max(pixel_data)
            }
            print(res)
    except Exception as e:
        return {'error': str(e)}

def remove_prefix(folder_path, prefix):
    # 获取文件夹内所有的图片
    image_files = [f.path for f in os.scandir(folder_path) if f.is_file() and f.name.lower().endswith('.png')]

    # 遍历图片并修改名称
    for image_file in image_files:
        # 获取原始文件名和扩展名
        base_name, ext = os.path.splitext(os.path.basename(image_file))

        # 检查前缀是否存在，并且删除前缀
        if base_name.startswith(prefix):
            new_name = base_name[len(prefix):] + ext
            new_path = os.path.join(folder_path, new_name)

            # 重命名文件
            os.rename(image_file, new_path)

            print(f"Renamed {image_file} to {new_path}")

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

def convert_to_grayscale_and_replace(input_folder):
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            # 打开原始图片
            with Image.open(image_path) as img:
                # 将图片转为灰度模式
                grayscale_img = img.convert("L")

                # 保存单通道图片并替代原图片
                grayscale_img.save(image_path)

                print(f"Converted and replaced {image_path}")


def gray_to_rgb(folder_path):

    # 获取文件夹A中的所有文件
    file_list = os.listdir(folder_path)

    # 遍历文件夹A中的每个文件
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
    # 打开灰度图
            with Image.open(file_path) as img_gray:
                # 转为 NumPy 数组
                img_array = np.array(img_gray)

                # 创建三通道图像
                img_rgb = Image.fromarray(np.stack((img_array,) * 3, axis=-1), 'RGB')

                # 保存覆盖原有的图片
                img_rgb.save(file_path)

# 将某个文件夹内的所有图片的非零像素点改为1
def process_annotations_onefile(folder_path):
    
    subfolder_path = os.path.join(folder_path)
    image_names = [f for f in os.listdir(subfolder_path) if f.endswith('.png') or f.endswith('.jpg')]
    
    for image_name in image_names:
        image_path = os.path.join(subfolder_path, image_name)
        
        # Open the image
        image = Image.open(image_path)
        image = image.convert('L')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Set non-zero pixels to 1
        image_array[image_array != 0] = 1
        
        # Convert back to PIL Image
        processed_image = Image.fromarray(image_array)
        
        # Save the processed image
        processed_image.save(os.path.join(subfolder_path,  image_name))
        print(f"成功处理图片{image_name}")

def rotate_images(folder_path):
    # 获取文件夹内所有的图片
    image_files = [f.path for f in os.scandir(folder_path) if f.is_file() and f.name.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 遍历图片并进行旋转
    for image_file in image_files:
        # 打开图片
        with Image.open(image_file) as img:
            # 顺时针旋转90度
            rotated_img = img.transpose(Image.Transpose.ROTATE_270)

            # 保存旋转后的图片
            rotated_img.save(image_file)

            print(f"Rotated {image_file} by 90 degrees clockwise")


# 尺寸为4096，裁剪为256×256，每张长图裁剪16张图片
def crop_and_save(input_image_path, output_folder, start_index):
    # 打开原始照片
    with Image.open(input_image_path) as img:
        # 获取原始照片的宽度和高度
        width, height = img.size

        # 每张小照片的宽度
        tile_width = 256

        # 逐个裁剪并保存
        for i in range(16):
            # 计算裁剪区域的左上角和右下角坐标
            left = i * tile_width
            upper = 0
            right = left + tile_width
            lower = height

            # 裁剪照片
            tile = img.crop((left, upper, right, lower))

            # 保存裁剪后的照片
            output_path = os.path.join(output_folder, f"{start_index + i}.png")
            tile.save(output_path)

            print(f"Cropped and saved {output_path}")

# 裁剪图片
def process_images(img_folder, annotation_folder, output_folder):
    # 获取img文件夹中的所有图像文件
    img_files = [f for f in os.listdir(img_folder) if f.lower().endswith('.png')]

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    # 获取裁剪的起始序号
    start_index = 1
    # 逐个处理每张图像
    for img_file in img_files:
        img_path = os.path.join(img_folder, img_file)

        mask_name_without_extension = img_file.split('.')[0]

        mask_file = mask_name_without_extension + "_mask.png"
        annotation_path = os.path.join(annotation_folder, mask_file)


        # 裁剪并保存img中的图像
        crop_and_save(img_path, output_folder, start_index)

        # 如果annotation中有对应的图像，也进行裁剪并保存
        if os.path.exists(annotation_path):
            crop_and_save(annotation_path, f"{annotation_folder}_cut", start_index)
        start_index = start_index + 16

def is_all_black(image_path):
    # 打开图像
    with Image.open(image_path) as img:
        # 获取像素值
        pixels = img.getdata()

        # 判断是否所有像素值都为0
        return all(pixel == 0 for pixel in pixels)

def remove_all_black_images(annotation_folder, img_folder):
    # 获取所有图像文件
    annotation_files = [f for f in os.listdir(annotation_folder) if f.lower().endswith('.png')]

    # 逐个检查并删除
    for annotation_file in annotation_files:
        annotation_path = os.path.join(annotation_folder, annotation_file)
        img_path = os.path.join(img_folder, annotation_file)

        # 如果像素值全为0，则删除annotation和img中的图像
        if is_all_black(annotation_path):
            os.remove(annotation_path)
            print(f"Removed {annotation_path}")

            # 删除对应的img图像
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"Removed {img_path}")

# 将某个文件夹内的所有图片的非零像素点改为class_number
def process_annotations_Toclass(folder_path,class_number=1):

    image_names = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    for image_name in image_names:
        image_path = os.path.join(folder_path, image_name)
        
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
        processed_image.save(os.path.join(folder_path,  image_name))
        print(f"成功处理图片{image_name}")

# 将像素值大于150的点设置为255，小于或等于150的点设置为0
def process_annotations(folder_path):

    image_names = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    for image_name in image_names:
        image_path = os.path.join(folder_path, image_name)
        
        # Open the image
        image = Image.open(image_path)
        img  = image.convert('L')
         # 获取像素值
        pixels = img.getdata()

        # 对每个像素进行处理
        processed_pixels = [255 if pixel > 150 else 0 for pixel in pixels]

        # 创建新的图像对象
        processed_img = Image.new("L", img.size)
        processed_img.putdata(processed_pixels)

        
        # Save the processed image
        processed_img.save(os.path.join(folder_path,  image_name))
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
                #'pixel_data': pixel_data,
                'max':max(pixel_data)
            }
            print(res)
    except Exception as e:
        return {'error': str(e)}

def process_annotations_onefile(folder_path):
    
    subfolder_path = os.path.join(folder_path)
    image_names = [f for f in os.listdir(subfolder_path) if f.endswith('.png') or f.endswith('.jpg')]
    
    for image_name in image_names:
        image_path = os.path.join(subfolder_path, image_name)
        
        # Open the image
        image = Image.open(image_path)
        image = image.convert('L')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Set non-zero pixels to 1
        image_array[image_array != 0] = 1
        
        # Convert back to PIL Image
        processed_image = Image.fromarray(image_array)
        
        # Save the processed image
        processed_image.save(os.path.join(subfolder_path,  image_name))
        print(f"成功处理图片{image_name}")

# 单通道图片改为三通道图片
def convert_to_three_channels(folder_path):
    # 获取文件夹中所有图片的文件名
    file_names = [file for file in os.listdir(folder_path) if file.endswith(".png")]

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)

        # 打开单通道图片
        with Image.open(file_path) as img:
            # 将单通道图片转为三通道图片
            img_rgb = Image.merge("RGB", (img, img, img))

            # 保存覆盖原有图片
            img_rgb.save(file_path)

            print(f"Converted and saved {file_path}")

# 三通道图片改为单通道图片
def convert_to_single_channel(folder_path):
    # 获取文件夹中所有图片的文件名
    file_names = [file for file in os.listdir(folder_path) if file.endswith(".png")]

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)

        # 打开三通道图片
        with Image.open(file_path) as img:
            # 将三通道图片转为单通道图片
            img_gray = img.convert("L")

            # 保存覆盖原有图片
            img_gray.save(file_path)

            print(f"Converted and saved {file_path}")

# AITEX中有两个标注掩膜是同一张图片的，需要合并
def merge_mask(mask1_path, mask2_path):
    # 读取 mask1 和 mask2
    mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)

    # 合并到同一个灰度图
    merged_mask = np.maximum(mask1, mask2)

    # 保存合并后的灰度图
    cv2.imwrite('raw_data/AITEX/annotation/0097_030_03_mask.png', merged_mask)

def deleteNolabel():
    # 图像文件夹路径
    img_folder = "raw_data/DAGM/img/Test"

    # 标签文件夹路径
    label_folder = "raw_data/DAGM/Label"

    # 获取所有图像文件名
    img_files = os.listdir(img_folder)

    # 遍历图像文件夹中的图像
    for img_filename in img_files:
        img_path = os.path.join(img_folder, img_filename)

        # 检查文件是否是文件而不是文件夹
        if os.path.isfile(img_path):
            # 构造对应的标签文件名
            label_filename = img_filename.replace(".PNG", "_label.PNG")

            # 构造标签文件的路径
            label_path = os.path.join(label_folder, label_filename)

            # 如果对应的标签文件不存在，删除图像文件
            if not os.path.exists(label_path):
                os.remove(img_path)
                print(f"Deleted {img_filename} as there is no corresponding label.")

# resize成256*256
def resize_images(input_folder, size=(256, 256)):
    # 创建输出文件夹（如果不存在）
    output_folder = input_folder
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中的所有文件
    img_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.bmp', '.jpeg'))]

    for img_file in img_files:
        # 构建输入文件的完整路径
        input_path = os.path.join(input_folder, img_file)
        if False:
            # 分割文件名和扩展名
            name, ext = os.path.splitext(img_file)

            # 移除可能的前导零，并转换为整数
            number = int(name.lstrip("0"))

            # 构造新的文件名
            img_file = f"{number}{ext}"

        # 构建输出文件的完整路径
        output_path = os.path.join(output_folder, img_file)

        # 打开图像文件
        img = Image.open(input_path)

        # 进行 resize 操作
        resized_img = img.resize(size, Image.NEAREST)

        # 保存新的图像文件
        resized_img.save(output_path)
        print(f"Resized {img_file} to {size} and saved to {output_folder}")


if __name__ == "__main__":

    split_images_to_files('IVD_data/IVD_noPavement')
    move_img_annotationV2('IVD_data/IVD_noPavement')


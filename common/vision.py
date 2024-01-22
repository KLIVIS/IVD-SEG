import os
from PIL import Image
import numpy as np
import random

def process_images(folder_path):
    # 构建annotationsALL文件夹的路径
    annotations_path = os.path.join(folder_path, 'all', 'annotationsAll')
    # 构建maskvision文件夹的路径
    maskvision_path = os.path.join(folder_path, 'all', 'maskvision')

    # 如果maskvision文件夹不存在，则创建
    if not os.path.exists(maskvision_path):
        os.makedirs(maskvision_path)

    # 遍历annotationsALL文件夹中的所有图片
    for filename in os.listdir(annotations_path):
        if filename.endswith('.png'):
            image_path = os.path.join(annotations_path, filename)

            # 打开图片
            img = Image.open(image_path)
            
            # 将图片转换为NumPy数组
            img_array = np.array(img)

            # 将非零像素设为255
            img_array[img_array != 0] = 255

            # 创建处理后的Image对象
            processed_img = Image.fromarray(img_array)

            # 保存到maskvision文件夹中，保持相同的文件名
            processed_img_path = os.path.join(maskvision_path, filename)
            processed_img.save(processed_img_path)
            print(f"处理完图片{filename}")

def create_image_matrix(folder_path, output_path,matrix_size=(14, 14), image_count=196):
    prefix = 'IVD_data/'
    index = folder_path.find(prefix)
    if index != -1:
        # 使用切片获取位置之后的全部字符
        result_name = folder_path[index + len(prefix):]
        print(result_name)
    else:
        print("Prefix not found in the input string.")


    # 获取图片文件路径列表
    image_folder = os.path.join(folder_path, 'all', 'imagesAll')
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    # 如果图片数量不足，多次重复使用图片
    while len(image_files) < image_count:
        image_files += image_files

    # 随机抽取所需数量的图片
    selected_images = random.sample(image_files, image_count)

    # 创建空白图像矩阵
    matrix = np.zeros((256 * matrix_size[0], 256 * matrix_size[1], 3), dtype=np.uint8)

    # 将选定的图片填充到矩阵中
    for i, image_file in enumerate(selected_images):
        row = i // matrix_size[1]
        col = i % matrix_size[1]
        image_path = os.path.join(image_folder, image_file)
        img = Image.open(image_path)

        # 如果是单通道的图片，转为三通道
        if img.mode == 'L':
            img = img.convert('RGB')

        matrix[row * 256:(row + 1) * 256, col * 256:(col + 1) * 256, :] = np.array(img)

    # 在生成的图片中随机选择一张图片
    random_image_path = os.path.join(image_folder, random.choice(image_files))
    random_image = Image.open(random_image_path)
    random_image = random_image.resize((1736, 1736), Image.ANTIALIAS)


    # 将选定的图片覆盖到中间 6x6 的区域
    matrix[-1736:, -1736:, :] = np.array(random_image)
    # 在右方和下方生成宽度为56的白线将图片方阵分隔开
    matrix[-1792:-1736, -1792:, :] = 255  # 白线在下方
    matrix[-1792:, -1792:-1736, :] = 255  # 白线在右方

    # 将NumPy数组转换为图像
    result_image = Image.fromarray(matrix)
    result_image.save(os.path.join(output_path, result_name+'.png'))


def generate_image_matrix(dataset_path, output_path, matrix_size=(14, 14)):
    prefix = 'IVD_data/'
    index = dataset_path.find(prefix)
    if index != -1:
        # 使用切片获取位置之后的全部字符
        result_name = dataset_path[index + len(prefix):]
        print(result_name)
    else:
        print("Prefix not found in the input string.")

    images_path = os.path.join(dataset_path, 'all', 'imagesAll')

    # 获取所有图片的文件名
    image_filenames = [filename for filename in os.listdir(images_path) if filename.endswith('.png')]
    num_images = len(image_filenames)

    # 如果图片数量不足 98，可以重复使用
    while num_images < 196:
        image_filenames.extend(random.sample(image_filenames, min(196 - num_images, num_images)))

    # 打乱图片顺序
    random.shuffle(image_filenames)

    # 创建输出文件夹
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 生成图片方阵
    composite_image = Image.new('RGB', (matrix_size[0] * 256, matrix_size[1] * 256))

    for i in range(matrix_size[1]):
        for j in range(matrix_size[0]):
            # 获取当前位置的图片
            index = i * matrix_size[0] + j
            if index < len(image_filenames):
                img_filename = image_filenames[index]

                # 打开图片
                img = Image.open(os.path.join(images_path, img_filename))

                # 将图片贴到合适的位置
                composite_image.paste(img, (j * 256, i * 256))
            else:
                # 处理不足 98 组图片的情况，可以添加你需要的逻辑
                pass

    # 保存生成的图片方阵
    composite_image.save(os.path.join(output_path, result_name+'.png'))

def resize_images(input_folder, output_folder, target_size=(1536, 1536)):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有图片文件
    image_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # 循环处理每张图片
    for image_file in image_files:
        # 构建输入和输出文件的路径
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        # 打开图片并resize
        img = Image.open(input_path)
        resized_img = img.resize(target_size, Image.ANTIALIAS)

        # 保存resize后的图片
        resized_img.save(output_path)

if __name__ == "__main__":
    folder_path_list =(
        'IVD_data/AITEX',
        'IVD_data/BeanTech',
        'IVD_data/BSData',
        'IVD_data/Crackforest',
        'IVD_data/DAGM',
        'IVD_data/kolektorsdd',
        'IVD_data/kolektorsdd2',
        'IVD_data/magnetic_tile',
        'IVD_data/mvtec_ad',
        'IVD_data/phonescreen',
        'IVD_data/RSDDs',
        'IVD_data/THC'
    )
    input_folder_path = 'IVD_data/vision'
    output_folder_path = 'IVD_data/vision_smaller'
    resize_images(input_folder_path, output_folder_path)

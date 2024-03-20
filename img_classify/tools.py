import os

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array
from sklearn.model_selection import train_test_split
from PIL import Image
import random

def train_test_val_split(images=None, labels=None, train_size=.60, test_size=.20, val_size=.20, random_state=30):
    """
    Tách tập dữ liệu để Train/Test/Val
    :param images: Ndarray, chứa tập gồm toàn bộ ảnh của tập dữ liệu
    :param labels: Ndarray, chứa nhãn của tập dữ liệu
    :param train_size: Float, đặt kích thước tập Train (Mặc định: .60)
    :param test_size: Float, đặt kích thước tập Test (Mặc định: .20)
    :param val_size: Float, đặt kích thước tập Val (Mặc định: .20)
    :param random_state: Int, đặt tỉ lệ ngẫu nhiên
    :return List gồm 3 tuple con chứa ảnh & nhãn đã tách cho mỗi tập Train/Test/Val
    """
    if type(images) is not np.ndarray:
        raise TypeError('Tham số images phải là một Ndarray !')

    if len(images) == 0:
        raise IndexError('Tham số images chứa mảng rỗng !')

    if type(images) is not np.ndarray:
        raise TypeError('Tham số labels phải là một Ndarray !')

    if len(images) == 0:
        raise IndexError('Tham số images chứa mảng rỗng !')

    if not all([type(train_size) is float, type(test_size) is float, type(val_size) is float]):
        raise ValueError(
            'Tham số train_size, test_size, val_size phải là kiểu Float !')

    if not .01 <= train_size <= 1. or not .01 <= test_size <= 1. or not .01 <= val_size <= 1.:
        raise ValueError(
            'Tham số train_size, test_size, val_size phải từ .001 - 1. !')

    if not .01 <= train_size + test_size + val_size <= 1.:
        raise ValueError(
            'Tổng của 3 giá trị train_size, test_size, val_size phải từ .01 - 1. !')

    try:
        train_images, test_images, train_labels, test_labels = train_test_split(
            images,
            labels,
            train_size=train_size,
            test_size=test_size,
            stratify=labels,
            random_state=random_state
        )
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_images,
            train_labels,
            test_size=val_size,
            stratify=train_labels,
            random_state=random_state
        )

        print('== Thống kê số lượng ảnh sau khi tách ==')
        print(f'Train: {train_images.shape[0]} ảnh')
        print(f'Test: {test_images.shape[0]} ảnh')
        print(f'Val: {val_images.shape[0]} ảnh')

        return [(train_images, train_labels), (test_images, test_labels), (
                val_images, val_labels)]

    except Exception as ex:
        print('Quá trình tách dữ liệu bị lỗi !', str(ex))


def image_augmentation_by_class(ds_path=None, batch_size=32, num_img=50, img_size=(128, 128), img_model=None,
                                classes=None,
                                exclude_class=None):
    """
    Trình tăng cường ảnh theo nhãn
    :param ds_path: Str, đường dẫn tập dữ liệu
    :param batch_size: Int, kích thước mỗi lô
    :param num_img: Int, số lượng ảnh cho mỗi nhãn (Mặc định: 50)
    :param img_size: Tuple/List, kích thước ảnh (Mặc định: 128 x 128)
    :param img_model: ImageDataGeneration, mô hình tăng cường ảnh (dùng hàm ImageDataGeneration)
    :param classes: Tuple/List, chứa các lớp của tập dữ liệu
    :param exclude_class: Tuple/List, chứa các nhãn bị loại trừ khi tăng cường ảnh
    :return Xuất ảnh đã tăng cường vào từng thư mục con của mỗi nhãn
    """

    if ds_path == '' or ds_path is None:
        raise ValueError('Tham số ds_path không được để trống !')

    if not os.path.exists(ds_path):
        raise FileNotFoundError(
            'Tham số ds_path chứa đường dẫn thư mục sai hoặc không tồn tại !')

    if type(num_img) is not int:
        raise TypeError('Tham số num_img phải là kiểu Int !')

    if num_img <= 0:
        raise ValueError('Tham số num_img phải lớn hơn hoặc bằng 0 !')

    if type(img_size) not in (tuple, list):
        raise TypeError('Tham số img_size phải là kiểu Tuple/List !')

    if type(img_model) is not ImageDataGenerator:
        raise TypeError('Tham số img_size phải là kiểu ImageDataGenerator !')

    if type(classes) not in (tuple, list):
        raise TypeError('Tham số classes phải là kiểu Tuple/List !')

    print(f'=> Bắt đầu thêm {batch_size * num_img} ảnh cho các nhãn...')
    for label in classes:
        if label in exclude_class:
            continue
        images = []
        image_path = os.path.join(ds_path, label)
        for image in os.listdir(image_path):
            if image.split('.')[1] in ('jpg', 'png', 'jpeg'):
                with Image.open(os.path.join(image_path, image)) as img:
                    img = img.convert('RGB')
                    img = img.resize(img_size)
                    img = img_to_array(img)
                    images.append(img)
        images = np.asarray(images, dtype=np.float32)
        i = 0
        for _ in img_model.flow(
            images,
            batch_size=batch_size,
            save_to_dir=image_path,
            save_prefix='aug',
            save_format='jpg'
        ):
            i += 1
            if i > num_img:
                break


def fix_imbalance_with_image_augmentation(ds_path=None, img_size=(128, 128), img_model=None,
                                          classes=None):
    """
    Tái cân bằng tập dữ liệu bằng phương pháp tăng cuờng ảnh
    :param ds_path: Str, đường dẫn tập dữ liệu
    :param img_size: Tuple/List, kích thước ảnh (Mặc định: 128 x 128)
    :param img_model: ImageDataGeneration, mô hình tăng cường ảnh (dùng hàm ImageDataGeneration)
    :param classes: Tuple/List, chứa nhãn của tập dữ liệu
    :return Xuất ảnh đã tăng cường vào từng thư mục con của mỗi nhãn
    """

    if ds_path == '' or ds_path is None:
        raise ValueError('Tham số ds_path không được để trống !')

    if not os.path.exists(ds_path):
        raise FileNotFoundError(
            'Tham số ds_path chứa đường dẫn thư mục sai hoặc không tồn tại !')

    if type(img_size) not in (tuple, list):
        raise TypeError('Tham số img_size phải là kiểu Tuple/List !')

    if type(img_model) is not ImageDataGenerator:
        raise TypeError('Tham số img_size phải là kiểu ImageDataGenerator !')

    if type(classes) not in (tuple, list):
        raise TypeError('Tham số classes phải là kiểu Tuple/List !')

    count_files_by_class = []

    for i in classes:
        path = os.path.join(ds_path, i)
        count = len(os.listdir(path))
        count_files_by_class.append(count)
    v_max = max(count_files_by_class)
    for a in range(len(classes)):
        a_class = classes[a]
        num_img = v_max - count_files_by_class[a]
        if count_files_by_class[a] / v_max >= .95:
            print(f'Loại trừ nhãn {a_class} do đã cân bằng !')
        else:
            images = []
            image_path = os.path.join(ds_path, a_class)
            print(f'Bắt đầu khởi tạo thêm {num_img} ảnh cho nhãn {a_class}')
            for image in os.listdir(image_path):
                if image.split('.')[1] in ('jpg', 'png', 'webp'):
                    with Image.open(os.path.join(image_path, image)) as img:
                        img = img.convert('RGB')
                        img = img.resize(img_size)
                        img = img_to_array(img)
                        images.append(img)
            images = np.asarray(images, dtype=np.float32)
            i = 0
            for _ in img_model.flow(
                images,
                seed=69,
                batch_size=1,
                save_to_dir=image_path,
                save_prefix='aug',
                save_format='jpg'
            ):
                i += 1
                if i > num_img:
                    break


def random_rgb_color(num_color=3):
    """
    Trình tạo màu RGB ngẫu nhiên
    :param num_color: Int, số lượng màu muốn khởi tạo (Mặc định: 3)
    :return List chứa màu RGB được tạo ngẫu nhiên
    """

    if type(num_color) is not int or num_color <= 0:
        raise ValueError('Tham số num_color phải là kiểu Int & lớn hơn 0 !')

    colors = tuple(np.random.choice(range(256), size=3))
    colors = [colors for _ in range(num_color)]
    return colors

def random_hex_color(num_color=3):
    """
    Trình tạo màu Hex ngẫu nhiên
    :param num_color: Int, số lượng màu muốn khởi tạo (Mặc định: 3)
    :return List chứa màu Hex được tạo ngẫu nhiên
    """
    if type(num_color) is not int or num_color <= 0:
        raise ValueError('Tham số num_color phải là kiểu Int & lớn hơn 0 !')
    
    hex_colors = '#' + str().join(random.choice('ABCDEF0123456789') for _ in range(6))
    hex_colors = [hex_colors for _ in range(num_color)]
    return hex_colors
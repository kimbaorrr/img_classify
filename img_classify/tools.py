import numpy as np
from sklearn.model_selection import train_test_split
import random
from albumentations import Compose

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


def fix_imbalance_with_image_augmentation(images=None, labels=None, compose=None, classes=None):
    """
    Tái cân bằng tập dữ liệu bằng phương pháp tăng cường ảnh dùng thư viện Albumentation
    :param images: Ndarray, chứa tập ảnh đồng kích thước shape (n, w, h, c)
    :param labels: Ndarray, chứa tập nhãn có kiểu số nguyên 
    :param img_model: Albumentation Compose, mô hình tăng cường ảnh (dùng hàm Compose của thư viện Albumentation)
    :param classes: Tuple/List, chứa nhãn của tập dữ liệu
    :return Xuất ảnh đã tăng cường vào 2 tập images và labels
    """

    if not type(images) is np.ndarray:
        raise TypeError('Tham số images phải là kiểu Ndarray !')
    
    if not type(labels) is np.ndarray:
        raise TypeError('Tham số labels phải là kiểu Ndarray !')

    if type(compose) is not Compose:
        raise TypeError('Tham số compose phải là kiểu Compose !')

    if type(classes) not in (tuple, list):
        raise TypeError('Tham số classes phải là kiểu Tuple/List !')

    x = np.unique(labels)
    y = np.bincount(labels)
    count_file_per_class = [count / max(y) for count in y]

    # Kiểm tra độ cân bằng giữa các lớp
    for cls, count_file in zip(x, count_file_per_class):
        if count_file < .95:
            print(f'Nhãn {classes[cls]} bị mất cân bằng !')

    # Tạo vòng lặp cân bằng lớp
    while True:
        balanced = True
        for i, (count_file, label) in enumerate(zip(count_file_per_class, labels)):
            if count_file < .95:
                balanced = False
                augmented = compose(image=images[i])
                labels = np.append(labels, label)
                images = np.append(images, augmented['image'])

        # Tính toán lại tỉ lệ các nhãn
        y = np.bincount(labels)
        count_file_per_class = [count / max(y) for count in y]

        # Nếu tất cả đã cân bằng thì dừng vòng lặp
        if balanced:    
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

    hex_colors = '#' + str().join(random.choice('ABCDEF0123456789')
                                  for _ in range(6))
    hex_colors = [hex_colors for _ in range(num_color)]
    return hex_colors

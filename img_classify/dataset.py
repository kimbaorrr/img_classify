import os
import random
from glob import glob

import numpy as np
import seaborn as sns
from matplotlib import image as mlt, pyplot as plt


def check_balance(dir_path=None, classes=None, ds_name='Train', img_save_path=''):
    """
    Kiểm tra mức độ cân bằng giữa các lớp
    Args:
            dir_path: Str, đường dẫn thư mục Train/Test
            classes: Tuple/List, chứa các lớp của tập dữ liệu
            ds_name: Str, loại tập dữ liệu (Mặc định: Train)
            img_save_path: Str, vị trí xuất ảnh thống kê (Mặc định: Vị trí hiện tại)
    Returns:
            In đồ thị thống kê & tính độ chênh lệch giữa các lớp
    """

    if dir_path == '' or dir_path is None:
        raise ValueError('Tham số dir_path không được để trống !')

    if not os.path.exists(dir_path):
        raise FileNotFoundError(
            'Tham số dir_path chứa đường dẫn đến thư mục không tồn tại !')

    if type(classes) not in (tuple, list):
        raise TypeError('Tham số classes phải là Tuple hoặc List !')

    if len(classes) == 0:
        raise IndexError('Tham số classes chứa mảng rỗng !')

    y = []
    for i in classes:
        y_path = os.path.join(dir_path, i)
        count = len(os.listdir(y_path))
        y.append(count)
    plt.title(f'Thống kê số lượng ảnh của từng lớp thuộc tập {ds_name}')
    sns.barplot(
        x=classes,
        y=y
    )
    plt.xlabel('Nhãn')
    plt.ylabel('Số lượng ảnh')
    if img_save_path != '':
        plt.savefig(img_save_path)
    plt.show()
    v_max = max(y)
    print(
        f'== MỨC CHÊNH LỆCH GIỮA CÁC LỚP TẬP {ds_name.upper()} SO VỚI NHÃN CAO NHẤT ==')
    for a in range(len(y)):
        print(f'Nhãn {classes[a]}:', np.round(y[a] / v_max * 100, 2))


def rand_image_viewer(dir_path=None, classes=None, cmap='viridis'):
    """
    Trình xem ảnh ngẫu nhiên
    Args:
            dir_path: Str, đường dẫn thư mục ảnh
            classes: Tuple/List, chứa các lớp của tập dữ liệu
            cmap: Str, tên bản đồ màu (Mặc định: viridis)
    Returns:
            In ảnh cùng với lớp (x) và tên tệp (y) lên màn hình
    """

    if dir_path == '' or dir_path is None:
        raise ValueError('Tham số dir_path không được để trống !')

    if not os.path.exists(dir_path):
        raise FileNotFoundError(
            'Tham số dir_path chứa đường dẫn đến thư mục không tồn tại !')

    if type(classes) not in (tuple, list):
        raise TypeError('Tham số classes phải là Tuple hoặc List !')

    if len(classes) == 0:
        raise IndexError('Tham số classes chứa mảng rỗng !')

    if cmap not in ('viridis', 'gray'):
        raise ValueError(
            'Tham số cmap phải được chỉ định là viridis hoặc gray !')

    rand_class = random.sample(classes, 1)[0]
    path = os.path.join(dir_path, rand_class)
    for i in ('*.jpg', '*.png, *.jpeg'):
        img_to_show = random.sample(glob(os.path.join(path, i)), 1)[0]
    image = mlt.imread(os.path.join(path, img_to_show))
    plt.imshow(image, cmap=cmap)
    plt.xlabel(rand_class)
    plt.colorbar()
    plt.show()

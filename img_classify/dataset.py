import os
import random
from glob import glob

import numpy as np
import seaborn as sns
from matplotlib import image as mlt, pyplot as plt

from img_classify.others import sns_global

sns_global()

def check_dir(dir_path=None):
	"""
		Đếm số tệp & thư mục con có trong thư mục cha
		Args:
			dir_path: Str, đường dẫn thư mục cha
		Returns:
			Liệt kê số tệp & thư mục con có trong thư mục cha
	"""
	if dir_path == '' or dir_path is None:
		raise ValueError('Tham số dir_path không được để trống !')

	if not os.path.exists(dir_path):
		raise FileNotFoundError('Parameter dir_path contains an invalid or non-existent directory path !')

	for root_dir, sub_dir, files in os.walk(dir_path):
		print(f'=> Have {len(sub_dir)} subdirectories & {len(files)} files in {root_dir}')

def create_label_from_dir(train_path=None):
	"""
	Khởi taọ tập nhãn từ tên các thư mục con trong thư mục Train
	Args:
		train_path: Str, đường dẫn thư mục Train
	Returns:
		List/Tuple chứa nhãn của tập dữ liệu
	"""
	if train_path == '' or train_path is None:
		raise ValueError('Parameter train_path can not be empty !')

	if not os.path.exists(train_path):
		raise FileNotFoundError('Tham số dir_path chứa đường dẫn đến thư mục không tồn tại !')

	return sorted(os.listdir(train_path))

def check_balance(dir_path=None, class_names=None, ds_name='Train', img_save_path='./check_balance.jpg'):
	"""
	Kiểm tra mức độ cân bằng của tập dữ liệu
	Args:
		dir_path: Str, đường dẫn thư mục Train/Test
		class_names: Tuple/List, chứa nhãn của tập dữ liệu
		ds_name: Str, loại tập dữ liệu (Mặc định: Train)
		img_save_path: Str, vị trí xuất ảnh thống kê (Mặc định: Vị trí hiện tại)
	Returns:
		In đồ thị thống kê & tính độ chênh lệch giữa các nhãn
	"""

	if dir_path == '' or dir_path is None:
		raise ValueError('Tham số dir_path không được để trống !')

	if not os.path.exists(dir_path):
		raise FileNotFoundError('Tham số dir_path chứa đường dẫn đến thư mục không tồn tại !')

	if type(class_names) not in (tuple, list):
		raise TypeError('Tham số class_names phải là Tuple hoặc List !')

	if len(class_names) == 0:
		raise IndexError('Tham số class_names chứa mảng rỗng !')

	y = []
	for i in range(len(class_names)):
		y_path = str(os.path.join(dir_path, class_names[i]))
		count = len(os.listdir(y_path))
		y.append(count)
	plt.title(f'Thống kê số lượng ảnh của từng nhãn thuộc tập {ds_name}')
	sns.barplot(
		x=class_names,
		y=y
	)
	plt.xlabel('Nhãn')
	plt.ylabel('Số lượng ảnh')
	if img_save_path != '':
		plt.savefig(img_save_path)
	plt.show()
	v_max = max(y)
	print(f'== MỨC CHÊNH LỆCH GIỮA CÁC NHÃN TẬP {ds_name.upper()} SO VỚI NHÃN CAO NHẤT ==')
	for a in range(len(y)):
		print(f'Nhãn {class_names[a]}:', np.round(y[a] / v_max * 100, 3))

def rand_image_viewer(dir_path=None, class_names=None, cmap='viridis'):
	"""
	Trình xem ảnh ngẫu nhiên
	Args:
		dir_path: Str, đường dẫn thư mục ảnh
		class_names: Tuple/List, chứa nhãn của tập dữ liệu
		cmap: Str, tên bản đồ màu (Default: viridis)
	Returns:
		In ảnh cùng với nhãn (x) và tên tệp (y) lên màn hình
	"""

	if dir_path == '' or dir_path is None:
		raise ValueError('Tham số dir_path không được để trống !')

	if not os.path.exists(dir_path):
		raise FileNotFoundError('Tham số dir_path chứa đường dẫn đến thư mục không tồn tại !')

	if type(class_names) not in (tuple, list):
		raise TypeError('Tham số class_names phải là Tuple hoặc List !')

	if len(class_names) == 0:
		raise IndexError('Tham số class_names chứa mảng rỗng !')

	if cmap not in ('viridis', 'gray'):
		raise ValueError('Tham số cmap phải được chỉ định là viridis hoặc gray !')

	a = random.randint(0, len(class_names) - 1)
	path = str(os.path.join(dir_path, class_names[a]))
	for i in ('*.jpg', '*.png, *.jpeg'):
		images_list = random.sample(glob(os.path.join(path, i)), 1)
	show_image = images_list[0]
	image = mlt.imread(path + '/' + show_image)
	plt.imshow(image, cmap=cmap)
	plt.xlabel(class_names[a])
	plt.colorbar()
	plt.show()

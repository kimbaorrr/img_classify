import os
import random
from glob import glob

import numpy as np
import seaborn as sns
from matplotlib import image as mlt, pyplot as plt
from img_classify.others import sns_global

sns_global()

class CheckFiles:
	"""
	Check files & subdirectories in the parent directory
	Args:
		dir_path: Str, Parent directory path
	Returns:
		Count the number of files & subdirectories in the parent directory
	"""

	def __init__(
		self,
		dir_path=None
	):

		if dir_path == '' or dir_path is None:
			raise ValueError('Parameter dir_path can not be empty !')

		if not os.path.exists(dir_path):
			raise FileNotFoundError('Parameter dir_path contains an invalid or non-existent directory path !')

		self.dir_path = dir_path

		for root_dir, sub_dir, files in os.walk(self.dir_path):
			print(f'=> Have {len(sub_dir)} subdirectories & {len(files)} files in {root_dir}')

class CreateLabelFromDir:
	"""
	Create labels for a dataset (Get the value of each subdirectory as the label)
	Args:
		train_path: Str, Train directory path
	Returns:
		List/Tuple containing labels of dataset
	"""

	def __init__(
		self,
		train_path=None
	):

		if train_path == '' or train_path is None:
			raise ValueError('Parameter train_path can not be empty !')

		if not os.path.exists(train_path):
			raise FileNotFoundError('Parameter train_path contains an invalid or non-existent directory path !')

		self.train_path = train_path
		self.output = None

		self.output = sorted(os.listdir(self.train_path))
		print(f'=> Your label are: {self.output}')

class CheckBalance:
	"""
	Check balance of label of dataset
	Args:
		dir_path: Str, Train/Test directory path
		class_names: Tuple/List/Ndarray, containing label of dataset
		ds_name: Str, The name of dataset (Train/Test) (Default: Train)
		img_save_path: Str, vị trí xuất ảnh thống kê (Mặc định: Vị trí hiện tại)
	Returns:
		In đồ thị thống kê & tính độ chênh lệch giữa các nhãn
	"""

	def __init__(
		self,
		dir_path=None,
		class_names=None,
		ds_name='Train',
		img_save_path='./check_balance.jpg'
	):

		if dir_path == '' or dir_path is None:
			raise ValueError('Tham số dir_path không được để trống !')

		if not os.path.exists(dir_path):
			raise FileNotFoundError('Parameter dir_path contains an invalid or non-existent directory path !')

		if type(class_names) not in (tuple, list):
			raise TypeError('Tham số class_names phải là Tuple hoặc List !')

		if len(class_names) == 0:
			raise IndexError('Tham số class_names chứa mảng rỗng !')

		self.dir_path = dir_path
		self.class_names = class_names
		self.img_save_path = img_save_path
		self.ds_name = ds_name

		y = []
		for i in range(len(self.class_names)):
			path = os.path.join(self.dir_path, self.class_names[i])
			count = len(os.listdir(path))
			y.append(count)
		plt.title(f'Thống kê số lượng ảnh của từng nhãn thuộc tập {self.ds_name}')
		sns.barplot(
			x=self.class_names,
			y=y
		)
		plt.xlabel('Nhãn')
		plt.ylabel('Số lượng ảnh')
		if self.img_save_path != '':
			plt.savefig(self.img_save_path)
		plt.show()
		v_max = max(y)
		print(f'== MỨC CHÊNH LỆCH GIỮA CÁC NHÃN TẬP {self.ds_name.upper()} SO VỚI NHÃN CAO NHẤT==')
		for a in range(len(y)):
			print(f'Nhãn {self.class_names[a]}:', np.round(y[a] / v_max * 100, 3))

class RandImageViewer:
	"""
	Random image viewer
	Args:
		dir_path: Str, Image directory path
		class_names: Tuple/List/Ndarray, Containing label of dataset
		cmap: Str, Choosing colormap (Default: viridis)
	Returns:
		In ảnh cùng với nhãn (x) và tên tệp (y) lên màn hình
	"""

	def __init__(
		self,
		dir_path=None,
		class_names=None,
		cmap='viridis'
	):

		if dir_path == '' or dir_path is None:
			raise ValueError('Tham số dir_path không được để trống !')

		if not os.path.exists(dir_path):
			raise FileNotFoundError('Tham số dir_path chứa đường dẫn thư mục sai hoặc không tồn tại !')

		if type(class_names) not in (tuple, list):
			raise TypeError('Tham số class_names phải là Tuple hoặc List !')

		if len(class_names) == 0:
			raise IndexError('Tham số class_names chứa mảng rỗng !')

		if cmap not in ('viridis', 'gray'):
			raise ValueError('Tham số cmap phải được chỉ định là viridis hoặc gray !')

		self.dir_path = dir_path
		self.class_names = class_names
		self.cmap = cmap

		a = random.randint(0, len(self.class_names) - 1)
		path = os.path.join(self.dir_path, self.class_names[a])
		for i in ('*.jpg', '*.png, *.jpeg'):
			images_list = random.sample(glob(os.path.join(path, i)), 1)
		show_image = images_list[0]
		image = mlt.imread(path + '/' + show_image)
		plt.imshow(image, cmap=self.cmap)
		plt.xlabel(self.class_names[a])
		plt.colorbar()
		plt.show()

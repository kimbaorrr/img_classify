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
	Kiểm tra tệp & thư mục con trong thư mục cha
	Args:
		dir_path: Str, đường dẫn thư mục cha
	Returns:
		Đếm số tệp & thư mục con có trong thư mục cha
	"""

	def __init__(
		self,
		dir_path=None
	):

		if dir_path == '' or dir_path is None:
			raise ValueError('Tham số dir_path không được để trống !')
			return

		if not os.path.exists(dir_path):
			raise FileNotFoundError('Tham số dir_path chứa đường dẫn thư mục sai hoặc không tồn tại !')
			return

		self.dir_path = dir_path

		for dir_root, dir_sub, file_names in os.walk(self.dir_path):
			print(f'=> Có {len(dir_sub)} thư mục con và {len(file_names)} tệp trong {dir_root}')

class CreateLabelFromDir:
	"""
	Tạo nhãn cho tập dữ liệu (Lấy giá trị của từng thư mục con làm nhãn)
	Args:
		train_path: Str, đường dẫn thư mục Train
	Returns:
		List/Tuple chứa nhãn của tập dữ liệu
	"""

	def __init__(
		self,
		train_path=None
	):

		if train_path == '' or train_path is None:
			raise ValueError('Tham số train_path không được để trống !')
			return

		if not os.path.exists(train_path):
			raise FileNotFoundError('Tham số train_path chứa đường dẫn thư mục sai hoặc không tồn tại !')
			return

		self.train_path = train_path
		self.output = None

		self.output = sorted(os.listdir(self.train_path))
		print(f'=> Nhãn của bạn là: {self.output}')

class CheckBalance:
	"""
	Kiểm tra độ cân bằng của tập dữ liệu
	Args:
		dir_path: Str, Đường dẫn thư mục Train/Test
		class_names: Tuple/List/Ndarray, chứa nhãn của tập dữ liệu
		ds_name: Str, tên của tập dữ liệu (Train/Test) (Mặc định: Train)
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
			return

		if not os.path.exists(dir_path):
			raise FileNotFoundError('Tham số dir_path chứa đường dẫn thư mục sai hoặc không tồn tại !')
			return

		if type(class_names) not in (tuple, list):
			raise TypeError('Tham số class_names phải là Tuple hoặc List !')
			return

		if len(class_names) == 0:
			raise IndexError('Tham số class_names chứa mảng rỗng !')
			return

		self.dir_path = dir_path
		self.class_names = class_names
		self.img_save_path = img_save_path
		self.ds_name = ds_name

		try:
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
			if not self.img_save_path == '':
				plt.savefig(self.img_save_path)
			plt.show()
			v_max = max(y)
			print(f'== MỨC CHÊNH LỆCH GIỮA CÁC NHÃN TẬP {self.ds_name.upper()} SO VỚI NHÃN CAO NHẤT==')
			for a in range(len(y)):
				print(f'Nhãn {self.class_names[a]}:', np.round(y[a] / v_max * 100, 3))
		except:
			raise Exception('Quá trình thống kê bị lỗi !')

class ViewRandImage:
	"""
	Trình in ảnh ngẫu nhiên
	Args:
		dir_path: Str, Đường dẫn thư mục chứa ảnh
		class_names: Tuple/List/Ndarray, chứa nhãn của tập dữ liệu
		cmap: Str, chế độ ánh xạ màu (Mặc định: viridis)
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
			return

		if not os.path.exists(dir_path):
			raise FileNotFoundError('Tham số dir_path chứa đường dẫn thư mục sai hoặc không tồn tại !')
			return

		if type(class_names) not in (tuple, list):
			raise TypeError('Tham số class_names phải là Tuple hoặc List !')
			return

		if len(class_names) == 0:
			raise IndexError('Tham số class_names chứa mảng rỗng !')
			return

		if cmap not in ('viridis', 'gray'):
			raise ValueError('Tham số cmap phải được chỉ định là viridis hoặc gray !')
			return

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

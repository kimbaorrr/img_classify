import os

import kaggle as kg

from img_classify.others import get_root_dir

class KaggleDownloader:
	"""
	Trình tải tập dữ liệu từ thư viện Kaggle
	Args:
		dataset_name: Str, đặt tên cho tập dữ liệu
		dataset_url: Str, đường dẫn đến tập dữ liệu trên Kaggle
	Return:
		Tập dữ liệu đã được tải & giải nén
	"""

	def __init__(
		self,
		dataset_name=None,
		dataset_url=None
	):

		if dataset_name == '' or dataset_name is None:
			raise ValueError('Tham số dataset_name không được để trống !')
			return

		if dataset_url == '' or dataset_url is None:
			raise ValueError('Tham số dataset_url không được để trống !')
			return

		self.dataset_name = dataset_name.lower()
		self.dataset_url = dataset_url.lower()
		self.save_path = os.path.join(get_root_dir(), 'datasets')

		try:
			# Kiểm tra bộ dữ liệu đã có hay chưa ?
			ds = os.path.join(self.save_path, self.dataset_name)
			if os.path.isdir(ds):
				return
			# Xác thực đăng nhập
			kg.api.authenticate()
			# Tải dữ liệu
			kg.api.dataset_download_files(
				self.dataset_url,
				ds,
				unzip=True,
				quiet=False
			)
			print('=> Tải dữ liệu thành công !')
			print(f'=> Tập dữ liệu của bạn được lưu tại: {ds}')
		except:
			raise Exception(f'Quá trình tải bị lỗi !')

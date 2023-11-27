import os

import kaggle as kg

from img_classify.others import get_root_dir

class KaggleDownloader:
	"""
	Dataset downloader from Kaggle library
	Args:
		dataset_name: Str, Set name for dataset
		dataset_url: Str, Dataset URL suffix in format <owner>/<dataset-name>
	Return:
		The dataset has been downloaded & extracted to datasets folder
	"""

	def __init__(
		self,
		dataset_name=None,
		dataset_url=None
	):

		if dataset_name == '' or dataset_name is None:
			raise ValueError('Parameter dataset_name can not be empty !')
			return

		if dataset_url == '' or dataset_url is None:
			raise ValueError('Parameter dataset_url can not be empty !')
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
		except Exception:
			raise Exception('Quá trình tải bị lỗi !')

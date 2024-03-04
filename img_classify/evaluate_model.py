import os

import numpy as np
import seaborn as sns
from keras.callbacks import History
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

class EvalofModelwithImage:
	"""
	Trình đánh giá mô hình (Đầu vào 1 ảnh)
	Args:
		images: Ndarray, chứa tập gồm nhiều ảnh
		pred_labels: Ndarray, mang giá trị nhãn đã dự đoán bằng hàm model.predict()
		true_labels: Ndarray, chứa tập nhãn gốc để so sánh với kết quả dự đoán
		class_names: Tuple/List/Ndarray, chứa nhãn của tập dữ liệu
		cmap: Str, chế độ ánh xạ màu (Mặc định: 'viridis')
	Returns:
		In khung nhìn chứa ảnh & đồ thị đánh giá độ chính xác của mô hình
	"""

	def __init__(
			self,
			images=None,
			pred_labels=None,
			true_labels=None,
			class_names=None,
			cmap='viridis'
	):

		if type(images) is not np.ndarray:
			raise TypeError('Tham số images phải là một Ndarray !')

		if len(images) == 0:
			raise IndexError('Tham số images chứa mảng rỗng !')

		if type(pred_labels) is not np.ndarray:
			raise TypeError('Tham số pred_labels phải là một Ndarray !')

		if len(pred_labels) == 0:
			raise IndexError('Tham số pred_labels chứa mảng rỗng !')

		if type(true_labels) is not np.ndarray:
			raise TypeError('Tham số true_labels phải là một Ndarray !')

		if len(true_labels) == 0:
			raise IndexError('Tham số true_labels chứa mảng rỗng !')

		if type(class_names) not in (tuple, list):
			raise TypeError('Tham số class_names phải là Tuple/List !')

		if len(class_names) == 0:
			raise IndexError('Tham số class_names chứa mảng rỗng !')

		if cmap not in ('viridis', 'gray'):
			raise ValueError('Tham số cmap chỉ được chỉ định là viridis hoặc gray !')

		self.images = images
		self.pred_labels = pred_labels
		self.img_num = random.randint(0, len(images) - 1)
		self.true_labels = true_labels
		self.class_names = class_names
		self.cmap = cmap

		self.img_true_label = np.argmax(self.true_labels[self.img_num])
		self.img = self.images[self.img_num]
		self.img_pred_label = np.argmax(self.pred_labels[self.img_num])
		self.img_pred_acc = np.round(np.max(self.pred_labels[self.img_num]) * 100, 3)

	def plot_image(self):
		plt.imshow(self.img, cmap=self.cmap)
		color = 'b' if self.img_pred_label == self.img_true_label else 'r'
		plt.xlabel(
			f'{self.class_names[self.img_pred_label]} ({self.img_pred_acc}%)',
			color=color
		)
		plt.ylabel(
			f'{self.class_names[self.img_true_label]}'
		)
		plt.grid(False)

	def plot_value_array(self):
		len_of_class_names = len(self.class_names)
		plt.xticks(range(len_of_class_names))
		this_plot = plt.bar(range(len_of_class_names), self.pred_labels[self.img_num], color="gray")
		plt.ylim([0, 1])
		this_plot[self.img_pred_label].set_color('r')
		this_plot[self.img_true_label].set_color('b')

def eval_of_model_with_images(num_rows=5, num_cols=3, images=None, pred_labels=None, true_labels=None,
                              class_names=None, cmap='viridis', img_save_path='./eval_of_model_with_images.jpg'):
	"""
	Trình đánh giá mô hình qua nhiều ảnh
	:param num_rows: Int, số dòng m của ma trận
	:param num_cols: Int, số cột n của ma trận
	:param images: Ndarray, chứa tập gồm nhiều ảnh
	:param pred_labels: Ndarray, mang giá trị nhãn đã dự đoán bằng hàm model.predict()
	:param true_labels: Ndarray, chứa tập nhãn gốc để so sánh với kết quả dự đoán
	:param class_names: Tuple/List/Ndarray, chứa nhãn của tập dữ liệu
	:param cmap: Str, chế độ ánh xạ màu (Mặc định: 'viridis')
	:param img_save_path: Str, vị trí xuất ảnh đánh giá
	:: In khung nhìn trên ma trận num_rows * num_cols chứa ảnh & đồ thị đánh giá độ chính xác của mô hình
	"""
	if num_rows < 1 or num_cols < 1:
		raise ValueError('Tham số num_rows & num_cols phải >= 1 !')

	if type(num_rows) is not int or type(num_cols) is not int:
		raise ValueError('Tham số num_rows & num_cols phải là kiểu Int !')

	num_images = num_rows * num_cols
	plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
	for i in range(num_images):
		a = EvalofModelwithImage(images, pred_labels, true_labels, class_names, cmap)
		plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
		a.plot_image()
		plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
		a.plot_value_array()
	plt.tight_layout()
	if img_save_path != '':
		plt.savefig(img_save_path)
	plt.show()

def heatmap_plot(true_labels=None, pred_labels=None, class_names=None, categorical=True,
                 img_save_path='./eval_of_model_with_heatmap.jpg'):
	"""
	Trình tạo bản đồ nhiệt (Heatmap) để đánh giá mô hình
	:param true_labels: NdArray, chứa tập nhãn thực
	:param pred_labels: NdArray, chứa tập nhãn dự đoán bằng hàm model.predict()
	:param class_names: Tuple/List, chứa tập nhãn của tập dữ liệu
	:param categorical: True/False, xác định tập nhãn thực & nhãn dự đoán có Onehot Encode không ?
	:param img_save_path: Str, vị trí xuất ảnh đánh giá (Mặc định: Vị trí hiện tại)
	:: In bản đồ nhiệt lên màn hình
	"""
	if categorical:
		true_lb = np.argmax(true_labels, axis=1)
		pred_lb = np.argmax(pred_labels, axis=1)
	else:
		true_lb = true_labels
		pred_lb = pred_labels

	if len(class_names) <= 30:
		cm = confusion_matrix(true_lb, pred_lb)
		len_of_class_names = len(class_names)
		if len_of_class_names < 8:
			fig_width = 8
			fig_height = 8
		else:
			fig_width = int(len_of_class_names * .5)
			fig_height = int(len_of_class_names * .5)
		plt.figure(figsize=(fig_width, fig_height))
		sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=True)
		plt.xticks(np.arange(len_of_class_names) + .5, class_names)
		plt.yticks(np.arange(len_of_class_names) + .5, class_names)
		plt.xlabel('Nhãn dự đoán')
		plt.ylabel('Nhãn thực')
		plt.title('Đánh giá mô hình qua bản đồ nhiệt Heatmap')
		if img_save_path != '':
			plt.savefig(img_save_path)
		plt.show()

def EvalofTraining(history=None, img_save_path='./'):
	"""
	Trình đánh giá quá trình Training
	Args:
		history: Object/History object, chứa lịch sử Training do hàm model.fit() trả về
		img_save_path: Str, vị trí xuất ảnh đánh giá (Mặc định: Vị trí hiện tại)
	Returns:
		In ảnh đánh giá quá trình Training gồm Loss & Accuracy sau mỗi Epoch
	"""

	if type(history) not in (History, object):
		raise TypeError('Tham số history phải là một đối tượng History !')

	if not history:
		raise IndexError('Tham số history chứa mảng rỗng !')

	if not os.path.exists(img_save_path):
		os.mkdir(img_save_path)

	history = history.history
	epochs = len(history['loss'])
	len_of_epochs = range(1, epochs + 1)

	plt.title('Loss')
	plt.plot(len_of_epochs, history['loss'], 'ro', label='loss')
	plt.plot(len_of_epochs, history['val_loss'], 'b', label='val_loss')
	if max(len_of_epochs) <= 30:
		plt.xticks(len_of_epochs)
	plt.xlabel('Epochs')
	plt.ylabel('% (Percentage)')
	plt.legend()
	plt.grid(False if max(len_of_epochs) > 30 else True)
	if img_save_path != '':
		plt.savefig(os.path.join(img_save_path, 'loss.jpg'))
	plt.show()

	plt.title('Accuracy')
	plt.plot(len_of_epochs, history['accuracy'], 'ro', label='accuracy')
	plt.plot(len_of_epochs, history['val_accuracy'], 'b', label='val_accuracy')
	if max(len_of_epochs) <= 30:
		plt.xticks(len_of_epochs)
	plt.xlabel('Epochs')
	plt.ylabel('% (Percentage)')
	plt.legend()
	plt.grid(False if max(len_of_epochs) > 30 else True)
	if img_save_path != '':
		plt.savefig(os.path.join(img_save_path, 'accuracy.jpg'))
	plt.show()

import kaggle as kg


def kaggle_downloader(ds_name=None, ds_url=None, save_path=None):
    """
    Trình tải tập dữ liệu từ thư viện Kaggle
    Args:
            ds_name: Str, đặt tên cho tập dữ liệu
            ds_url: Str, chuỗi URL đến tập dữ liệu, định dạng: <owner>/<dataset-name>
            save_path: Str, đường dẫn lưu tập dữ liệu
    Return:
            Tập dữ liệu đã được tải & giải nén
    """

    ds_name = ds_name.lower()
    ds_url = ds_url.lower()

    if ds_name == '' or ds_name is None:
        raise ValueError('Tham số dataset_name không được để trống !')
        return

    if ds_url == '' or ds_url is None:
        raise ValueError('Tham số dataset_url không được để trống !')
        return

    if save_path == '' or save_path is None:
        raise ValueError('Tham số save_path không được để trống !')
        return

    try:
        # Xác thực đăng nhập
        kg.api.authenticate()
        # Tải dữ liệu
        kg.api.dataset_download_files(
            ds_url,
            save_path,
            unzip=True,
            quiet=False
        )
        print('=> Tải dữ liệu thành công !')
        print(f'=> Tập dữ liệu của bạn được lưu tại: {save_path}')
    except:
        raise Exception('Quá trình tải bị lỗi !')

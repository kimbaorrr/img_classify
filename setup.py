import setuptools

with open('README.rst', 'r') as fh:
    long_description = fh.read()
with open('requirements.txt', 'r') as fh:
    requirements = [line.strip() for line in fh.readlines()]

setuptools.setup(
    name='img_classify',
    version='1.0.0',
    author='Nguyễn Kim Bảo',
    author_email='nguyenkimbao.0708@gmail.com',
    description='Các công cụ dùng cho phân loại hình ảnh',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    packages=setuptools.find_packages(),
    python_requires='>=3.11, <3.13',
    install_requires=requirements,
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)

import os

abs_path = os.path.abspath(__file__)
file_dir = os.path.dirname(abs_path)
base_dir = file_dir = os.path.dirname(file_dir)
BASE_PATH = base_dir


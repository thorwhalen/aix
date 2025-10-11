"""Utils for AIX."""

import os
from importlib.resources import files
from functools import partial

from config2py import simple_config_getter, get_app_config_folder, process_path


pkg_name = 'aix'

get_config = simple_config_getter(pkg_name)

data_files = files(pkg_name) / "data"
templates_files = data_files / "templates"
app_data_dir = os.environ.get(
    f"{pkg_name.upper()}_APP_DATA_DIR", get_app_config_folder(pkg_name)
)
app_data_dir = process_path(app_data_dir, ensure_dir_exists=True)
djoin = partial(os.path.join, app_data_dir)
model_info_dir = process_path(djoin("model_info"), ensure_dir_exists=True)

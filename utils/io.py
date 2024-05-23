import os


class FileOpt:
  def __init__(self) -> None:
    pass

  @staticmethod
  def get_file_lines(file_path: str):
    with open(file_path, 'r') as f:
      return f.readlines()
  
  @staticmethod
  def write_file_lines(file_path: str, lines: list):
    with open(file_path, 'w') as f:
      f.writelines(lines)
  
  @classmethod
  def copy_file(cls, src: str, dst: str):
    lines = cls.get_file_lines(src)
    cls.write_file_lines(dst, lines)

  @classmethod
  def save_config(cls, src, des):
    from omegaconf import OmegaConf
    from time import gmtime, strftime
    os.makedirs(des, exist_ok=True)
    src_name = os.path.basename(src).split(".")[0]
    des_code = os.path.join(des, f"{src_name}.py")
    src_config = os.path.join(os.path.dirname(src), 'configs', f"{src_name}.yml")
    des_config = os.path.join(des, f"{src_name}.yml")
    src_dir = os.path.abspath(os.path.dirname(src))

    content = cls.get_file_lines(src)
    prefix = ['import sys\n', 
              f'sys.path.append(r"{src_dir}")\n',
              'config_dir = ""\n',
              f'datetime = "{strftime("%Y%m%d", gmtime())}"\n',] 
    content = prefix + content
    cls.write_file_lines(des_code, content)

    config = OmegaConf.load(src_config)
    config.others.train = False
    OmegaConf.save(config, des_config)
import numpy as np
import SimpleITK as sitk

from skimage.transform import radon, iradon
from tqdm import tqdm
from threading import Thread, Lock
from ..utils.img_process import crop_image, resize_image, \
  resample_image_by_spacing, get_suv_factor
  

PATH = 'path'
RAW = 'raw'
ITK = 'itk'
IMG = 'img'
NONAME = 'noname'

'''
images_dict =
  {
    '30G': {
      'path': [path1, path2, path3, ...],
      RAW: [raw1, raw2, raw3, ...], 
      ITK: [itk1, itk2, itk3, ...], # Simple ITK Image in numpy array
    },
    'CT': {
      'path': [path1, path2, path3, ...],
      RAW: [raw1, raw2, raw3, ...],
      ITK: [itk1, itk2, itk3, ...], # Simple ITK Image in numpy array
    },
    ...
  }
np_data = 
  {
    RAW: {
      '30G': [raw1, raw2, raw3, ...],
      'CT': [raw1, raw2, raw3, ...],  # numpy array
    }
    ITK: {
      '30G': [itk1, itk2, itk3, ...],
      'CT': [itk1, itk2, itk3, ...],  # numpy array
    }
  }
'''


class AbstractGeneralMI:
  def __init__(self, images_dict: dict, image_keys=None, label_keys=None,
               pid=None, process_param=None, img_type=None,
               raw_process=None, data_process=None, post_process=None):
    self.IMG_TYPE = 'nii.gz'  # image raw file type
    self.PRO_TYPE = 'pkl'  # metadata file type of image
    self.pid = pid

    self.raw_process = (lambda x: x) if raw_process is None else raw_process
    self.data_process = (lambda x: x) if data_process is None else data_process
    self.post_process = (lambda x: x) if post_process is None else post_process

    self._image_keys, self._label_keys = [], []
    self.images_dict = images_dict
    self.np_data = {RAW: {}, ITK: {}}
    
    self.custom_dict = {}
    self.image_keys = image_keys
    self.label_keys = label_keys

    self.img_type = img_type
    self.process_param = {
      'ct_window': None,
      'norm': None,  # only min-max,
      'shape': None,  # [320, 320, 240]
      'crop': None,  # [30, 30, 10]
      'clip': None,  # [1, None]
      'percent': None,  # 99.9
    } if process_param is None else process_param

  # region: basic function

  def __len__(self):
    return len(self.pid)

  def __getitem__(self, item):
    assert item is not np.int_
    pid = self.pid[item]

    img_dict = {}
    np_data = {RAW: {}, ITK: {}}
    for type_name in self.images_dict.keys():
      img_dict[type_name] = {}
      img_dict[type_name][PATH] = self.images_dict[type_name][PATH][item]
      keys = img_dict[type_name].keys()
      if RAW in keys:
        img_dict[type_name][RAW] = self.images_dict[type_name][RAW][item]
      if ITK in keys:
        img_dict[type_name][ITK] = self.images_dict[type_name][ITK][item]

      if self.np_data[RAW].get(type_name) is not None:
        np_data[RAW][type_name] = self.np_data[RAW][type_name][item]
      if self.np_data[ITK].get(type_name) is not None:
        np_data[ITK][type_name] = self.np_data[ITK][type_name][item]

    new_obj = self.__class__(img_dict, image_keys=self.image_keys,
                             label_keys=self.label_keys,
                             pid=pid, process_param=self.process_param,
                             img_type=self.img_type)
    new_obj.np_data = np_data
    return new_obj

  def index(self, pid):
    index = np.where(self.pid == pid)
    if len(index[0]) == 0:
      raise ValueError(f'pid {pid} not found')
    return int(index[0][0])

  def _remove(self, pid):
    for k, v in self.images_dict.items():
      self.images_dict[k][PATH] = np.delete(v[PATH], pid)
      if RAW in v.keys():
        self.images_dict[k][RAW] = np.delete(v[RAW], pid)
      if ITK in v.keys():
        self.images_dict[k][ITK] = np.delete(v[ITK], pid)
    for k, v in self.np_data[RAW].items():
      self.np_data[RAW][k] = np.delete(v, pid)
    for k, v in self.np_data[ITK].items():
      self.np_data[ITK][k] = np.delete(v, pid)
    self.pid = np.delete(self.pid, pid)

  def remove(self, pid):
    if isinstance(pid, (str, np.str_)):
      self.remove(self.index(pid))
    elif isinstance(pid, (int, np.int_)):
      self._remove(pid)
    elif isinstance(pid, (list, np.ndarray)):
      assert len(pid) != 0
      in_type = set([type(_) for _ in pid])
      assert len(in_type) == 1
      
      if in_type == {int}:
        self._remove(pid)
        return
      ids = []
      for p in pid:
        ids.append(self.index(p))
      self._remove(ids)

  def rm_void_data(self):
    use_keys = set(self.image_keys + [self.STD_key])
    rm_pids = []
    for i, pi in enumerate(self.pid):
      for key in use_keys:
        if self.images_dict[key][PATH][i] == '':
          rm_pids.append(pi)
          break
    self.remove(rm_pids)

  def clean_mem(self):
    self.init_image_dict(self.image_keys, reset=True)

  def pre_load(self, threads=4):
    if threads == 1:
      # todo: add progress bar
      assert False, 'not implemented'
      for key in self.image_keys:
        self.images[key][:]
      return
    
    total_num = len(self)
    if threads > total_num:
      threads = total_num
    slices = [list(range(i, total_num, threads)) for i in range(threads)]
    sum_num = len(slices) * (len(self.image_keys) - 1)

    # lock = Lock()

    with tqdm(total=sum_num, unit='threads', leave=False) as bar:
      t_list = []

      def load_imgdata(self, key, slices):
        for num in slices:
          path = self.images_dict[key][PATH][num]
          local_img = self.load_img(path)
          raw_img = self.raw_process(local_img, key, num)
          if key != self.STD_key:
            load_imgdata(self, self.STD_key, [num])
          data_img = self.data_process(raw_img, key, num)
          self.images_dict[key][RAW][num] = raw_img
          self.images_dict[key][ITK][num] = data_img
          self.np_data[RAW][key][num] = sitk.GetArrayViewFromImage(raw_img)
          self.np_data[ITK][key][num] = sitk.GetArrayFromImage(data_img)
        bar.update(1)

      for key in self.image_keys:
        if key == self.STD_key: continue
        for i in range(threads):
          t_list.append(Thread(target=load_imgdata, args=(self, key, slices[i])))
          t_list[-1].start()
      for t in t_list:
        t.join()
        
  def get_img_type(self, data_type):
    # todo: std_key?
    for key, item in self.img_type.items():
      if data_type in item:
        return key
    return 'Unknown'
  
  def init_image_dict(self, value, reset=False):
    for key in value:
      assert key in self.images_dict.keys()
      if key not in self.image_keys or reset:
        length = len(self.images_dict[key][PATH])
        if RAW not in self.images_dict[key].keys() or reset:
          self.images_dict[key][RAW] = np.array([None] * length)
          self.images_dict[key][ITK] = np.array([None] * length)
        self.np_data[RAW][key] = np.array([None] * length)
        self.np_data[ITK][key] = np.array([None] * length)
          
  @classmethod
  def init_data(cls, img_keys, csv_path, img_type, process_param):
    data = np.genfromtxt(csv_path, delimiter=',', dtype=str)
    types = data[0][1:]
    pid = data[1:, 0]
    path_array = data[1:, 1:]

    img_dict = {}
    for i, type_name in enumerate(types):
      img_path = path_array[:, i]
      img_dict[type_name] = {'path': img_path}

    mi_data = cls(img_dict,
                  image_keys=img_keys,
                  label_keys=[],
                  pid=pid, process_param=process_param, img_type=img_type)
    total_data = len(mi_data)
    mi_data.rm_void_data()
    print(f"Loaded data: {len(mi_data)}/{total_data}")

    return mi_data

  # endregion: basic function

  # region: special properties

  @property
  def images(self):
    # return numpy.ndarray type img
    return self.np_data[ITK]
  
  @property
  def raw_images(self):
    # return numpy.ndarray type img without process
    return self.np_data[RAW]

  @property
  def itk_imgs(self):
    # return sitk.Image type img
    itk_dict = {}
    for key in self.image_keys:
      itk_dict[key] = self.images_dict[key][ITK]
    return itk_dict

  @property
  def itk_raws(self):
    # return sitk.Image type img without process
    itk_dict = {}
    for key in self.image_keys:
      itk_dict[key] = self.images_dict[key][RAW]  
    return itk_dict

  @property
  def image_keys(self):
    return self._image_keys

  @image_keys.setter
  def image_keys(self, value):
    if value is None:
      return
    self.init_image_dict(value)
    self._image_keys = value

  @property
  def STD_key(self):
    return self.img_type['STD'][0]

  # endregion: special properties

  # region: static functions

  @staticmethod
  def load_img(filepath, array=False):
    assert isinstance(filepath, str)
    if not array:
      return sitk.ReadImage(filepath)
    else:
      return sitk.GetArrayViewFromImage(sitk.ReadImage(filepath)).astype(np.float32)

  @staticmethod
  def percentile(img, percent):
    arr = sitk.GetArrayFromImage(img)
    p = np.percentile(arr, percent)
    indices = arr >= p
    arr[indices] = 0
    arr[indices] = np.max(arr)
    modified_image = sitk.GetImageFromArray(arr)
    modified_image.CopyInformation(img)
    return modified_image

  @staticmethod
  def crop_by_margin(img, margins):
    ori_size = img.GetSize()
    start_index = [i for i in margins]
    size = [size - 2 * margin for size, margin in zip(ori_size, margins)]
    new_img = sitk.RegionOfInterest(img, size, start_index)
    return new_img

  @staticmethod
  def write_img(img, filepath, refer_img=None):
    assert isinstance(filepath, str)
    if not isinstance(img, sitk.Image):
      img = sitk.GetImageFromArray(img)
    if refer_img is not None:
      img.SetOrigin(refer_img.GetOrigin())
      img.SetDirection(refer_img.GetDirection())
      img.SetSpacing(refer_img.GetSpacing())
    sitk.WriteImage(img, filepath)

  @staticmethod
  def mask2onehot(seg, labels: list):
    onehot = np.zeros_like(seg, dtype=bool)
    onehot[np.isin(seg, labels)] = True
    return onehot

  @staticmethod
  def radon_transform(data, theta=None, circle=True,
                      preserve_range=True):
    if theta is None:
      theta = np.linspace(0., 180., max(data.shape), endpoint=False)
    sinogram = radon(data, theta=theta, circle=circle,
                     preserve_range=preserve_range)

    return sinogram

  @staticmethod
  def radon_reverse(sinogram, theta=None, circle=True,
                    filter_name='cosine', preserve_range=True,
                    **kwargs):
    if theta is None:
      theta = np.linspace(0., 180., max(sinogram.shape), endpoint=False)
    data = iradon(sinogram, theta=theta, circle=circle,
                  filter_name=filter_name, preserve_range=preserve_range,
                  **kwargs)

    return data

  @staticmethod
  def gaussian_filter(img, sigma):
    return sitk.SmoothingRecursiveGaussian(img, sigma)

  # endregion: static functions


class GeneralMI(AbstractGeneralMI):
  """
  a general data framework for pet/ct
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args,
                     raw_process=self.pre_process,
                     data_process=self.process,
                     post_process=self.post_process,
                     **kwargs)
  # region: data process

  def pre_process(self, img, data_type, item):
    new_img = img
    img_type = self.get_img_type(data_type)
    if img_type == 'MASK':
      resample_method = sitk.sitkNearestNeighbor
    else:
      resample_method = sitk.sitkLinear

    new_img = resample_image_by_spacing(new_img, (1.0, 1.0, 1.0),
                                        resample_method)
    if img_type == 'PET':
      new_img = self.suv_transform(new_img, self.get_tags(item))
    elif img_type == 'CT':
      if self.process_param.get('ct_window'):
        wc = self.process_param['ct_window'][0]
        wl = self.process_param['ct_window'][1]
        new_img = sitk.IntensityWindowing(new_img, wc - wl/2, wc + wl/2, 0, 255)
      # else:
        # new_img = sitk.RescaleIntensity(new_img, 0, 255)
    # if img_type != 'PET' and img_type != 'MASK':
    #   std_type = self.STD_key
    #   new_img = resize_image(new_img, self.images_dict[std_type][ITK][item],
    #                          resamplemethod=resample_method)
    if self.process_param.get('crop'):
      new_img = GeneralMI.crop_by_margin(new_img, self.process_param['crop'])
    if self.process_param.get('shape'):
      new_img = crop_image(new_img, self.process_param['shape'])
    if self.process_param.get('clip'):
      from copy import deepcopy
      clip = deepcopy(self.process_param['clip'])
      min_p = np.min(sitk.GetArrayViewFromImage(new_img))
      max_p = np.max(sitk.GetArrayViewFromImage(new_img))
      if clip[0] is None:
        clip[0] = min_p
      elif clip[1] is None:
        clip[1] = max_p
      clip[0] = float(max(clip[0], min_p))
      clip[1] = float(min(clip[1], max_p))
      new_img = sitk.IntensityWindowing(new_img, clip[0], clip[1],
                                        clip[0], clip[1])
    if self.process_param.get('percent'):
      new_img = self.percentile(new_img, self.process_param['percent'])
    return new_img

  def process(self, img, data_type, item):
    new_img: sitk.Image = img
    img_type = self.get_img_type(data_type)
    if 'MASK' == img_type:
      pass
    elif 'CT' == img_type:
      if self.process_param.get('norm'):
        new_img = sitk.RescaleIntensity(new_img, 0.0, 1.0)
    elif 'PET' == img_type:
      if self.process_param.get('norm'):
        if data_type == self.STD_key:
          new_img = sitk.RescaleIntensity(new_img, 0.0, 1.0)
        else:
          new_img = self.normalize(new_img, self.process_param['norm'],
                                   self.itk_raws[self.STD_key][item])

    return new_img

  def post_process(self):
    pass

  def reverse_norm_suv(self, img, item):
    return img * np.max(self.raw_images[self.STD_key][item])

  def get_tags(self, item):
    from joblib import load
    filepath = self.images_dict[self.STD_key][PATH][item]\
      .replace(self.IMG_TYPE, self.PRO_TYPE)
    return load(filepath)

  # endregion: data process

  # region: static functions

  @staticmethod
  def suv_transform(img, tag):
    suv_factor, _, _ = get_suv_factor(tag)
    return sitk.Multiply(img, suv_factor)

  @staticmethod
  def suv_reverse(img, tag):
    suv_factor, _, _ = get_suv_factor(tag)
    if not isinstance(img, sitk.Image):
      img = sitk.GetImageFromArray(img.astype('int16'))
    return sitk.DivideReal(img, suv_factor)

  @staticmethod
  def normalize(img, type, refer_pet=None):
    if type == 'min-max':
      img = sitk.RescaleIntensity(img, 0.0, 1.0)
    elif type == 'PET':
      assert refer_pet is not None
      refer_max = np.max(sitk.GetArrayViewFromImage(refer_pet))
      img = sitk.DivideReal(img, float(refer_max))
    return img

  # endregion: static functions

  # region: test functions

  def get_stat(self):
    stat_dict = {
      'sex': [],
      'weight': [],
      'age': [],
      'dose': []
    }
    for i in range(len(self.pid)):
      tag = self.get_tags(i)
      stat_dict['sex'].append(tag['PatientSex'])
      stat_dict['weight'].append(int(tag['PatientWeight']))
      stat_dict['age'].append(int(tag['PatientAge'][:-1]))
      stat_dict['dose'].append(int(tag['RadiopharmaceuticalInformationSequence'][0]
                      ['RadionuclideTotalDose'].value//1000000))
    return stat_dict

  @classmethod
  def get_test_sample(cls, csv_path, img_keys=None):

    img_type = {
      'CT': ['CT'],
      'PET': ['30G', '20S', '40S',
              '60G-1', '60G-2', '60G-3',
              '90G', '120S', '120G', '240G', '240S'],
      'MASK': ['CT_seg'],
      'STD': ['30G'],
    }

    process_param = {
      'ct_window': None,
      'norm': 'PET',  # only min-max,
      'shape': [440, 440, 560],  # [320, 320, 240]
      'crop': [0, 0, 10],  # [30, 30, 10]
      'clip': None,  # [1, None]
    }

    image_keys = img_keys if img_keys else ['30G', 'CT', '240S']

    test = cls.init_data(image_keys, csv_path, img_type, process_param)
    return test

  # endregion: test functions









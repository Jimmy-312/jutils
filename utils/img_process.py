import numpy as np
import SimpleITK as sitk


# region: suv calc

def dicom_time(t):
  t = str(int(float(t)))
  if len(t) == 5:
    t = '0' + t
  h_t = float(t[0:2])
  m_t = float(t[2:4])
  s_t = float(t[4:6])
  return h_t * 3600 + m_t * 60 + s_t


def get_suv_factor(tags):
  ST = tags['SeriesTime']
  RIS = tags['RadiopharmaceuticalInformationSequence'][0]
  RST = str(RIS['RadiopharmaceuticalStartTime'].value)
  RTD = str(RIS['RadionuclideTotalDose'].value)
  RHL = str(RIS['RadionuclideHalfLife'].value)
  PW = tags['PatientWeight']
  RS = tags['RescaleSlope']
  RI = tags['RescaleIntercept']

  decay_time = dicom_time(ST) - dicom_time(RST)
  decay_dose = float(RTD) * pow(2, -float(decay_time) / float(RHL))
  SUVbwScaleFactor = (1000 * float(PW)) / decay_dose

  return SUVbwScaleFactor, RS, RI

# endregion: suv calc

# region: image process

def crop_image(image, target_size):
  original_size = image.GetSize()

  # 计算填充或裁剪的大小差异
  size_diff = [target_size[i] - original_size[i] for i in range(3)]

  # 计算填充或裁剪的边界值
  lower_crop = [-int(size_diff[i] / 2) if size_diff[i] < 0 else 0 for i in range(3)]
  upper_crop = [-size_diff[i] - lower_crop[i] if size_diff[i] < 0 else 0 for i in range(3)]

  lower_pad = [int(size_diff[i] / 2) if size_diff[i] > 0 else 0 for i in range(3)]
  upper_pad = [size_diff[i] - lower_pad[i] if size_diff[i] > 0 else 0 for i in range(3)]

  # 对图像进行对称填充
  image = sitk.ConstantPad(image, lower_pad, upper_pad)

  # 对图像进行裁剪
  image = sitk.Crop(image, lower_crop, upper_crop)

  return image


def resize_image(ori_img, target_img=None,
                     size=None, spacing=None, origin=None, direction=None,
                     resamplemethod=sitk.sitkLinear, raw=True):
  """
  用itk方法将原始图像resample到与目标图像一致
  :param ori_img: 原始需要对齐的itk图像
  :param target_img: 要对齐的目标itk图像
  :param resamplemethod: itk插值方法: sitk.sitkLinear-线性  sitk.sitkNearestNeighbor-最近邻
  :return:img_res_itk: 重采样好的itk图像
  使用示范：
  import SimpleITK as sitk
  target_img = sitk.ReadImage(target_img_file)
  ori_img = sitk.ReadImage(ori_img_file)
  img_r = resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkLinear)
  """
  if target_img is not None:
    size = target_img.GetSize()  # 目标图像大小  [x,y,z]
    spacing = target_img.GetSpacing()  # 目标的体素块尺寸    [x,y,z]
    origin = target_img.GetOrigin()  # 目标的起点 [x,y,z]
    direction = target_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]
  else:
    size = size if size is not None else ori_img.GetSize()
    spacing = spacing if spacing is not None else ori_img.GetSpacing()
    origin = origin if origin is not None else ori_img.GetOrigin()
    direction = direction if direction is not None else ori_img.GetDirection()

  if size == ori_img.GetSize():
    if raw:
      return ori_img
    else:
      return sitk.GetArrayFromImage(ori_img)
  # itk的方法进行resample
  resampler = sitk.ResampleImageFilter()
  resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
  # 设置目标图像的信息
  resampler.SetSize(size)  # 目标图像大小
  resampler.SetOutputOrigin(origin)
  resampler.SetOutputDirection(direction)
  resampler.SetOutputSpacing(spacing)
  # 根据需要重采样图像的情况设置不同的type
  if resamplemethod == sitk.sitkNearestNeighbor:
    resampler.SetOutputPixelType(sitk.sitkUInt8)  # 近邻插值用于mask的，保存uint8
  else:
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    # 线性插值用于PET/CT/MRI之类的，保存float32
  resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
  resampler.SetInterpolator(resamplemethod)
  itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
  if raw:
    return itk_img_resampled
  else:
    return sitk.GetArrayFromImage(itk_img_resampled)


def resample_image_by_spacing(image, new_spacing, method=sitk.sitkLinear):
  original_spacing = image.GetSpacing()
  original_size = image.GetSize()

  # 计算缩放因子
  spacing_ratio = [osp / nsp for osp, nsp in
                   zip(original_spacing, new_spacing)]
  new_size = [int(osz * sr) for osz, sr in zip(original_size, spacing_ratio)]

  # 创建一个Resample滤波器
  resample = sitk.ResampleImageFilter()
  resample.SetOutputSpacing(new_spacing)
  resample.SetSize(new_size)
  resample.SetOutputOrigin(image.GetOrigin())
  resample.SetOutputDirection(image.GetDirection())

  if method == sitk.sitkNearestNeighbor:
    resample.SetOutputPixelType(sitk.sitkUInt8)  # 近邻插值用于mask的，保存uint8
  else:
    resample.SetOutputPixelType(sitk.sitkFloat32)
  resample.SetInterpolator(method)
  # 执行重采样
  resampled_image = resample.Execute(image)

  return resampled_image

# endregion: image process

# region: image sample

def windows_choose(distr: np.ndarray, windows_size):
  assert np.NaN not in distr
  x = np.linspace(0, distr.shape[0] - 1, distr.shape[0])
  result = np.random.choice(x, p=distr)
  result = result - windows_size / 2

  if result < 0: result = 0
  if result > distr.shape[0] - windows_size:
    result = distr.shape[0] - windows_size

  return int(result)


def get_random_window(arr: np.ndarray, windows_size, true_rand=False):
  # todo: true random
  
  def normalize(arr: np.array):
    norm = np.linalg.norm(arr, ord=1)
    return arr / norm
  
  assert len(arr.shape) == len(windows_size)
  # todo: the input array should not have the channel dimension

  arr = arr != 0
  # arr = arr[:, ..., 0]

  dimension = len(arr.shape)
  sub_arr = []
  for i in range(dimension):
    sub_arr.append(np.any(arr, axis=tuple([j for j in range(dimension) if j != i])))

  dist_list = []
  for s_arr in sub_arr:
    dist_list.append(normalize(s_arr.ravel()))

  pos = []
  for i, dist in enumerate(dist_list):
    pos.append(windows_choose(dist, windows_size[i]))

  return pos


def get_sample(arr: np.ndarray, pos, windows_size):
  assert len(pos) == len(windows_size)

  for i, p, s in zip(range(len(pos)), pos, windows_size):
    arr = np.take(arr, range(p, p + s), axis=i)
  return arr


def gen_windows_pair(arr1: np.ndarray, arr2: np.ndarray, batch_size,
                windows_size, **kwargs):
  features = []
  targets = []
  index = np.random.choice(range(arr1.shape[0]), batch_size)
  arr1, arr2 = arr1[index], arr2[index]

  for i in range(batch_size):
    pos = get_random_window(arr1[i], windows_size, **kwargs)
    features.append(get_sample(arr1[i], pos, windows_size))
    targets.append(get_sample(arr2[i], pos, windows_size))
  features = np.stack(features)
  targets = np.stack(targets)

  return features, targets

def gen_windows(arr: np.ndarray, windows_size, **kwargs):
  pos = get_random_window(arr, windows_size, **kwargs)
  arr_crop = get_sample(arr, pos, windows_size)

  return arr_crop

# endregion: image sample
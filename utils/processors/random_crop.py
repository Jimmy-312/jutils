import numpy as np


class RandomCrop:
  def __init__(self, windows_size, arr=None, true_rand=False):
    self.windows_size = tuple(windows_size)
    if windows_size is None:
      return
    self.true_rand = true_rand
    if arr:
      self.gen_pos(arr)

  def __call__(self, arr):  
    return self.get_sample(arr, self.pos, self.windows_size)
  
  def gen_pos(self, arr):
    self.pos = self.get_random_window(arr, self.windows_size)

  def get_random_window(self, arr: np.ndarray, windows_size):  
    def normalize(arr: np.array):
      norm = np.linalg.norm(arr, ord=1)
      return arr / norm
    
    arr = np.squeeze(arr)
    assert len(arr.shape) == len(windows_size)
    # did: the input array should not have the channel dimension
    
    pos = []
    # did: true random
    if self.true_rand:
      for i, size in enumerate(arr.shape):
        index = np.random.randint(size)
        pos.append(self.windows_choose(index, 
                                       windows_size[i], size))
      return pos

    arr = arr != 0
    dimension = len(arr.shape)  
    sub_arr = []
    for i in range(dimension):
      sub_arr.append(np.any(arr, axis=tuple([j for j in range(dimension) if j != i])))

    dist_list = []
    for s_arr in sub_arr:
      dist_list.append(normalize(s_arr.ravel()))
    
    for i, dist in enumerate(dist_list):
      pos.append(self.windows_choose_by_distr(dist, windows_size[i]))

    return pos
  
  def windows_choose_by_distr(self, distr: np.ndarray, windows_size):
    assert np.NaN not in distr
    x = np.linspace(0, distr.shape[0] - 1, distr.shape[0])
    result = np.random.choice(x, p=distr)
    return self.windows_choose(result, windows_size, distr.shape[0])

  def mult_gen(self, arr_list, refer_arr):
    num = arr_list[0].shape[0]
    result = [np.zeros((num, ) + self.windows_size) for _ in arr_list]
    for i in range(num):
      self.gen_pos(refer_arr[i])
      for j, arr in enumerate(arr_list):
        result[j][i] = self(arr[i, ..., 0])
    result = [np.expand_dims(_, axis=-1) for _ in result]
    return result

  @staticmethod
  def get_sample(arr: np.ndarray, pos, windows_size):
    assert len(pos) == len(windows_size)
    # supported [..., h, w] with windows [h, w]
    indices = len(arr.shape) - len(pos)
    assert indices >= 0

    for i, p, s in zip(range(len(pos)), pos, windows_size):
      arr = np.take(arr, range(p, p + s), axis=i+indices)
    return arr
  
  @staticmethod
  def windows_choose(pos, windows_size, max_size):
    result = pos - windows_size / 2

    if result < 0: result = 0
    if result > max_size - windows_size:
      result = max_size - windows_size

    return int(result)
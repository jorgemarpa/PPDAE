import glob
import numpy as np
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from bisect import bisect
from sklearn import preprocessing

root = "/Users/jorgetil/Astro/PPD-AE"
colab_root = "/content/drive/MyDrive"
exalearn_root = "/home/jorgemarpa/data/imgs"


class MyRotationTransform:
    """Rotate by a random N times 90 deg."""

    def __init__(self):
        pass

    def __call__(self, x):
        shape = x.shape
        return np.rot90(x, np.random.choice([0, 1, 2, 3]), axes=[-2, -1]).copy()


class MyFlipVerticalTransform:
    """Random vertical flip."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, x):
        if np.random.uniform() >= self.prob:
            return np.flip(x, -2).copy()
        else:
            return x

class MyZeroPadding:
  """Adds zero paddin to images, when needed."""
  def __init__(self):
      pass
      
  def __call__(self, x):
    MASTER_DIM = 187
    dims = x.shape[-1]
    if (dims < MASTER_DIM):
      desired_width, desired_heigh = ((MASTER_DIM - x.shape[-1])//2, (MASTER_DIM - x.shape[-1])//2)
      padding = (desired_width, desired_width, 
                desired_heigh, desired_heigh)
      return pad(torch.tensor(x), padding, value=0.0).numpy()
    else: return x


class MyNormTransform:
    """Normalization."""

    def __init__(self, mean=0.0, std=1.0):
        self.mean = np.array(mean)
        self.std = np.array(std)
        if self.mean.ndim == 1:
            self.mean = self.mean[None, :, None, None]
        if self.std.ndim == 1:
            self.std = self.std[None, :, None, None]

    def __call__(self, x):
        return (x - self.mean) / self.std


# load pkl synthetic light-curve files to numpy array
class ProtoPlanetaryDisks(Dataset):
    """
    Dataset class that loads synthetic images of Protoplanetary disks,
    the dataset has shape [N, C, H, W] = [36518, 1, 187, 187]
    ...
    Attributes
    ----------
    imgs        : array
        array with images
    meta        : array
        array with physical parameters asociated to each image
    meta_names  : list
        list with the names of the physical parameters (8 columns)
        m_dust = 'mass of the dust'
        Rc     = 'critical radius when exp drops(size)'
        f_exp  = 'flare exponent'
        H0     = 'scale hight'
        Rin    = 'inner raidus'
        sd_exp = 'surface density exponent'
        alpha  = 'dust stettling'
        inc    = 'inclination'
    img_dim     : int
        image dimension, assuming square ratio
    img_channel : int
        number of channels per image
    transform   : bool
        apply rotation and flip transformation
    transform_fx : torchvision transformations
        set of transformations to be applyed when calling an item
    Methods
    -------
    __getitem__(self, index)
        return data in the index position, apply transform_fx if transform
        is True
    __len__(self)
        return the total length of the entire dataset
    get_dataloader(self, batch_size=32, shuffle=True,
                   test_split=0.2, random_seed=42)
        return a dataloader object for trainning and testing
    """

    def __init__(
        self,
        machine="colab",
        transform=True,
        par_norm=False,
        image_norm="global",
        nchannels=1,
        𝜆=['600nm', '870um']
    ):
        """
        Parameters
        ----------
        machine    : bool, optional
            which machine is been used (colab, exalearn, [local])
        transform  : bool, optional
            if apply or not image transformation when getting new item
        par_norm   : bool, optional
            load parameters that are scaled to [0,1] when True, or raw images
            when False.
        nchannels  :  int, optional
            number of channels to use. If 1 then user must spicify which wavelength
            to use inside the array.
        𝜆          : string array, required
            Specifies which wavelenght to use ([600nm], 870um). Important to extract
            the right data from the right directory. NOTE: the directory in the first position
            must be the main directory.
        """
        MASTER_DIM = 187
        if machine == "local":
            ppd_path = "%s/data/PPD/partitions" % (root)
        elif machine == "colab":
            ppd_path = ['%s/data/PPDAE/partitions/%s' % (colab_root, L) for L in 𝜆]
        elif machine == "exalearn":
            ppd_path = "%s/PPD/partitions" % (exalearn_root)
        else:
            raise ("Wrong host, please select local, colab or exalearn")

        self.par_train = np.load(
            "%s/param_arr_gridandfiller_train_all.npy" % (ppd_path[0])
        ) # we assume all batches were split using the same random seed

        self.imgs_paths = []
        self.imgs_memmaps = []
        self.start_indices = []
        self.data_count = []

        for i in range(nchannels):

          ims = sorted(
            glob.glob(
                "%s/img_array_gridandfiller_%snorm_train_*.npy"
                % (ppd_path[i], image_norm)
                ))
          self.imgs_paths.append(ims)
          self.imgs_memmaps.append([np.load(path, mmap_mode="r") for path in self.imgs_paths[i]])
          self.start_indices.append([0] * len(self.imgs_paths[i]))
          self.data_count.append(0)
        
          for index, memmap in enumerate(self.imgs_memmaps[i]):
              self.start_indices[i][index] = self.data_count[i]
              self.data_count[i] += memmap.shape[0]

        self.par_names = [
            "m_dust",
            "Rc",
            "f_exp",
            "H0",
            "Rin",
            "sd_exp",
            "alpha",
            "inc",
        ]

        self.par_test = np.load(
              "%s/param_arr_gridandfiller_test.npy" % (ppd_path[0]) 
          ) # we use master
        
        if nchannels == 1:
          self.imgs_test = np.load(
              "%s/img_array_gridandfiller_%snorm_test.npy"
              % (ppd_path[0], image_norm)
          )
        else:
          self.imgs_test = np.empty([len(self.par_test), nchannels, MASTER_DIM, MASTER_DIM])

          for i in range(nchannels):
            im = np.load(
                "%s/img_array_gridandfiller_%snorm_test.npy"
                % (ppd_path[i], image_norm)
            )

            if im.shape[-1] < MASTER_DIM:
              mzp = MyZeroPadding()
              im = mzp(im)
            self.imgs_test[:, i] = im[:len(self.par_test), 0, :, :] # temp, remove len(self.par_test) after data matches

        # after transformation, all images will have the same properties
        self.img_dim = MASTER_DIM if nchannels > 1 else self.imgs_test[0].shape[-1]
        self.img_channels = nchannels
        self.transform = transform

        self.transform_fx = torchvision.transforms.Compose(
            [MyRotationTransform(), 
             MyFlipVerticalTransform(), 
             MyZeroPadding()] if nchannels > 1 else [MyRotationTransform(),
                                                     MyFlipVerticalTransform()]
        )
        self.par_norm = par_norm
        self.MinMaxSc = preprocessing.MinMaxScaler()
        self.MinMaxSc.fit(np.concatenate([self.par_train, self.par_test]))

    def __len__(self):
        return len(self.par_train) + len(self.par_test)

    def __getitem__(self, index):
      imgs = np.empty([self.img_channels, self.img_dim, self.img_dim])
      for i in range(self.img_channels):
        memmap_index = bisect(self.start_indices[i], index) - 1
        index_in_memmap = index - self.start_indices[i][memmap_index]
        img = self.imgs_memmaps[i][memmap_index][index_in_memmap]

        if self.transform:
            img = self.transform_fx(img)
        imgs[i] = img[0]
      
      par = self.par_train[index]
      if self.par_norm:
          par = self.MinMaxSc.transform(par.reshape(1, -1))[0]

      return imgs.astype(np.float32), par

    def get_dataloader(
        self, batch_size=32, shuffle=True, val_split=0.2, random_seed=42
    ):
        """
        Parameters
        ----------
        batch_size : int
            size of each batch
        shuffle    : bool
            whether to shuffle or not the samples
        val_split : float
            fraction of the dataset to be used as validation sample
        random_seed: int
            initialization of random seed
        Returns
        -------
        train_loader :
            dataset loader with training instances
        val_loader  : 
            dataset loader with validation instances
        test_loader  :
            dataset loader with testing instances
        """
        np.random.seed(random_seed)
        if val_split == 0.0:
            train_loader = DataLoader(
                self, batch_size=batch_size, shuffle=shuffle, drop_last=False
            )
            val_loader = None
        else:
            # Creating data indices for training and val splits:
            dataset_size = len(self.par_train)
            indices = list(range(dataset_size))
            split = int(np.floor(val_split * dataset_size))
            if shuffle:
                np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]
            del indices, split

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)

            train_loader = DataLoader(
                self, batch_size=batch_size, sampler=train_sampler, drop_last=False
            )
            val_loader = DataLoader(
                self, batch_size=batch_size, sampler=val_sampler, drop_last=False
            )

        if self.par_norm:
            aux_par_test = self.MinMaxSc.transform(self.par_test)
        else:
            aux_par_test = self.par_test
        test_ds = TensorDataset(
            torch.Tensor(self.imgs_test), torch.Tensor(aux_par_test)
        )
        del aux_par_test
        test_loader = DataLoader(test_ds, batch_size=batch_size, drop_last=False)

        return train_loader, val_loader, test_loader

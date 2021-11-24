from PIL import Image
from glob import glob
import os
from torch.utils.data import Dataset
import math
import numpy as np
import random
from .imagenet_dic import IMAGENET_DIC

def get_imagenet_dataset(data_root, config, class_num=None, random_crop=True, random_flip=False):
    train_dataset = IMAGENET_dataset(data_root, mode='train', class_num=class_num, img_size=config.data.image_size,
                                     random_crop=random_crop, random_flip=random_flip)
    test_dataset = IMAGENET_dataset(data_root, mode='val', class_num=class_num, img_size=config.data.image_size,
                                    random_crop=random_crop, random_flip=random_flip)

    return train_dataset, test_dataset


###################################################################


class IMAGENET_dataset(Dataset):
    def __init__(self, image_root, mode='val', class_num=None, img_size=512, random_crop=True, random_flip=False):
        super().__init__()
        if class_num is not None:
            self.data_dir = os.path.join(image_root, mode, IMAGENET_DIC[str(class_num)][0], '*.JPEG')
            self.image_paths = sorted(glob(self.data_dir))
        else:
            self.data_dir = os.path.join(image_root, mode, '*', '*.JPEG')
            self.image_paths = sorted(glob(self.data_dir))
        self.img_size = img_size
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.class_num = class_num

    def __getitem__(self, index):
        f = self.image_paths[index]
        pil_image = Image.open(f)
        pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.img_size)
        else:
            arr = center_crop_arr(pil_image, self.img_size)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        # y = [self.class_num, IMAGENET_DIC[str(self.class_num)][0], IMAGENET_DIC[str(self.class_num)][1]]
        # y = self.class_num

        return np.transpose(arr, [2, 0, 1])#, y

    def __len__(self):
        return len(self.image_paths)


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

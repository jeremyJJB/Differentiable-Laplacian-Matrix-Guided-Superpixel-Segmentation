"""
This code mainly comes from SCN: https://github.com/fuy34/superpixel_fcn

Other code is also used and links provided to SSM. The other baselines all use the SCN code.
"""
from torch.utils import data as data
import numpy as np
import torch
from pathlib import Path
from skimage import io
import scipy
import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning as L
from torch.utils.data import DataLoader, Dataset
import cv2
from einops import rearrange
from skimage.segmentation import find_boundaries
import torch.nn.functional as F
import random
from albumentations.core.transforms_interface import ImageOnlyTransform
from eval_spixel import ssm_flow_transforms


class CustomPhotometricDistort(ImageOnlyTransform):
    """albumentations wrapper for  PhotometricDistort. This is only used for SSM"""

    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.photometric_distort = ssm_flow_transforms.PhotometricDistort()

    def apply(self, img, **params):
        img = self.photometric_distort(img)
        return img

    def get_transform_init_args_names(self):
        return ()


def convert_label(label):
    """
    This code is from SCN: https://github.com/fuy34/superpixel_fcn/blob/master/data_preprocessing/pre_process_bsd500.py#L64

    This is used across all baselines to preprocess the dataset labels so that they only have 50 classes

    :param label:
    :return:
    """
    problabel = np.zeros((label.shape[0], label.shape[1], 50)).astype(np.float32)
    ct = 0
    for t in np.unique(label).tolist():
        if ct >= 50:
            print('give up sample because label shape is larger than 50: {0}'.format(np.unique(label).shape))
            break
        else:
            problabel[:, :, ct] = (label == t)  # one hot
        ct = ct + 1

    label2 = np.squeeze(np.argmax(problabel, axis=-1))  # squashed label e.g. [1. 3. 9, 10] --> [0,1,2,3], (h*w)
    return label2, problabel


def get_labels_more(label):
    """
    This is a helper function to get the number of classes for the image. This is not used for data preprocessing but rather
    statistics on the og dataset.
    :param label:
    :return:
    """
    ct = 0
    for t in np.unique(label).tolist():
        if ct >= 50:
            print('give up sample because label shape is larger than 50: {0}'.format(np.unique(label).shape))
            morethan50 = 1
            break
        else:
            morethan50 = 0
        ct = ct + 1
    num_classes = np.unique(label).shape[0]

    return num_classes, morethan50


def see_if_split_match(split):
    """
    Helper function to ensure the images in the created split match the images in SCN.

    Here are where the lists for the different splits are: https://github.com/fuy34/superpixel_fcn/tree/master/data_preprocessing

    :param split:
    :return:
    """

    image_dir = Path("/path/to/images/BSD500/", split)
    filt_scn_split = Path("/path/to/file lists/BSD500/", split[:-1] + '.txt')

    our_imgs = list(image_dir.glob('*.npy'))  # clean up images
    our_imgs_clean = [int(img.stem.split('_')[0]) for img in our_imgs]
    final_our_imgs = set(our_imgs_clean) # we are using sets since we have multiple masks per each image, want to ensure
    # each image is present

    # Their images
    scn_images = set(np.loadtxt(filt_scn_split, dtype=int))
    only_in_us = final_our_imgs - scn_images
    only_in_them = scn_images - final_our_imgs

    print("Elements only for us:", only_in_us)
    print("Elements only for them:", only_in_them)  # imgs that they have and we dont have


def make_train_transform(train_size: int = -1, model_name: str = None):
    if model_name == "ssm":
        """
        https://github.com/jiaxhm/SSMamba/blob/main/main.py#L119
        """
        train_transform = A.Compose(
            [
                A.RandomCrop(height=train_size, width=train_size, pad_if_needed=False, p=1.0),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                CustomPhotometricDistort(),
                A.Normalize(
                    mean=(0.411, 0.432, 0.45),
                    std=(1, 1, 1)),
                ToTensorV2(),
            ],
            additional_targets={'edge': 'mask'}
        )
    else:
        # all other models had the following transform: https://github.com/fuy34/superpixel_fcn/blob/master/main.py#L113
        train_transform = A.Compose(
            [
                A.RandomCrop(height=train_size, width=train_size, pad_if_needed=False, p=1.0),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=(0.411, 0.432, 0.45),
                    std=(1, 1, 1)),
                ToTensorV2(),
            ]
        )
    return train_transform


def make_val_transform(val_size: int = -1, model_name: str = None):
    """
    The 320 comes from SCN https://github.com/fuy34/superpixel_fcn/blob/master/datasets/BSD500.py#L69 and other models followed suit
    :return:
    """
    if model_name == "ssm":
        # just need the additional targets
        val_transform = A.Compose(
            [
                A.CenterCrop(height=val_size, width=val_size, pad_if_needed=False, p=1.0),
                A.Normalize(
                    mean=(0.411, 0.432, 0.45),
                    std=(1, 1, 1)),
                ToTensorV2(),
            ],
            additional_targets={'edge': 'mask'}
        )
    else:
        val_transform = A.Compose(
            [
                A.CenterCrop(height=val_size, width=val_size, pad_if_needed=False, p=1.0),
                A.Normalize(
                    mean=(0.411, 0.432, 0.45),
                    std=(1, 1, 1)),
                ToTensorV2(),
            ]
        )
    return val_transform


def data_preprocess_test():
    """
    Separate function for the test split because we need to save the csv for the benchmarking library used later. Also
    use the jpgs for inference
    :return:
    """
    image_dir = Path("/path/to/BSD500/images_og", "test")
    gt_dir = Path("/path/to/BSD500/ground_truth_og", "test")
    final_image_dir = Path("/path/to/BSD500/images_preprocess", "test")
    final_gt_dir = Path("/path/to/BSD500/ground_truth_preprocess", "test")

    if not final_image_dir.exists():
        final_image_dir.mkdir(parents=True)
    if not final_gt_dir.exists():
        final_gt_dir.mkdir(parents=True)

    gt_masks = gt_dir.glob('*.mat')

    for mask_fp in gt_masks:
        img_ = io.imread(image_dir / (mask_fp.stem + ".jpg"))

        img_cv2 = cv2.imread(image_dir / (mask_fp.stem + ".jpg"))

        cv2.imwrite(str(mask_fp.stem + "custom.jpg"), img_cv2.astype(np.uint8))

        H_, W_, _ = img_.shape
        img = img_
        mask = scipy.io.loadmat(mask_fp)
        for i in range(len(mask['groundTruth'][0])):
            gtseg = mask['groundTruth'][0][i][0][0][0]

            label_, _ = convert_label(gtseg)
            path_save_mask = final_gt_dir / (f"{mask_fp.stem}-{i}.csv")
            np.savetxt(path_save_mask, (label_ + 1).astype(int), fmt='%i', delimiter=",")
            cv2.imwrite(final_image_dir / (mask_fp.stem + f"-{i}.jpg"),
                        cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))  #


def data_preprocess(split):
    """
    Convert train/val data to npy format and do the data preprocessing.

    :param split: should not be test.
    :return:
    """
    if split=="test":
        print("this is a safety check that you should use another function for test set")
        return


    image_dir = Path("/path/to/BSD500/images_og", split)
    gt_dir = Path("/path/to/BSD500/ground_truth_og", split)

    final_image_dir = Path("/path/to/BSD500/images_preprocess", split)
    final_gt_dir = Path("/path/to/BSD500/ground_truth_preprocess", split)

    # Note this assumes the top level dirs have already been made
    if not final_image_dir.exists():
        final_image_dir.mkdir(parents=True)
    if not final_gt_dir.exists():
        final_gt_dir.mkdir(parents=True)

    gt_masks = gt_dir.glob('*.mat')

    num_classes_list = []
    for mask_fp in gt_masks:
        mask = scipy.io.loadmat(mask_fp)

        # load the image
        img_ = io.imread(image_dir / (mask_fp.stem + ".jpg"))

        # crop image FCN does this
        H_, W_, _ = img_.shape

        # crop to 16*n size
        if H_ == 321 and W_ == 481:
            img = img_[:320, :480, :]
        elif H_ == 481 and W_ == 321:
            img = img_[:480, :320, :]
        else:
            raise Exception('it is not BSDS500 images')

        # ['Segmentation'][0][0] just grabs the mask for the corresponding human annotation. Stored as array(array(array
        # ['groundTruth'][0][i] gives the ith mask for the image
        # the mat object is dictionary and ['groundTruth'] gives the gt masks which is in a single element list and so
        # to get the masks we need ['groundTruth'][0] which gives a list of typically 5
        for i in range(len(mask['groundTruth'][0])):
            gtseg = mask['groundTruth'][0][i][0][0][0]
            # https://github.com/fuy34/superpixel_fcn/blob/master/data_preprocessing/pre_process_bsd500.py#L103

            # now save the image with _i
            np.save(final_image_dir / (mask_fp.stem + f"_{i}.npy"), img)

            label_, _ = convert_label(gtseg)

            num_class, morethan50 = get_labels_more(gtseg)
            if morethan50:
                num_classes_list.append(num_class)

            if H_ == 321 and W_ == 481:
                label = label_[:320, :480]
            elif H_ == 481 and W_ == 321:
                label = label_[:480, :320]
            else:
                raise Exception('dimensions are wrong')
            np.save(final_gt_dir / (mask_fp.stem + f"_{i}.npy"), label)


class CustomDataset(Dataset):
    """
    This is a helper class for the lightning module
    """
    def __init__(self, image_dir, gt_dir, edge_dir, transform=None, height: int = -1, width: int = -1,
                 model_name: str = "None", random_scale=True, flag_train=True):
        """

        :param image_dir: where the preprocessed images are located
        :param gt_dir: where the preprocessed ground truth images are located
        :param transform:
        :param height:
        :param width:
        :param model_name: which model, so to switch the transforms as needed
        :param random_scale:
        """
        self.image_dir = Path(image_dir)
        self.gt_dir = Path(gt_dir)
        self.edge_dir = Path(edge_dir)
        self.transform = transform
        self.height = height
        self.width = width
        self.flag_train = flag_train
        self.model_name = model_name

        if self.model_name == "scn" or self.model_name == "cds" or self.model_name == "ssm":
            self.num_classes = 50
        elif self.model_name == "ainet":
            self.num_classes = 51
        else:
            raise NotImplementedError

        if self.model_name == "ssm":
            self.edge_files = sorted(self.edge_dir.glob("*.npy"))

        # Get sorted lists of image and ground truth files (assuming filenames match)
        self.image_files = sorted(self.image_dir.glob("*.npy"))
        self.gt_files = sorted(self.gt_dir.glob("*.npy"))
        assert len(self.image_files) != 0 or len(self.gt_files) != 0
        assert len(self.image_files) == len(self.gt_files), "Mismatch in image and ground truth count"
        self.scale = random_scale

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        gt_path = self.gt_files[idx]

        # Load image and ground truth
        img = np.load(image_path)
        gt = np.load(gt_path).astype(np.uint8)

        # custom data operations per model
        if self.scale and self.model_name == "cds":
            # https://github.com/rookiie/CDSpixel/blob/main/datasets/listdataset.py
            img, gt = self.generate_scale_label_cds(img, gt)

        if self.model_name == "ssm":
            # https://github.com/jiaxhm/SSMamba/blob/main/datasets/listdataset.py
            edge_path = self.edge_files[idx]
            l_edge = np.load(edge_path)
            l_edge[l_edge > 0.5] = 1.
            l_edge[(l_edge > 0) & (l_edge <= 0.5)] = 2.

        if self.scale and self.model_name == "ssm":
            img, gt, l_edge = self.generate_scale_label_ssm(img, gt, l_edge.astype(np.uint8))

        if self.model_name == "ssm":
            # https://github.com/jiaxhm/SSMamba/blob/main/main.py#L136

            # all other models
            augmentations = self.transform(image=img, mask=gt, edge=l_edge)
            image = augmentations["image"]
            mask = augmentations["mask"].long()
            edge = augmentations["edge"]

        else:
            # all other models
            augmentations = self.transform(image=img, mask=gt)
            image = augmentations["image"]
            mask = augmentations["mask"].long()

        flattened_gt = rearrange(mask, 'h w -> (h w)')
        one_hot_vectors = F.one_hot(flattened_gt,
                                    num_classes=self.num_classes).float()  # Note num_classes can be 51 for ainet
        feat_tensor_label = rearrange(one_hot_vectors, '(h w) c -> c h w', h=self.height, w=self.width)

        if self.model_name == "cds":
            # https://github.com/rookiie/CDSpixel/blob/main/datasets/listdataset.py
            return image, mask, flattened_gt, feat_tensor_label, self._get_sobel(image)  # CDS
        elif self.model_name == "ainet":
            # https://github.com/YanFangCS/AINET/blob/main/datasets/dataset_loader.py
            patch_labels, patch_posis = self.local_patch_sampler(mask)
            # This needs to be mask so that it matches the shape.
            patch_posis = torch.from_numpy(patch_posis).long()
            patch_labels = torch.from_numpy(patch_labels).float()

            if self.flag_train == 'train':
                image, label = self.patch_shuffle(image, mask)
                image, label = self.patch_shuffle(image, label)

            return image, mask, flattened_gt, feat_tensor_label, patch_posis, patch_labels
        elif self.model_name == "scn":
            return image, mask, flattened_gt, feat_tensor_label
        elif self.model_name == "ssm":
            return image, mask, flattened_gt, feat_tensor_label, edge
        else:
            raise NotImplementedError

    def _get_sobel(self, x):
        # https://github.com/rookiie/CDSpixel/blob/main/datasets/listdataset.py
        kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                   [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]
        kernels = np.asarray(kernels)  # 2,3,3
        kernels = np.expand_dims(kernels, 1).repeat(3, axis=1)
        kernels = torch.from_numpy(kernels.astype(np.float32))

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        sob = F.conv2d(x, kernels, stride=1, padding=1)
        return torch.sum(sob, dim=1, keepdim=True).repeat(1, 3, 1, 1).squeeze()

    def generate_scale_label_ssm(self, image, label, edge):
        # https://github.com/jiaxhm/SSMamba/blob/main/datasets/listdataset.py#L52
        f_scale = 0.7 + np.random.randint(0, 8) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        edge = cv2.resize(edge, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label, edge

    def generate_scale_label_cds(self, image, label):
        """
        Note not doing HSV as they do not use this. Also note there is a v2 of this method however the code only uses this version
        https://github.com/rookiie/CDSpixel/blob/main/datasets/listdataset.py#L51
        :param image:
        :param label:
        :return:
        """
        f_scale = 0.7 + np.random.randint(0, 8) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def local_patch_sampler(self, seg_label, patch_height=5, patch_width=5, disc=1, max_patch=500):
        # https://github.com/YanFangCS/AINET/blob/main/datasets/data_util.py#L164
        seg_boundaries = find_boundaries(seg_label) * 1
        # determine the patch number
        patch_num = np.sum(seg_boundaries) // (max(patch_height, patch_width) * disc)
        patch_num = min(max_patch, patch_num)

        seg_boundaries[0:patch_height, :] = 0
        seg_boundaries[:, 0:patch_width] = 0
        seg_boundaries[-patch_height:, :] = 0
        seg_boundaries[:, -patch_width:] = 0

        bd_index = np.where(seg_boundaries == 1)
        total_bs_pixels = bd_index[0].size

        patch_list = []  # record the patch posi, idx and offset
        label_list = []
        row_offset = patch_height // 2
        col_offset = patch_width // 2
        # tmp_boundaries = np.tile(seg_boundaries[:,:,None]*255, (1,1,3))
        # cv2.imwrite('bd.png', tmp_boundaries)
        for i in range(total_bs_pixels):
            rand_idx = random.randint(0, total_bs_pixels - 1)
            row_idx, col_idx = bd_index[0][rand_idx], bd_index[1][rand_idx]
            row_start, row_end = row_idx - row_offset, row_idx + col_offset + 1
            col_start, col_end = col_idx - col_offset, col_idx + col_offset + 1

            count = 0
            label_patch = seg_label[row_start:row_end, col_start:col_end]
            if np.unique(label_patch).size == 2:  # only consider the min patch with only two pixels
                patch_list.append(np.reshape(np.array([row_start, patch_height, col_start, patch_width]), (1, 4)))
                label_patch = self.select_label(label_patch)
                label_list.append(label_patch[None, :, :, :])
                # tmp_boundaries= cv2.rectangle(tmp_boundaries.astype(np.uint8), (col_start, row_start), (col_end, row_end), color=(0, 255, 0), thickness=1)

            if len(label_list) >= patch_num:
                break
        ''' 
        color_map = []
        for s in range(np.unique(seg_label).size):
           r = random.randint(0,255)
           g = random.randint(0,255)
           b = random.randint(0,255)
           color_map.extend([r,g,b])

        im = Image.fromarray(seg_label.astype(np.uint8))
        im.putpalette(color_map)
        im.save('label.png')
        cv2.imwrite('patch.png', tmp_boundaries)
        '''
        if len(label_list) == 0:
            patch_labels = np.ones((1, 4, patch_height, patch_width))
            patch_posi = np.reshape(np.array([1, patch_height, 1, patch_width]), (1, 4))
        else:
            patch_labels = np.concatenate(label_list, axis=0)
            patch_posi = np.concatenate(patch_list, axis=0)

        return patch_labels, patch_posi

    def select_label(self, label_patch):
        # https://github.com/YanFangCS/AINET/blob/main/datasets/data_util.py#L139
        labels = np.unique(label_patch)
        index1 = np.where(label_patch == labels[0])
        index2 = np.where(label_patch == labels[1])
        size1 = index1[0].size
        size2 = index2[0].size

        patch_label1_1 = np.zeros_like(label_patch)
        patch_label1_2 = np.zeros_like(label_patch)
        index1_1 = (index1[0][:size1 // 2], index1[1][:size1 // 2])
        index1_2 = (index1[0][size1 // 2:], index1[1][size1 // 2:])
        patch_label1_1[index1_1] = 1
        patch_label1_2[index1_2] = 1

        patch_label2_1 = np.zeros_like(label_patch)
        patch_label2_2 = np.zeros_like(label_patch)
        index2_1 = (index2[0][:size2 // 2], index2[1][:size2 // 2])
        index2_2 = (index2[0][size2 // 2:], index2[1][size2 // 2:])
        patch_label2_1[index2_1] = 1
        patch_label2_2[index2_2] = 1

        patchs = np.concatenate([patch_label1_1[None, :, :], patch_label1_2[None, :, :], patch_label2_1[None, :, :],
                                 patch_label2_2[None, :, :]], axis=0)

        return patchs

    def patch_shuffle(self, image_data, label_data, region_size=16):
        # https://github.com/YanFangCS/AINET/blob/main/datasets/data_util.py#L11
        # shuffle_flag = random.uniform(0,1.)
        shuffle_flag = np.random.rand()
        if shuffle_flag > 0.5:
            return image_data, label_data

        c, h, w = image_data.shape
        x_interval = h // region_size - 1
        y_interval = w // region_size - 1

        x_index1 = random.randint(0, x_interval * 16)  # * 16
        y_index1 = random.randint(0, y_interval * 16)  # * 16

        x_index2 = random.randint(0, x_interval * 16)  # * 16
        y_index2 = random.randint(0, y_interval * 16)  # * 16

        while x_index1 == x_index2 and y_index1 == y_index2:
            x_index2 = random.randint(0, x_interval * 16)
            y_index2 = random.randint(0, y_interval * 16)  # *16

        # image = copy.deepcopy(image_data)
        # label = copy.deepcopy(label_data)
        image = image_data
        label = label_data

        im_patch1 = image[:, x_index1:x_index1 + region_size, y_index1:y_index1 + region_size]
        im_patch2 = image[:, x_index2:x_index2 + region_size, y_index2:y_index2 + region_size]

        gt_patch1 = label[:, x_index1:x_index1 + region_size, y_index1:y_index1 + region_size]
        gt_patch2 = label[:, x_index2:x_index2 + region_size, y_index2:y_index2 + region_size]

        image_data[:, x_index1:x_index1 + region_size, y_index1:y_index1 + region_size] = im_patch2
        image_data[:, x_index2:x_index2 + region_size, y_index2:y_index2 + region_size] = im_patch1

        label_data[:, x_index1:x_index1 + region_size, y_index1:y_index1 + region_size] = gt_patch2
        label_data[:, x_index2:x_index2 + region_size, y_index2:y_index2 + region_size] = gt_patch1

        image_data, label_data = self.random_offset(image_data, label_data, x_index1, x_index2, y_index1, y_index2,
                                                    x_interval, y_interval)
        # pdb.set_trace()
        # cv2.imwrite('im.png', (image.permute(1,2,0).numpy()[:,:,::-1] + 0.5)*255)
        # cv2.imwrite('im_sf.png', (image_data.permute(1,2,0).numpy()[:,:,::-1] + 0.5)*255)
        # cv2.imwrite('label_sf.png', (label_data.permute(1,2,0).numpy()[:,:,0])*50)
        # pdb.set_trace()

        return image_data, label_data

    def random_offset(self, image_data, label_data, x_index1, x_index2, y_index1, y_index2, x_interval, y_interval):
        # https://github.com/YanFangCS/AINET/blob/main/datasets/data_util.py#L57
        h_or_v_flag = np.random.rand()  # determine which direction to conduct offset
        H_offset = h_or_v_flag > 0.5
        region_size = 16
        offset_dis = random.randint(0, 16)
        if offset_dis == 0:
            return image_data, label_data

        if H_offset:
            # random offset along horizon direction
            x_idx = random.randint(0, x_interval * 16)  # * 16
            start_idx = random.randint(0, y_interval * 16)
            end_idx = random.randint(start_idx + 16, (y_interval + 1) * 16)  # * 16
            # start_idx = start_idx * 16

            im_patch = image_data[:, x_idx:x_idx + region_size, start_idx:end_idx]
            gt_patch = label_data[:, x_idx:x_idx + region_size, start_idx:end_idx]
            # patch_len = end_idx - start_idx

            replace_or_zero = np.random.rand()
            if replace_or_zero > 0.5:  # replace
                # if replace_or_zero > 0.75:#forward
                bf = int((replace_or_zero > 0.75) * 2 - 1)
                new_im_patch = torch.cat([im_patch[:, :, -bf * offset_dis:], im_patch[:, :, :-bf * offset_dis]], dim=2)
                new_gt_patch = torch.cat([gt_patch[:, :, -bf * offset_dis:], gt_patch[:, :, :-bf * offset_dis]], dim=2)
                # else:#backward
                #    new_im_patch = torch.cat([im_patch[:,:, offset_dis:], im_patch[:, :, :offset_dis]], dim=2)
                #    new_gt_patch = torch.cat([gt_patch[:,:, offset_dis:], gt_patch[:, :, :offset_dis]], dim=2)

                image_data[:, x_idx:x_idx + region_size, start_idx:end_idx] = new_im_patch
                label_data[:, x_idx:x_idx + region_size, start_idx:end_idx] = new_gt_patch
            else:  # random fill
                random_im_patch = torch.rand(3, 16, offset_dis) * 2 - 1  # to -1---1
                random_gt_patch = torch.ones(1, 16, offset_dis) * 50
                if replace_or_zero < 0.25:  # forward
                    new_im_patch = torch.cat([random_im_patch, im_patch[:, :, :-offset_dis]], dim=2)
                    new_gt_patch = torch.cat([random_gt_patch, gt_patch[:, :, :-offset_dis]], dim=2)
                else:  # backward
                    new_im_patch = torch.cat([im_patch[:, :, offset_dis:], random_im_patch], dim=2)
                    new_gt_patch = torch.cat([gt_patch[:, :, offset_dis:], random_gt_patch], dim=2)

                image_data[:, x_idx:x_idx + region_size, start_idx:end_idx] = new_im_patch
                label_data[:, x_idx:x_idx + region_size, start_idx:end_idx] = new_gt_patch
        else:
            # random offset along horizon direction
            y_idx = random.randint(0, y_interval * 16)  # * 16
            start_idx = random.randint(0, x_interval * 16)
            end_idx = random.randint(start_idx + 16, (x_interval + 1) * 16)
            # start_idx = start_idx * 16

            im_patch = image_data[:, start_idx:end_idx, y_idx:y_idx + region_size]
            gt_patch = label_data[:, start_idx:end_idx, y_idx:y_idx + region_size]
            patch_len = end_idx - start_idx

            replace_or_zero = np.random.rand()
            if replace_or_zero > 0.5:  # replace
                # if replace_or_zero > 0.75:#forward
                bf = int((replace_or_zero > 0.75) * 2 - 1)
                new_im_patch = torch.cat([im_patch[:, -bf * offset_dis:, :], im_patch[:, :-bf * offset_dis, :]], dim=1)
                new_gt_patch = torch.cat([gt_patch[:, -bf * offset_dis:, :], gt_patch[:, :-bf * offset_dis, :]], dim=1)
                # else:#backward
                #    new_im_patch = torch.cat([im_patch[:,offset_dis:,:], im_patch[:, :offset_dis, :]], dim=1)
                #    new_gt_patch = torch.cat([gt_patch[:,offset_dis:,:], gt_patch[:, :offset_dis, :]], dim=1)

                image_data[:, start_idx:end_idx, y_idx:y_idx + region_size] = new_im_patch
                label_data[:, start_idx:end_idx, y_idx:y_idx + region_size] = new_gt_patch
            else:  # random fill
                random_im_patch = torch.rand(3, offset_dis, 16) * 2 - 1
                random_gt_patch = torch.ones(1, offset_dis, 16) * 50
                if replace_or_zero < 0.25:  # forward
                    new_im_patch = torch.cat([random_im_patch, im_patch[:, :-offset_dis, :]], dim=1)
                    new_gt_patch = torch.cat([random_gt_patch, gt_patch[:, :-offset_dis, :]], dim=1)
                else:  # backward
                    new_im_patch = torch.cat([im_patch[:, offset_dis:, :], random_im_patch], dim=1)
                    new_gt_patch = torch.cat([gt_patch[:, offset_dis:, :], random_gt_patch], dim=1)

                image_data[:, start_idx:end_idx, y_idx:y_idx + region_size] = new_im_patch
                label_data[:, start_idx:end_idx, y_idx:y_idx + region_size] = new_gt_patch

        return image_data, label_data


def custom_collate_ainet(batch):
    """
    Need this because the tensor for ainet from the dataloader are different sizes.

    :param batch:
    :return:
    """
    # Separate the variable-sized tensors from fixed-size ones
    images = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    flattened_gts = torch.stack([item[2] for item in batch])
    feat_tensor_labels = torch.stack([item[3] for item in batch])

    # Keep variable-sized tensors as lists (no stacking)
    patch_posis = [item[4] for item in batch]  # List of tensors
    patch_labels = [item[5] for item in batch]  # List of tensors

    return images, masks, flattened_gts, feat_tensor_labels, patch_posis, patch_labels


class CustomDataModule(L.LightningDataModule):
    def __init__(self, dataset_root: str, batch_size: int = 32, num_workers: int = 4, train_transform=None,
                 eval_transform=None, merge_train_val=False, train_size: int = -1, val_size: int = -1,
                 model_name: str = None):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_val_dataset = None
        self.merge_train_val = merge_train_val
        self.train_size = train_size
        self.val_size = val_size
        self.model_name = model_name

    def setup(self, stage=None):
        """Setup datasets for train, val, test splits"""
        if stage in (None, "fit"):
            train_dataset = CustomDataset(
                image_dir=self.dataset_root / "images_preprocess/train",
                gt_dir=self.dataset_root / "ground_truth_preprocess/train",
                edge_dir=self.dataset_root / "ssm_edges/train",
                transform=self.train_transform,
                height=self.train_size,
                width=self.train_size,
                model_name=self.model_name,
                flag_train=True,
            )
            val_dataset = CustomDataset(
                image_dir=self.dataset_root / "images_preprocess/val",
                gt_dir=self.dataset_root / "ground_truth_preprocess/val",
                edge_dir=self.dataset_root / "ssm_edges/val",
                transform=self.eval_transform,
                height=self.val_size,
                width=self.val_size,
                model_name=self.model_name,
                random_scale=False,
                flag_train=False,
            )
            if self.merge_train_val:

                val_dataset_for_train = CustomDataset(
                    image_dir=self.dataset_root / "images_preprocess/val",
                    gt_dir=self.dataset_root / "ground_truth_preprocess/val",
                    edge_dir=self.dataset_root / "ssm_edges/val",
                    transform=self.train_transform,
                    height=self.train_size,
                    width=self.train_size,
                    model_name=self.model_name,
                    flag_train=True,
                )

                # Merge training and validation datasets
                self.train_val_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset_for_train])
                self.val_dataset = val_dataset
            else:
                self.train_dataset = train_dataset
                self.val_dataset = val_dataset

    def train_dataloader(self):
        dataset = self.train_val_dataset if self.merge_train_val else self.train_dataset
        print("the length of train dataset is {}".format(len(dataset)))

        if self.model_name == "scn" or self.model_name == "cds" or self.model_name == "ssm":
            return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                              drop_last=True, pin_memory=True, persistent_workers=True)
            # if debug dataloader use the following
            # return DataLoader(dataset, batch_size=self.batch_size, num_workers=0, shuffle=False,
            #               drop_last=True, pin_memory=True, persistent_workers=False)
        elif self.model_name == "ainet":
            # if debug dataloader use the following
            # return DataLoader(dataset, batch_size=self.batch_size, num_workers=0, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=False, collate_fn=custom_collate_ainet) #DEBUG
            return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                              drop_last=True, pin_memory=True, persistent_workers=True, collate_fn=custom_collate_ainet)
        else:
            raise NotImplementedError

    def val_dataloader(self):
        print("the length of val dataset is {}".format(len(self.val_dataset)))
        if self.model_name == "scn" or self.model_name == "cds" or self.model_name == "ssm":
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                              drop_last=True, pin_memory=True, persistent_workers=True)
        elif self.model_name == "ainet":
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                              drop_last=True, pin_memory=True, persistent_workers=True, collate_fn=custom_collate_ainet)
        else:
            raise NotImplementedError


def grid_sp(center_id=None):
    """
    To visualize how the 3x3 block works on an image with the square blocks for the superpixels.
    :param center_id:
    :return:
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Parameters
    image_size = 208
    num_cells = 13
    cell_size = image_size // num_cells  # 52
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw vertical and horizontal grid lines
    for i in range(num_cells + 1):
        # Horizontal line at i * cell_size
        ax.axhline(y=i * cell_size, color='black', linewidth=1)
        # Vertical line at i * cell_size
        ax.axvline(x=i * cell_size, color='black', linewidth=1)

    # Label each cell with ID
    for row in range(num_cells):
        for col in range(num_cells):
            cell_id = row * num_cells + col
            # Compute center of the cell
            x_center = col * cell_size + cell_size / 2
            y_center = row * cell_size + cell_size / 2

            # Place the text with cell ID
            ax.text(x_center, y_center, str(cell_id),
                    fontsize=12, color='black',
                    ha='center', va='center')


    # Highlight the 3x3 kernel around the center_id
    # Compute the row, column of the center cell
    center_row = center_id // num_cells
    center_col = center_id % num_cells

    # A 3x3 kernel around the center would cover (center_row-1 to center_row+1, center_col-1 to center_col+1)
    kernel_cells = []
    for kr in range(center_row - 1, center_row + 2):
        for kc in range(center_col - 1, center_col + 2):
            # Check if within the grid boundaries (no padding)
            if 0 <= kr < num_cells and 0 <= kc < num_cells:
                kernel_cells.append((kr, kc))

    # Highlight these cells in green
    for (kr, kc) in kernel_cells:
        rect = patches.Rectangle((kc * cell_size, kr * cell_size), cell_size, cell_size,
                                 linewidth=0, edgecolor='none', facecolor='green', alpha=0.3)
        ax.add_patch(rect)

    # Adjust axes and show the figure
    ax.set_xlim(0, image_size)
    ax.set_ylim(image_size, 0)  # Because imshow uses inverted y by default
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def get_block_class_distributons():
    """
    To plot how many classes appear across all blocks for all images in a split. For example how many blocks have a
    single class, two and so on.
    :return:
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 18,  # Base font size
        # 'axes.titlesize': 16,  # Title font size
        'axes.labelsize': 18,  # Axis label font size
        'xtick.labelsize': 16,  # X-axis tick label size
        'ytick.labelsize': 16,  # Y-axis tick label size
        'legend.fontsize': 16,  # Legend font size
        'figure.titlesize': 18,  # Figure title size
        # 'font.family': 'serif',  # Use serif font (more formal)
        # 'font.serif': ['Times New Roman', 'Computer Modern Roman'],  # Preferred fonts
        'text.usetex': False,  # Set to True if you have LaTeX installed
        'axes.linewidth': 1.2,  # Thicker axis lines
        'axes.spines.top': False,  # Remove top spine
        'axes.spines.right': False,  # Remove right spine
        'axes.grid': True,  # Add grid
        'grid.alpha': 0.3,  # Light grid
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.5,  # Thicker plot lines
    })

    # Directory to save plots
    save_dir = Path("./class_distribution_plots/")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
    splits = ['train', 'val']
    # splits = ['train', 'val', 'test']

    for split in splits:
        print(f"-----------------{split}-------------")
        # need to look at the gt masks with no augmentation just preprocessed
        stride = 16
        gt_dir = Path("../databsd/ground_truth_preprocess") / split
        all_files = list(gt_dir.glob("*.npy"))

        all_class_counts = []  # To store per-image 600-length vectors
        for gt_file in all_files:
            gt = np.load(gt_file)  # shape is (320,480) so with stride, number of superpixels is 600
            gt_blocks = rearrange(gt, '(n1 p1) (n2 p2) -> (n1 n2) (p1 p2)', p1=stride, p2=stride)  # shape is (600,256)
            # Count unique classes per block
            counts = [len(np.unique(block)) for block in gt_blocks]
            all_class_counts.extend(counts)  # Flatten all image counts into one list

        # Create dissertation-ready histogram
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        # Calculate histogram data
        bins = range(1, max(all_class_counts) + 2)
        n, bins_edges, patches = ax.hist(all_class_counts,
                                         bins=bins,
                                         density=True,
                                         edgecolor='black',
                                         facecolor='#1f77b4',
                                         alpha=0.7,
                                         linewidth=1.2,
                                         align='left')

        # Customize the plot
        ax.set_xlabel('Number of Unique Classes per Block', fontweight='bold')
        ax.set_ylabel('Relative Frequency', fontweight='bold')

        # Improve tick formatting
        ax.tick_params(axis='both', which='major', labelsize=16, width=1.2)

        # Set integer ticks on x-axis
        ax.set_xticks(range(1, max(all_class_counts) + 1))

        # Add minor ticks for better readability
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', width=0.8, length=4)

        # Improve grid appearance
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.set_axisbelow(True)  # Put grid behind bars

        # Add statistics text box
        mean_classes = np.mean(all_class_counts)
        std_classes = np.std(all_class_counts)
        median_classes = np.median(all_class_counts)
        total_blocks = len(all_class_counts)

        stats_text = f'Statistics:\n'
        stats_text += f'Total blocks: {total_blocks:,}\n'
        stats_text += f'Mean: {mean_classes:.2f}\n'
        stats_text += f'Std: {std_classes:.2f}\n'
        stats_text += f'Median: {median_classes:.1f}'

        # Position the text box
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='white',
                          edgecolor='gray',
                          alpha=0.9),
                fontsize=16)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        output_path = save_dir / f"class_diversity_{split}.pdf"
        plt.savefig(output_path,
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none',
                    format='pdf')
        plt.close(fig)

        print(f"Saved plots for {split}:")
        print(f"  PDF: {output_path}")
        print(f"  Total blocks processed: {len(all_class_counts):,}")
        print(f"  Class range: {min(all_class_counts)} - {max(all_class_counts)}")



def prep_edge_files():
    """
    This is needed to make the edge files which is only needed for SSM.

    This function idea (not code) comes from the SSM which cites ESNET which cites a 2015 paper which clearly describes how to make these edge files
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_Richer_Convolutional_Features_CVPR_2017_paper.pdf
    Richer Convolutional Features for Edge Detection
    https://mmcheng.net/rcfEdge/
    :return:
    """
    from collections import defaultdict
    from skimage import segmentation
    import matplotlib.pyplot as plt
    path_to_images = Path("../databsd/ground_truth_preprocess/")
    save_dir = Path("../databsd/ssm_edges")
    subfolders = ["train", "val"]
    for folder in subfolders:
        current_images_path = path_to_images / folder
        current_path_to_save = save_dir / folder
        npy_files = list(current_images_path.glob("*.npy"))
        grouped_files = defaultdict(list)
        for file_path in npy_files:
            # Get the filename without extension
            filename = file_path.stem  # e.g., '42078_4'
            # Split on last underscore to get base name
            base_name = filename.rsplit('_', 1)[0]  # e.g., '42078'
            # Add to dictionary
            grouped_files[base_name].append(file_path)

        # Convert to regular dict
        grouped_files = dict(grouped_files)
        print(f"Total unique images: {len(grouped_files)}")
        for gt_key in grouped_files.keys():
            gt_masks = grouped_files[gt_key]
            store_all_labels_per_img = []
            for gt_mask in gt_masks:
                temp = np.load(gt_mask).astype(np.uint8)
                boundaries = segmentation.find_boundaries(temp, mode='thick')  # or mode='inner', 'outer', 'subpixel'
                boundaries_binary = boundaries.astype(np.uint8)
                store_all_labels_per_img.append(boundaries_binary)
            average_boundaries = np.mean(store_all_labels_per_img, axis=0)
            # now we want to save this edge file for each annotation
            for gt_mask in gt_masks:
                filepath = current_path_to_save / gt_mask.stem
                np.save(filepath.with_suffix(".npy"), average_boundaries)
                plt.imsave(filepath.with_suffix(".png"), average_boundaries, cmap='gray')


if __name__ == '__main__':
    # one off function calls
    # prep_edge_files()
    get_block_class_distributons()
    # grid_sp(30)

    # data preprocess pipeline
    # print('------------------val--------------------')
    # data_preprocess('val/')
    # print('------------------train--------------------')
    # data_preprocess('train/')
    # print('------------------test--------------------')
    # data_preprocess_test()

    # check splits

    # print('------------------val--------------------')
    # see_if_split_match('val/')
    # print('------------------train--------------------')
    # see_if_split_match('train/')
    # print('------------------test--------------------')
    # see_if_split_match('test/')

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from pathlib import Path
from utils.trainutils import init_spixel_grid_border_copy as init_spixel_grid, update_spixl_map
from connectivity import enforce_connectivity
from skimage.segmentation import mark_boundaries
import os
from skimage import io
import torch.nn.functional as pf
import argparse
from einops import rearrange
from scipy import ndimage
import csv
from infer import rename_checkpoint_keys
from multiprocessing import get_context
import sys


def count_components_per_superpixel(label_map):
    """
    Count connected components inside each superpixel of a label map.

    Args:
        label_map: 2D numpy array of integer superpixel labels
    Returns:
        dict: {label: num_components}
    """
    unique_labels = np.unique(label_map)
    components_per_sp = {}

    structure = ndimage.generate_binary_structure(2, 2) # 8 connectivity

    for label in unique_labels:
        mask = (label_map == label)
        if np.any(mask):
            labeled_mask, num_components = ndimage.label(mask, structure=structure)
            # labeled_mask now has: 0=background, 1=component1, 2=component2, etc.
            components_per_sp[label] = num_components
        else:
            components_per_sp[label] = 0

    return components_per_sp



def stray_pixels_per_superpixel(label_map):
    """
    For each superpixel label, return the number of pixels that belong to
    all connected components except the largest one

    Args:
        label_map: 2D numpy array of integer superpixel labels.
    Returns:
        dict: {label: stray_pixel_count}
              where stray_pixel_count = (sum of areas of all components) - (area of largest component)
    """
    unique_labels = np.unique(label_map)
    stray_pixels = {}

    structure = ndimage.generate_binary_structure(2, 2)# 8 connectivity

    for label in unique_labels:
        mask = (label_map == label)
        if not np.any(mask):
            stray_pixels[label] = 0
            continue

        labeled_mask, num_components = ndimage.label(mask, structure=structure)
        if num_components <= 1:
            # either one component or none => nothing outside the largest
            stray_pixels[label] = 0
            continue

        # Count pixels per component label (ignore 0 which is background)
        comp_ids = labeled_mask[mask].ravel()           # only foreground pixels
        counts = np.bincount(comp_ids)   # indices 1..num_components
        comp_sizes = counts[1:]  # ignore background
        max_size = comp_sizes.max()

        # Check for ties
        if np.sum(comp_sizes == max_size) > 1:
            max_size = 0
        stray_pixels[label] = int(comp_sizes.sum() - max_size)

    return stray_pixels


def calc_stats(spixel_label_map_NOec,list_data_all, img_name):
    components_dict = count_components_per_superpixel(spixel_label_map_NOec)
    pix_per_compon = stray_pixels_per_superpixel(spixel_label_map_NOec)
    for key, comp_count in components_dict.items():
        if comp_count == 1:
            assert pix_per_compon.get(key, None) == 0, (
                f"Assertion failed: Label {key} has {comp_count} component(s) "
                f"but stray pixel count is {pix_per_compon.get(key, None)}"
            )

    sum_all_pre = sum(components_dict.values())
    sum_all = sum_all_pre-len(components_dict)
    list_data_all.append((img_name,sum_all,sum(pix_per_compon.values()) ))


@torch.no_grad()
def calculate_metrics_images(img_path, model, scale, device, list_data, save_path, plot_flag):

    eval_transform = A.Compose(
        [
            A.Normalize(
                mean=(0.411, 0.432, 0.45),
                std=(1, 1, 1)),
            ToTensorV2(),
        ]
    )

    img_ = io.imread(img_path) #RGB
    H_, W_, _ = img_.shape
    seg_denom = int(600 * scale * scale) * 1.0
    if H_ == 321 and W_==481:
        # BSD
        img = cv2.resize(img_, (int(480 * scale), int(320 * scale)), interpolation=cv2.INTER_CUBIC) #og. cv2 does width first
        sp_ids = init_spixel_grid(img_height=320 * scale, img_width=480 * scale, device=device, batch_size=1, downsize=16)
    elif H_ == 481 and W_ == 321:
        # BSD
        sp_ids = init_spixel_grid(img_height=480 * scale, img_width=320 * scale, device=device, batch_size=1, downsize=16)
        img = cv2.resize(img_, (int(320 * scale), int(480 * scale)), interpolation=cv2.INTER_CUBIC) # og, cv2 does width first

    elif H_ == 448 and W_ == 608:
        # NYU
        sp_ids = init_spixel_grid(img_height=480 * scale, img_width=640 * scale, device=device, batch_size=1,
                                  downsize=16)
        img = cv2.resize(img_, (int(640 * scale), int(480 * scale)),
                         interpolation=cv2.INTER_CUBIC)  # og, cv2 does width first
        seg_denom = int(1200 * scale * scale) * 1.0
    else:
        # VOC
        img = cv2.resize(img_, (int(480 * scale), int(480 * scale)), interpolation=cv2.INTER_CUBIC) #og. cv2 does width first
        sp_ids = init_spixel_grid(img_height=480 * scale, img_width=480 * scale, device=device, batch_size=1, downsize=16)


    augments = eval_transform(image=img)
    img1 = augments["image"].to(device)
    # img1 = augments["image"]
    _, h_img, w_img = img1.shape

    Q_9, _ = model(img1.unsqueeze(0))

    curr_spixl_map = update_spixl_map(sp_ids, Q_9, device=device)
    ori_sz_spixel_map = pf.interpolate(curr_spixl_map.type(torch.float), size=(H_, W_), mode='nearest').type(torch.int)
    spix_index_np = ori_sz_spixel_map.squeeze().detach().cpu().numpy().transpose(0, 1)
    spixel_label_map_NOec = spix_index_np.astype(np.int64)
    calc_stats(spixel_label_map_NOec=spixel_label_map_NOec,list_data_all=list_data,img_name=img_path.stem)

    if plot_flag:
        spix_index_np = spix_index_np.astype(np.int64)
        segment_size = (spix_index_np.shape[0] * spix_index_np.shape[1]) / seg_denom
        min_size = int(0.06 * segment_size)
        max_size = int(3 * segment_size)
        spixel_label_map_EC = enforce_connectivity(spix_index_np[None, :, :], min_size, max_size)[0]
        augments_new = eval_transform(image=img_)
        ori_img = augments_new["image"]

        mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.cuda().unsqueeze(0).dtype).view(3, 1, 1)
        given_img_np = (ori_img + mean_values).clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)

        # #with EC
        spixel_bd_image = mark_boundaries(given_img_np / np.max(given_img_np), spixel_label_map_EC.astype(int), color=(0, 1, 1))
        spixel_viz = spixel_bd_image.astype(np.float32).transpose(2, 0, 1)
        spixel_viz_EC = (spixel_viz * 255).astype(np.uint8)

        # # NO EC
        spixel_bd_image = mark_boundaries(given_img_np / np.max(given_img_np), spixel_label_map_NOec.astype(int), color=(0, 1, 1))
        spixel_viz = spixel_bd_image.astype(np.float32).transpose(2, 0, 1)
        spixel_viz_NOEC = (spixel_viz * 255).astype(np.uint8)
        spixl_save_name = os.path.join(save_path, 'spixel_viz', img_path.stem + '_EC.png')
        img_stan_EC = rearrange(spixel_viz_EC, 'c h w -> h w c')
        io.imsave(spixl_save_name, img_stan_EC)
        #
        spixl_save_name = os.path.join(save_path, 'spixel_viz', img_path.stem + '_NO_EC.png')
        img_stan_noEC = rearrange(spixel_viz_NOEC, 'c h w -> h w c')
        io.imsave(spixl_save_name, img_stan_noEC)
    return


def worker_code(scale,save_path_par,test_imgs,pretrained_weight_path,device, model_name, input_height,input_width, plot_flag):
    torch.set_num_threads(4)
    assert (input_height * scale % 16 == 0 and input_width * scale % 16 == 0)
    num_sp = int(input_height / 16 * scale * input_width / 16 * scale)
    save_name = f"SPixelNet_nSpixel_{num_sp}"
    save_path = save_path_par / save_name

    if not save_path.exists():
        save_path.mkdir()

    csv_path_all = save_path / Path("new_metrics.csv")
    header = ['ImageName', 'xc', 'stray_pix']
    # Create CSV file
    with open(csv_path_all, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    print('=> will save everything to {}'.format(save_path))

    if model_name == 'cds':
        from models.CDSpixel import get_cds_model
        model = get_cds_model()
    elif model_name == 'scn':
        from models.SCN import get_scn_model
        model = get_scn_model()
    elif model_name == 'ainet':
        from models.AInet import get_ainet_model
        model = get_ainet_model(device, Train_flag=False)
    elif model_name == 'ssm':
        from models.ssm.local_model import get_ssm_model
        model = get_ssm_model()
    else:
        raise NotImplementedError

    weight_load = torch.load(pretrained_weight_path, map_location=torch.device(device))
    state_dict = weight_load['state_dict']
    state_dict_final = {k.replace("model.", ""): v for k, v in state_dict.items()}
    if model_name == 'ssm':
        state_dict_final = rename_checkpoint_keys(state_dict_final)
    load_status = model.load_state_dict(state_dict_final, strict=False)
    # Check for missing or unexpected keys:
    if load_status.missing_keys:
        raise Exception(
            f"State dict loading mismatch:\n"
            f"Missing keys: {load_status.missing_keys}\n"
            f"Unexpected keys: {load_status.unexpected_keys}"
        )
    elif load_status.unexpected_keys:
        print(f"Unexpected keys: {load_status.unexpected_keys}")
    else:
        print("all keys loaded")
    print(f" the unexpected keys are: {load_status.unexpected_keys}")
    print("=> using pre-trained model '{}'".format(pretrained_weight_path))
    model.to(device)
    model.eval()

    if not os.path.isdir(os.path.join(save_path, 'spixel_viz')):
        os.makedirs(os.path.join(save_path, 'spixel_viz'))
    list_data =[]
    print("confirm length of test imgs", len(test_imgs))
    for test_img in test_imgs:
        calculate_metrics_images(img_path=test_img, model=model, scale=scale, device=device, list_data=list_data, save_path=save_path, plot_flag=plot_flag)

    with open(csv_path_all, mode='w', newline='') as f:
        writer = csv.writer(f)

        # Optional: write header
        writer.writerow(header)

        # Write all rows
        writer.writerows(list_data)
    print("finished with scale ", str(scale))


def main(output_folder, pretrained_weight_path, dir_name, device, model_name, dataset_name, plot_flag: bool):

    if dataset_name == "bsd":
        test_img_path = Path("./databsd/images_preprocess/test")
        scale_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        input_height = 320
        input_width = 480
    elif dataset_name=="nyu":
        test_img_path = Path("/path/to/NYU/img")
        scale_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
        input_height = 480
        input_width = 640

    elif dataset_name == "voc":
        test_img_path = Path("/path/to/voc_clean/val_imgs/")
        scale_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        input_height = 480
        input_width = 480
    else:
        raise ValueError


    test_imgs = test_img_path.glob('*.jpg')
    test_imgs = list(test_imgs)
    print("length of test images is: ", len(test_imgs))


    save_path_par = output_folder / f"{dir_name}"
    if not save_path_par.exists():
        save_path_par.mkdir(parents=False)


    args_iter = [(s, save_path_par, test_imgs, pretrained_weight_path, device, model_name, input_height, input_width, plot_flag) for s in scale_range]
    ctx = get_context("spawn")  # safer with PyTorch
    with ctx.Pool(processes=6) as pool: # was 4
        pool.starmap(worker_code, args_iter)

    ####### Debug
    # for scale in scales:
    #     worker_code(scale, save_path_par, test_imgs, pretrained_weight_path, device, model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str,
                        help="should match from submit job", required=True)
    parser.add_argument("--pretrained_weight_path", type=str,
                        help="the path to store the results", required=True)

    parser.add_argument("--model_name", type=str,
                        help="the type of model", required=True)
    parser.add_argument("--data_name", type=str,
                        help="which data to use", required=True)
    parser.add_argument("--plot_flag", type=int,
                        help="which data to use", required=True)


    cmdargs = parser.parse_args()
    # device = "cpu"
    device = "cuda"
    output_folder = Path("./results/test_set/new_metrics/")
    if not output_folder.exists():
        output_folder.mkdir(parents=False)


    folder_name = cmdargs.folder_name
    pretrained_weight_path=cmdargs.pretrained_weight_path
    model_name = cmdargs.model_name
    data_name = cmdargs.data_name
    plot_flag = cmdargs.plot_flag

    torch.cuda.empty_cache()
    main(output_folder= output_folder, pretrained_weight_path=pretrained_weight_path, device=device,
         dir_name=folder_name, model_name=model_name, dataset_name=data_name, plot_flag=plot_flag)
    sys.exit(0)
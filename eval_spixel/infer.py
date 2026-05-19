"""
This code primarily comes from SCN: https://github.com/fuy34/superpixel_fcn/blob/master/run_infer_bsds.py

Additional code referenced: https://github.com/rookiie/CDSpixel/blob/main/test_bsds.py#L51

SIN, AInet, and SSM were all referenced in this script.
"""

from pathlib import Path
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from models.SIN import update_spixel_map_sin
from utils.trainutils import init_spixel_grid_border_copy as init_spixel_grid, update_spixl_map
from connectivity import enforce_connectivity
import os
from skimage import io
import torch.nn.functional as pf
import argparse
import random
import sys


random.seed(100)
@torch.no_grad()
def test(img_path, model, scale, save_path, device, enf_con: bool):
    # albumentations already normalizes and so even though the flow transforms has two normalizing
    # the albumentations only needs one.
    eval_transform = A.Compose(
        [
            A.Normalize(
                mean=(0.411, 0.432, 0.45),
                std=(1, 1, 1)),
            ToTensorV2(),
        ]
    )


    img_ = io.imread(img_path)


    custom_pix = 0
    if model_name == "sin":
        custom_pix=15

    H_, W_, _ = img_.shape
    seg_denom = int(600 * scale * scale) * 1.0

    if H_ == 321 and W_==481:
        # BSD
        img = cv2.resize(img_, (int(480 * scale)-custom_pix, int(320 * scale)-custom_pix), interpolation=cv2.INTER_CUBIC) #og. cv2 does width first
        sp_ids = init_spixel_grid(img_height=320 * scale, img_width=480 * scale, device=device, batch_size=1, downsize=16)
    elif H_ == 481 and W_ == 321:
        # BSD
        sp_ids = init_spixel_grid(img_height=480 * scale, img_width=320 * scale, device=device, batch_size=1, downsize=16)
        img = cv2.resize(img_, (int(320 * scale)-custom_pix, int(480 * scale)-custom_pix), interpolation=cv2.INTER_CUBIC) # og, cv2 does width first

    elif H_ == 448 and W_ == 608:
        # NYU
        sp_ids = init_spixel_grid(img_height=480 * scale, img_width=640 * scale, device=device, batch_size=1,
                                  downsize=16)
        img = cv2.resize(img_, (int(640 * scale)-custom_pix, int(480 * scale)-custom_pix),
                         interpolation=cv2.INTER_CUBIC)  # og, cv2 does width first
        seg_denom = int(1200 * scale * scale) * 1.0
    else:
        # VOC
        img = cv2.resize(img_, (int(480 * scale)-custom_pix, int(480 * scale)-custom_pix), interpolation=cv2.INTER_CUBIC) #og. cv2 does width first
        sp_ids = init_spixel_grid(img_height=480 * scale, img_width=480 * scale, device=device, batch_size=1, downsize=16)


    augments = eval_transform(image=img)
    img1 = augments["image"].to(device)

    _, h_img, w_img = img1.shape

    if model_name == "sin":
        prob0_v, prob0_h, prob1_v, prob1_h, prob2_v, prob2_h, prob3_v, prob3_h = model(img1.unsqueeze(0))
        curr_spixl_map = update_spixel_map_sin(img1.cuda().unsqueeze(0), prob0_v, prob0_h, prob1_v, prob1_h, prob2_v,
                                           prob2_h, prob3_v, prob3_h)
    else:
        Q_9, _ = model(img1.unsqueeze(0)) # if error may need to modify the return to Q_9 = model(img1.unsqueeze(0))
        curr_spixl_map = update_spixl_map(sp_ids, Q_9, device=device)

    ori_sz_spixel_map = pf.interpolate(curr_spixl_map.type(torch.float), size=(H_, W_), mode='nearest').type(torch.int)
    spix_index_np = ori_sz_spixel_map.squeeze().detach().cpu().numpy().transpose(0, 1)

    if enf_con:
        spix_index_np = spix_index_np.astype(np.int64)
        segment_size = (spix_index_np.shape[0]*spix_index_np.shape[1]) / seg_denom
        min_size = int(0.06 * segment_size)
        max_size = int(3 * segment_size)
        spixel_label_map = enforce_connectivity(spix_index_np[None, :, :], min_size, max_size)[0]
    else:
        spixel_label_map = spix_index_np.astype(np.int64)


    # save the unique maps as csv for eval
    if not os.path.isdir(os.path.join(save_path, 'map_csv')):
        os.makedirs(os.path.join(save_path, 'map_csv'))

    output_path = os.path.join(save_path, 'map_csv', img_path.stem +".csv")
    np.savetxt(output_path, (spixel_label_map + 1).astype(int), fmt='%i', delimiter=",")



def main(output_folder, pretrained_weight_path, dir_name, device, enf_con: bool, model_name, dataset_name):

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


    save_path_par = output_folder / f"{dir_name}"
    if not save_path_par.exists():
        save_path_par.mkdir(parents=False)

    for scale in scale_range:

        assert (input_height * scale % 16 == 0 and input_width * scale % 16 == 0)

        num_sp = int(input_height/16 * scale * input_width/16 * scale)
        save_name = f"SPixelNet_nSpixel_{num_sp}"
        save_path = save_path_par/save_name

        print('=> will save everything to {}'.format(save_path))

        if not save_path.exists():
            save_path.mkdir()


        if model_name == 'cds':
            from models.CDSpixel import get_cds_model
            model = get_cds_model()

        elif model_name == 'scn':
            from models.SCN import get_scn_model
            model = get_scn_model()
        elif model_name == 'ainet':
            from models.AInet import get_ainet_model
            model = get_ainet_model(device,Train_flag=False)
        elif model_name == 'ssm':
            from models.ssm.local_model import get_ssm_model
            model = get_ssm_model()# NOTE triton may only work on cuda:0
        elif model_name == 'sin':
            from models.SIN import get_sin_model
            model = get_sin_model()
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
            # some models will have unexpected keys such as cds, ssm. there is a second part of the model that is only used
            # during training
        else:
            print("all keys loaded")
        print("=> using pre-trained model '{}'".format(pretrained_weight_path))
        model.to(device)
        model.eval()

        for test_img in test_imgs:
            test(img_path=test_img, model=model, scale=scale, save_path=save_path, device=device, enf_con=enf_con)


def rename_checkpoint_keys(checkpoint_dict):
    """
    Rename keys in checkpoint to match the model structure.
    Converts 'local_' prefix to 'local_model.' prefix.
    """
    new_state_dict = {}

    for key, value in checkpoint_dict.items():
        # If key starts with 'local_' but not 'local_model.', replace it
        if key.startswith('local_') and not key.startswith('local_model.'):
            # Replace 'local_' with 'local_model.'
            new_key = key.replace('local_', 'local_model.', 1)
        else:
            new_key = key

        new_state_dict[new_key] = value

    return new_state_dict

if __name__ == '__main__':
    # import pydevd
    # pydevd.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str,
                        help="should match from submit job", required=True)
    parser.add_argument("--pretrained_weight_path", type=str,
                        help="the path to store the results", required=True)

    parser.add_argument("--model_name", type=str,
                        help="the type of model", required=True)

    parser.add_argument("--data_name", type=str,
                        help="which data to use", required=True)

    parser.add_argument("--enforce_con", type=int,
                        help="0 for no, 1 for yes", required=True)


    cmdargs = parser.parse_args()

    device = "cuda:0"
    output_folder = Path("./results/test_set/")

    if not output_folder.exists():
        output_folder.mkdir(parents=False)

    folder_name = cmdargs.folder_name
    enforce_con_bool = cmdargs.enforce_con
    pretrained_weight_path=cmdargs.pretrained_weight_path
    model_name = cmdargs.model_name
    data_name = cmdargs.data_name

    torch.cuda.empty_cache()
    main(output_folder= output_folder, pretrained_weight_path=pretrained_weight_path, device=device,
         dir_name=folder_name, enf_con=enforce_con_bool, model_name=model_name, dataset_name=data_name)
    sys.exit(0)
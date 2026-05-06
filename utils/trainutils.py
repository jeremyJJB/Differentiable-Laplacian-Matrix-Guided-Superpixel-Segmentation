import torch
from einops import rearrange
from skimage.segmentation.slic_superpixels import _enforce_label_connectivity_cython
import numpy as np
import torch.nn.functional as pf


def poolfeat(input, prob, sp_h=2, sp_w=2):
    # https://github.com/fuy34/superpixel_fcn/blob/master/train_util.py#L75
    def feat_prob_sum(feat_sum, prob_sum, shift_feat):
        feat_sum += shift_feat[:, :-1, :, :]
        prob_sum += shift_feat[:, -1:, :, :]
        return feat_sum, prob_sum

    b, _, h, w = input.shape

    h_shift_unit = 1
    w_shift_unit = 1
    p2d = (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit)
    feat_ = torch.cat([input, torch.ones([b, 1, h, w]).to(prob.device)], dim=1)
    prob_feat = pf.avg_pool2d(feat_ * prob.narrow(1, 0, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w

    send_to_top_left = pf.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2 * h_shift_unit:,
                       2 * w_shift_unit:]

    feat_sum = send_to_top_left[:, :-1, :, :].clone()
    prob_sum = send_to_top_left[:, -1:, :, :].clone()

    prob_feat = pf.avg_pool2d(feat_ * prob.narrow(1, 1, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top = pf.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2 * h_shift_unit:, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top)

    prob_feat = pf.avg_pool2d(feat_ * prob.narrow(1, 2, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top_right = pf.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2 * h_shift_unit:, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top_right)

    prob_feat = pf.avg_pool2d(feat_ * prob.narrow(1, 3, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    left = pf.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, left)

    prob_feat = pf.avg_pool2d(feat_ * prob.narrow(1, 4, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    center = pf.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit,
             w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, center)

    prob_feat = pf.avg_pool2d(feat_ * prob.narrow(1, 5, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    right = pf.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit,
            :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, right)

    prob_feat = pf.avg_pool2d(feat_ * prob.narrow(1, 6, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_left = pf.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_left)

    prob_feat = pf.avg_pool2d(feat_ * prob.narrow(1, 7, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom = pf.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit,
             w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom)

    prob_feat = pf.avg_pool2d(feat_ * prob.narrow(1, 8, 1), kernel_size=(sp_h, sp_w),
                             stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_right = pf.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_right)

    pooled_feat = feat_sum / (prob_sum + 1e-8)

    return pooled_feat


def upfeat(input, prob, up_h=2, up_w=2):
    # https://github.com/fuy34/superpixel_fcn/blob/master/train_util.py#L130
    # input b*n*H*W  downsampled
    # prob b*9*h*w
    b, c, h, w = input.shape

    h_shift = 1
    w_shift = 1

    p2d = (w_shift, w_shift, h_shift, h_shift)
    feat_pd = pf.pad(input, p2d, mode='constant', value=0)

    gt_frm_top_left = pf.interpolate(feat_pd[:, :, :-2 * h_shift, :-2 * w_shift], size=(h * up_h, w * up_w),
                                    mode='nearest')
    feat_sum = gt_frm_top_left * prob.narrow(1, 0, 1)

    top = pf.interpolate(feat_pd[:, :, :-2 * h_shift, w_shift:-w_shift], size=(h * up_h, w * up_w),
                        mode='nearest')
    feat_sum += top * prob.narrow(1, 1, 1)

    top_right = pf.interpolate(feat_pd[:, :, :-2 * h_shift, 2 * w_shift:], size=(h * up_h, w * up_w),
                              mode='nearest')
    feat_sum += top_right * prob.narrow(1, 2, 1)

    left = pf.interpolate(feat_pd[:, :, h_shift:-w_shift, :-2 * w_shift], size=(h * up_h, w * up_w),
                         mode='nearest')
    feat_sum += left * prob.narrow(1, 3, 1)

    center = pf.interpolate(input, (h * up_h, w * up_w), mode='nearest')
    feat_sum += center * prob.narrow(1, 4, 1)

    right = pf.interpolate(feat_pd[:, :, h_shift:-w_shift, 2 * w_shift:], size=(h * up_h, w * up_w),
                          mode='nearest')
    feat_sum += right * prob.narrow(1, 5, 1)

    bottom_left = pf.interpolate(feat_pd[:, :, 2 * h_shift:, :-2 * w_shift], size=(h * up_h, w * up_w),
                                mode='nearest')
    feat_sum += bottom_left * prob.narrow(1, 6, 1)

    bottom = pf.interpolate(feat_pd[:, :, 2 * h_shift:, w_shift:-w_shift], size=(h * up_h, w * up_w),
                           mode='nearest')
    feat_sum += bottom * prob.narrow(1, 7, 1)

    bottom_right = pf.interpolate(feat_pd[:, :, 2 * h_shift:, 2 * w_shift:], size=(h * up_h, w * up_w),
                                 mode='nearest')
    feat_sum += bottom_right * prob.narrow(1, 8, 1)

    return feat_sum


def calc_inner_sps_ids(padded_num_sp, device):
    """
    So the border blocks do not have a full window (3x3 block kernel). We pad blocks around the entire image, which now
    gives a full window to each orginial superpixel id. Note however with the padded blocks the id numbering changes.
    However, we only want to grab the og sp ids, so we calculate the block padding to grab them.

    :param padded_num_sp:
    :param device:
    :return:
    """
    # Compute the linear indices of the INNER original blocks within the padded grid
    rows = torch.arange(1, padded_num_sp - 1, device=device)
    cols = torch.arange(1, padded_num_sp - 1, device=device)
    rr, cc = torch.meshgrid(rows, cols)  # both shape [N_h, N_w]
    center_ids = (rr * padded_num_sp + cc).reshape(-1)
    # Define the 9 relative offsets (top‐left → bottom‐right) in the padded flattening
    offsets = torch.tensor([
        -padded_num_sp - 1, -padded_num_sp, -padded_num_sp + 1,
        -1, 0, 1,
        padded_num_sp - 1, padded_num_sp, padded_num_sp + 1
    ], device=device, dtype=torch.long)  # [9]

    # build the block‐ID grid for all windows: [num_windows,9]
    block_ids = center_ids.unsqueeze(1) + offsets.unsqueeze(0)
    return block_ids

def init_spixel_grid_border_copy(img_height, img_width,device, downsize=16, batch_size: int=-1):
    # https://github.com/fuy34/superpixel_fcn/blob/master/train_util.py#L11

    # get spixel id for assignment
    n_spixl_h = int(np.floor(img_height/downsize))
    n_spixl_w = int(np.floor(img_width/downsize))

    spixel_height = int(img_height / (1. * n_spixl_h))
    spixel_width = int(img_width / (1. * n_spixl_w))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)

    spix_idx_tensor =  np.repeat(
        np.repeat(spix_idx_tensor_, spixel_height,axis=1), spixel_width, axis=2)

    torch_spix_idx_tensor = torch.from_numpy(
                np.tile(spix_idx_tensor, (batch_size, 1, 1, 1))).type(torch.float).to(device)
    return torch_spix_idx_tensor

def update_spixl_map(spixl_map_idx_in, assig_map_in, device):
    # https://github.com/fuy34/superpixel_fcn/blob/master/train_util.py#L201

    assig_map = assig_map_in.clone()

    b,_,h,w = assig_map.shape
    _, _, id_h, id_w = spixl_map_idx_in.shape

    if (id_h == h) and (id_w == w):
        spixl_map_idx = spixl_map_idx_in
    else:
        spixl_map_idx = pf.interpolate(spixl_map_idx_in, size=(h,w), mode='nearest')

    assig_max,_ = torch.max(assig_map, dim=1, keepdim= True) # gets the max value for every pixel for the sp assignment
    assignment_ = torch.where(assig_map == assig_max, torch.ones(assig_map.shape).to(device),torch.zeros(assig_map.shape).to(device))

    new_spixl_map_ = spixl_map_idx * assignment_ # winner take all, put the sp id where maximum value was.
    # the shape here is still [b,9,h,w] but the along 9 dimension it is all zeros but for the spixel id of the maximum
    new_spixl_map = torch.sum(new_spixl_map_,dim=1,keepdim=True).type(torch.long) #update so more straightforward int is depreciated

    return new_spixl_map


def shift9pos(input, h_shift_unit=1,  w_shift_unit=1):
    # https://github.com/fuy34/superpixel_fcn/blob/master/train_util.py#L51

    # input should be padding as (c, 1+ height+1, 1+width+1)
    input_pd = np.pad(input, ((h_shift_unit, h_shift_unit), (w_shift_unit, w_shift_unit)), mode='edge')
    input_pd = np.expand_dims(input_pd, axis=0)

    # assign to ...
    top     = input_pd[:, :-2 * h_shift_unit,          w_shift_unit:-w_shift_unit]
    bottom  = input_pd[:, 2 * h_shift_unit:,           w_shift_unit:-w_shift_unit]
    left    = input_pd[:, h_shift_unit:-h_shift_unit,  :-2 * w_shift_unit]
    right   = input_pd[:, h_shift_unit:-h_shift_unit,  2 * w_shift_unit:]

    center = input_pd[:,h_shift_unit:-h_shift_unit,w_shift_unit:-w_shift_unit]

    bottom_right    = input_pd[:, 2 * h_shift_unit:,   2 * w_shift_unit:]
    bottom_left     = input_pd[:, 2 * h_shift_unit:,   :-2 * w_shift_unit]
    top_right       = input_pd[:, :-2 * h_shift_unit,  2 * w_shift_unit:]
    top_left        = input_pd[:, :-2 * h_shift_unit,  :-2 * w_shift_unit]

    shift_tensor = np.concatenate([     top_left,    top,      top_right,
                                        left,        center,      right,
                                        bottom_left, bottom,    bottom_right], axis=0)
    return shift_tensor



@torch.no_grad()
def compute_asa(assign_hard, flattened_gt, enfoce_con, img_size, num_sp):
    #  used to calculate asa during model runs.

    b, N = assign_hard.shape  # N = H*W, m = number of superpixels

    # Step 2: Compute ASA for each batch
    asa_scores = torch.zeros(b, device=flattened_gt.device)  # Store ASA per batch
    for batch_idx in range(b):
        single_img = assign_hard[batch_idx]
        if enfoce_con:
            # enforce connectivity does not work on batch inputs only returns a single mask even when provide
            # multiple batches, needs to be done like this
            sps = assign_hard[batch_idx].contiguous().cpu().numpy()
            sps = rearrange(sps, '(h w) -> h w', h=img_size, w=img_size)
            segment_size = (img_size * img_size) / (int(num_sp) * 1.0)
            min_size = int(0.06 * segment_size)
            max_size = int(3 * segment_size)
            single_img_enforce = _enforce_label_connectivity_cython(sps[None], min_size, max_size)[0]
            single_img = torch.from_numpy(single_img_enforce).to(flattened_gt.device)


        # Step 2: Create mapping from superpixel -> most frequent GT label
        unique_superpixels = torch.unique(single_img)  # Get unique superpixel IDs
        correct_pixel_count = 0  # Count pixels assigned to correct ground-truth segment

        for s in unique_superpixels:
            sp_mask = (single_img == s)  # Boolean mask for pixels in superpixel s
            if enfoce_con:
                sp_mask = rearrange(sp_mask, ' h w -> (h w)')

            # Get the corresponding ground-truth labels for those pixels
            gt_labels = flattened_gt[batch_idx][sp_mask]

            if gt_labels.numel() > 0:  # If there are pixels in this superpixel
                # Find the most common GT label in this superpixel
                most_common_label = torch.mode(gt_labels).values

                # Count pixels where the GT label matches the majority label
                correct_pixel_count += (gt_labels == most_common_label).sum().item()
        # Compute ASA for this batch
        asa_scores[batch_idx] = correct_pixel_count / N

    mean_asa = torch.mean(asa_scores).item()
    return mean_asa

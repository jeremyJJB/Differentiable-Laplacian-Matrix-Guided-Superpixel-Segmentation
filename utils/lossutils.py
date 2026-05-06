import torch
import torch.nn.functional as pf
from einops import rearrange


class EmbedLoss:

    def __init__(self, margin=None, random_sample_size=None):
        self.margin = margin
        self.random_sample_size = random_sample_size

    def random_sample_class_pixels(self, labels, pixel_embeddings):
        """
        This function does not use a boolean mask to reduce the memory foot print from h*w, h*w, to
        :param: labels [b,1,h,w]
        :return:
        """

        with torch.no_grad():
            # contains a list of len(b) where each element is a tensor that stores the unique class labels
            unique_labels_per_img = [label.unique() for label in labels]

            # len(masks_temp) for each image we generate a boolean mask for each class present in that image. For example, given b=2
            # the first image has 3 classes [1,2,3] and the second image has 7 classes. The length of masks_temp is 10. Each image has boolean mask for each class present.
            masks_temp = [img == single for label, img in zip(unique_labels_per_img, labels) for single in label]

            # len(rand_integer_indices) = len(masks_temp), since we have boolean masks, b_mask.nonzero() gives the coordinates of everywhere a True occurs
            # len(b_mask.nonzero()) is the number of Trues in the boolean mask. torch.randperm(len(b_mask.nonzero())) returns a list of random integers from 0 to len(b_mask.nonzero())-1
            rand_integer_indices = [torch.randperm(len(b_mask.nonzero())) for b_mask in masks_temp]

            # b_mask.nonzero() all the coordinates of True (like above).
            # [rand_ints[:self.random_sample_size], :] takes the random permutation of integers and grab the first self.random_sample_size
            # if not len(b_mask.nonzero())> self.random_sample_size (this means that there are fewer than 300 pixels belonging to that class)
            # then we grab all the randomly selected coordinates for that class.
            # In other words rand_indices hold all the randomly selected coordinates for the given class for a given image
            rand_indices = [b_mask.nonzero()[rand_ints[:self.random_sample_size], :] if len(
                b_mask.nonzero()) > self.random_sample_size else b_mask.nonzero()[rand_ints, :] for b_mask, rand_ints in
                            zip(masks_temp, rand_integer_indices)]

            # now sample for cases that have less than the number of pixels for random_smaple_size
            rand_indices = [rand_ints if rand_ints.shape[0] == self.random_sample_size else rand_ints[
                torch.randint(0, rand_ints.shape[0], (self.random_sample_size,), device=labels.device)] for rand_ints in
                            rand_indices]

        # the c here just corresponds to the image, for instance with b=4, c =0,1,2,3. then for each c we then need to grab the randomly selected indices from rand_indices,
        # this is represented by i. going back to our example above. c=0 for the first image, then i=0,1,2 (only three classes) c+i serves as the index into the random_indices list.
        # c,label in enumerate(unique_labels_per_img) this gives c (our image index) and label which is  a tensor that holds all the unique class labels. then
        # for i,_ in enumerate(label), loops through all the unique class labels for a single c (img).  rand_indices[c+i][:,0] x-cord and y-cord follows.
        # This generates the indices that we want to grab from the actual pixel_embeddings tensor.
        # len(selected_pixels) = len(rand_integer_indices) = len(masks_temp). Each element in the list of selected pixels is tensor that holds the pixel embeddings for the randomly
        # selected pixels coordinates for that class (for that specific image) [10,self.random_sample_size]
        selected_pixels = [pixel_embeddings[c][:, rand_indices[c+i][:,0], rand_indices[c+i][:,1]] for c,label in enumerate(unique_labels_per_img) for i,_ in enumerate(label)]

        return selected_pixels, unique_labels_per_img


    def _contrastive_loss_random_sample_pairs(self,pixel_embeddings, segmentation_labels):
        b, height, width = segmentation_labels.size()
        selected_pix_embed, class_labels_per_img = self.random_sample_class_pixels(labels=segmentation_labels, pixel_embeddings=pixel_embeddings)
        min_distances = torch.full((b,), float(0))

        for c,single_img_labels in enumerate(class_labels_per_img):
            if len(single_img_labels) >1:
                # min_distances is equal to zero, so we only want to do this embedding loss for images that contain more than one class
                # otherwise the loss should be zero for that image.
                distances = []

                # loop through all the class pairs in a given image
                for i,single_label in enumerate(single_img_labels):
                    for j in range(i+1,len(single_img_labels)):

                        # now just do the distance calculation between the two embedding vectors (of different classes) for the same image, we grab the minimum distance
                        # for each class combination
                        distances.append(torch.min(torch.linalg.vector_norm(selected_pix_embed[c+j] - selected_pix_embed[c+i], dim=0)))

                # for a given image, we have the minimum distance for each class combination, we now want to grab the minimum of those minimums
                min_distances[c] = min(distances)

        return torch.square(pf.relu(self.margin - torch.mean(min_distances)))

def trace_laplacian_all_windows(node_values):
    """
    node_values: [b, num_windows, H, W]  (here H=W=48)
    returns: trace_per_window of shape [b, num_sp]

    The 48 is because the stride is always 16 for the superpixels.
    Note on terminology, we refer the window as the 3x3 blocks for center for each superpixel. This is the max area for
    the superpixel.
    """
    b, n_windows, H, W = node_values.shape

    # 1) Merge batch & window into one “batch” dim, and add a channel dim
    x = rearrange(node_values, 'b n h w -> (b n) 1 h w')  # [(b*n),1,H,W]

    # 2) Unfold into 3x3 patches around each pixel (with padding=1):
    #    patches: [b*nw, 9, H*W]
    # the padding here is not values (refers to the size) it means that it is padding with zero which is okay because,
    # the padding only affects the blocks that we added (virtual).
    patches = pf.unfold(x, kernel_size=3, padding=1)  # [b*nw, 9, N], N=H*W

    # 3) Extract center vs the 8 neighbors:
    center = patches[:, 4:5, :]               # [b*nw,1,N]
    nbr_idxs = [0,1,2,3,5,6,7,8]
    neighs   = patches[:, nbr_idxs, :]        # [b*nw,8,N]

    # 4) Compute edge‐weights = product of probabilities
    edge_weights = center * neighs            # [b*nw,8,N]

    # 5) Node degree = sum of its 8 edges
    degrees = edge_weights.sum(dim=1)         # [b*nw, N]

    # 6) Laplacian trace = sum of all degrees
    trace_flat = degrees.sum(dim=1)     # [b*nw]

    trace_per_window = rearrange(trace_flat, '(b n) -> b n', b=b, n=n_windows)
    return trace_per_window

def get_3x3_spixel_ids(sp_i: int, N_w: int, N_h: int) -> list:
    """

    :param sp_i: index of superpixel that want the neighbor ids
    :param N_w: number sp in width
    :param N_h: number sp in height
    :return:

    typically N_w=N_h.
    This is just a helper function to return the neighbor ids for a given superpixel id. Used for plotting.
    """
    r = sp_i // N_w
    c = sp_i % N_w
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < N_h and 0 <= cc < N_w:
                neighbor_id = rr * N_w + cc
                neighbors.append(neighbor_id)
    return neighbors



def compute_alignment_loss(content):
    """
    https://github.com/rookiie/CDSpixel/blob/main/loss.py#L40
    :param content:
    :return:
    """
    contentA, contentB = content
    b, c, h, w = contentA.shape
    contentA = pf.unfold(contentA, kernel_size=16, stride=16).permute(0, 2, 1).view(b, -1, c,
                                                                                   16 * 16).contiguous()  # B, sp, dim, grid
    contentB = pf.unfold(contentB, kernel_size=16, stride=16).permute(0, 2, 1).view(b, -1, c,
                                                                                   16 * 16).contiguous()  # B, sp, dim, grid

    contentA = torch.mean(contentA, dim=3)
    contentB = torch.mean(contentB, dim=3)

    Pa = pf.softmax(contentA, dim=2)
    Pb = pf.softmax(contentB, dim=2)

    kl_divergence = torch.sum(Pa * torch.log(Pa / Pb), dim=2).mean()

    return kl_divergence


def cross_entropy_loss_edge(prediction, label):
    """
    https://github.com/jiaxhm/SSMamba/blob/main/loss.py#L38
    :param prediction:
    :param label:
    :return:
    """
    label = label.unsqueeze(1).long() # put an extra dimension to match input
    mask = label.float()

    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    cost = torch.nn.BCEWithLogitsLoss(weight=mask)(prediction.float(), label.float())
    return cost * 10.

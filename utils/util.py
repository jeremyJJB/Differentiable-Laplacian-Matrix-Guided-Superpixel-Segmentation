from skimage.segmentation import mark_boundaries
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # normal runtime
# matplotlib.use('TkAgg') # debug show plots
from matplotlib.backends.backend_pdf import PdfPages
from lightning.pytorch.callbacks import Callback
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from einops import rearrange
import numpy as np
import torch
import os
import math
import random
import matplotlib.colors as mcolors
from utils.trainutils import update_spixl_map, compute_asa
from connectivity import enforce_connectivity
import argparse
import json
from utils.lossutils import get_3x3_spixel_ids

def get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        help="the model name to use", required=True)
    parser.add_argument("--result_path", type=str,
                        help="the path to store the results", required=True)
    parser.add_argument("--lr", type=float,
                        help="the learning rate for the model", required=True)
    parser.add_argument("--data_dir", type=str,
                        help="path to the data", required=True)
    parser.add_argument("--device", type=str,
                        help="this should be the id of the free gpu that has been assigned to you", required=True)
    parser.add_argument("--epochs", type=int,
                        help="How many epochs are you running for?", required=False, default=500)
    parser.add_argument("--weight_decay", type=float,
                        help="weight decay to use in optimizer", required=False)
    parser.add_argument("--batch_size", type=int,
                        help="batch size", required=True)
    parser.add_argument("--num_batches", type=float,
                        help="between 0 and 1.0 for what percent of the data you want to use", required=True)
    parser.add_argument("--train_size", type=int,
                        help="the size of the training images", required=True)
    parser.add_argument("--val_size", type=int,
                        help="the size of the training images", required=True)
    parser.add_argument('--loss_weights', type=json.loads, default={})
    parser.add_argument("--remote", type=str,
                        help="local computer or remote computer", required=True)

    parser.add_argument("--num_workers", type=int,
                        help="local computer or remote computer", required=True)

    parser.add_argument("--pipeline", type=str,
                        help="local computer or remote computer", required=True)

    parser.add_argument("--weights_to_load", type=str,default=None,
                        help="load weights for tv typically", required=False)
    parser.add_argument("--loss_name", type=str,default=None,
                        help="which loss to use", required=True)
    return parser.parse_args()


def prep_model_options():
    cmdargs = get_cmd_args()
    model_options = {'model_name': cmdargs.model_name,
                     'result_path': cmdargs.result_path, 'weight_decay': cmdargs.weight_decay,
                     'batch_size': cmdargs.batch_size, 'num_workers': cmdargs.num_workers, 'device': cmdargs.device,
                     "num_batches": cmdargs.num_batches, 'pipeline': cmdargs.pipeline,
                     'data_dir': cmdargs.data_dir, 'lr': cmdargs.lr, 'loss_name': cmdargs.loss_name,
                     'epochs': cmdargs.epochs, 'train_size': cmdargs.train_size, 'val_size': cmdargs.val_size, 'stride': 16, 'remote': cmdargs.remote,
                     'loss_weights': cmdargs.loss_weights, 'weights_to_load': cmdargs.weights_to_load}

    if model_options['device']=='cpu':
        model_options['accelerator'] = 'cpu'
    else:
        model_options['accelerator'] = 'cuda'

    if not os.path.exists(model_options['result_path']):
        os.makedirs(model_options['result_path'])

    if model_options['model_name']=='scn':
        assert model_options['lr'] == 5e-5
        model_options['LR_decay_epoch'] = 1000
        assert model_options['batch_size'] == 8
        assert model_options['val_size'] == 320
    elif model_options['model_name']=='cds':
        assert model_options['lr'] == 5e-4
        model_options['LR_decay_epoch'] = 10000
        assert model_options['batch_size'] == 8
        assert model_options['val_size'] == 320
        # for baseline number of epochs for tsv is 736
    elif model_options['model_name']=='ainet':
        # baseline t_sep_v 3K epochs
        if model_options['pipeline'] == 'tv':
            assert model_options['lr'] == 4e-5
            assert model_options['epochs'] == 1500
        if model_options['pipeline'] == 't_sep_v':
            assert model_options['lr'] == 8e-5
        model_options['LR_decay_epoch'] = 2985
        # 2985 from 1063/16=67, 67*2985 = 200,000K iterations
        assert model_options['batch_size'] == 16
        assert model_options['val_size'] == 208
    elif model_options['model_name']=='ssm':
        assert model_options['lr'] == 5e-4
        model_options['LR_decay_epoch'] = 10000
        assert model_options['batch_size'] == 16
        assert model_options['val_size'] == 320
    else:
        raise ValueError(f"Unknown model: {model_options['model_name']}")

    return model_options


class MetricsCallback(Callback):
    def __init__(self, train_type: str = 'Train', model_name: str = '',):
        super().__init__()
        self.train_losses = []
        self.train_recon_loss = []
        self.train_contrastive_loss = []
        self.train_othertwo_loss = []
        self.train_otherone_loss = []
        self.train_pos_loss = []
        self.train_lap = []
        self.train_mean_asa = []
        self.val_losses = []
        self.val_recon_loss = []
        self.val_contrastive_loss = []
        self.val_othertwo_loss = []
        self.val_otherone_loss = []
        self.val_pos_loss = []
        self.val_lap = []
        self.val_mean_asa = []
        self.model_name = model_name
        self.train_type = train_type
        self.val_asa_enforce_b = []
        self.val_asa_enforce = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Retrieve the logged train_loss from the current epoch
        train_loss = trainer.logged_metrics.get('train_loss')
        train_recon_loss = trainer.logged_metrics.get('train_recon_loss')
        train_contrastive_loss = trainer.logged_metrics.get('train_contrastive_loss')
        train_othertwo_loss = trainer.logged_metrics.get('train_othertwo_loss')
        train_otherone_loss = trainer.logged_metrics.get('train_otherone_loss')
        train_pos_loss = trainer.logged_metrics.get('train_pos_loss')
        train_lap = trainer.logged_metrics.get('train_lap')
        train_mean_asa = trainer.logged_metrics.get('train_mean_asa')

        if train_loss is not None:
            self.train_losses.append(train_loss.item())  # Ensure to convert tensor to Python scalar
            self.train_recon_loss.append(train_recon_loss.item())
            self.train_contrastive_loss.append(train_contrastive_loss.item())
            self.train_othertwo_loss.append(train_othertwo_loss.item())
            self.train_otherone_loss.append(train_otherone_loss.item())
            self.train_pos_loss.append(train_pos_loss.item())
            self.train_lap.append(train_lap.item())
            self.train_mean_asa.append(train_mean_asa.item())

    def on_train_end(self, trainer, pl_module):
        print(
            f"{self.train_type} completed. Min {self.train_type} loss is {min(self.train_losses)} at epoch: {self.train_losses.index(min(self.train_losses))}")


    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Called after each validation batch.
        """
        if trainer.current_epoch % 25 != 0:
            return

        if self.model_name == "cds":
            img_input, gt_labels, flattened_gt, feat_label, sob = batch
        elif self.model_name == "scn":
            img_input, gt_labels, flattened_gt, feat_label = batch
        elif self.model_name == "ainet":
            img_input, gt_labels, flattened_gt, feat_label, patch_posi, patch_label = batch
        elif self.model_name == "ssm":
            img_input, gt_labels, flattened_gt, feat_label, edge  = batch
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
        Q_9, _ = pl_module.model(img_input)
        curr_spixl_map = update_spixl_map(pl_module.val_sp_ids, Q_9, device=pl_module.device)
        Q_hard_assign_id = rearrange(curr_spixl_map, ' b 1 h w -> b (h w)')
        b_asa = compute_asa(assign_hard=Q_hard_assign_id, flattened_gt=flattened_gt, enfoce_con=False, img_size=pl_module.val_width, num_sp=pl_module.val_num_sp)
        self.val_asa_enforce_b.append(b_asa)  # Store per-batch stats


    def on_validation_epoch_end(self, trainer, pl_module):
        # Retrieve the logged train_loss from the current epoch
        avg_stat = sum(self.val_asa_enforce_b) / len(self.val_asa_enforce_b) if self.val_asa_enforce_b else 0.0
        self.val_asa_enforce.append(avg_stat)

        val_loss = trainer.logged_metrics.get('val_loss')
        val_recon_loss = trainer.logged_metrics.get('val_recon_loss')
        val_contrastive_loss = trainer.logged_metrics.get('val_contrastive_loss')
        val_othertwo_loss = trainer.logged_metrics.get('val_othertwo_loss')
        val_otherone_loss = trainer.logged_metrics.get('val_otherone_loss')
        val_pos_loss = trainer.logged_metrics.get('val_pos_loss')
        val_lap = trainer.logged_metrics.get('val_lap')
        val_mean_asa = trainer.logged_metrics.get('val_mean_asa')
        if val_loss is not None:
            self.val_losses.append(val_loss.item())
            self.val_recon_loss.append(val_recon_loss.item())
            self.val_contrastive_loss.append(val_contrastive_loss.item())  # Ensure to convert tensor to Python scalar
            self.val_othertwo_loss.append(val_othertwo_loss.item())  # Ensure to convert tensor to Python scalar
            self.val_otherone_loss.append(val_otherone_loss.item())  # Ensure to convert tensor to Python scalar
            self.val_pos_loss.append(val_pos_loss.item())
            self.val_lap.append(val_lap.item())
            self.val_mean_asa.append(val_mean_asa.item())


    def write_to_file(self,dir_path):
        train_loss_fp = os.path.join(dir_path,'train_loss.txt')
        train_recon_loss_fp = os.path.join(dir_path,'train_recon_loss.txt')
        train_contrastive_loss_fp = os.path.join(dir_path,'train_contrastive_loss.txt')
        train_othertwo_loss_fp = os.path.join(dir_path,'train_othertwo_loss.txt')
        train_otherone_loss_fp = os.path.join(dir_path,'train_otherone_loss.txt')
        train_pos_loss_fp = os.path.join(dir_path,'train_pos_loss.txt')
        train_lap_fp = os.path.join(dir_path,'train_lap_loss.txt')
        train_mean_asa_fp = os.path.join(dir_path,'train_mean_asa.txt')

        val_loss_fp = os.path.join(dir_path,'val_loss.txt')
        val_recon_loss_fp = os.path.join(dir_path,'val_recon_loss.txt')
        val_contrastive_loss_fp = os.path.join(dir_path,'val_contrastive_loss.txt')
        val_othertwo_loss_fp = os.path.join(dir_path,'val_othertwo_loss.txt')
        val_otherone_loss_fp = os.path.join(dir_path,'val_otherone_loss.txt')
        val_pos_loss_fp = os.path.join(dir_path,'val_pos_loss.txt')
        val_lap_fp = os.path.join(dir_path,'val_lap.txt')
        val_mean_asa_fp = os.path.join(dir_path,'val_mean_asa.txt')


        self.write_list_to_file(train_loss_fp, self.train_losses)
        self.write_list_to_file(train_recon_loss_fp, self.train_recon_loss)
        self.write_list_to_file(train_contrastive_loss_fp, self.train_contrastive_loss)
        self.write_list_to_file(train_othertwo_loss_fp, self.train_othertwo_loss)
        self.write_list_to_file(train_otherone_loss_fp, self.train_otherone_loss)
        self.write_list_to_file(train_pos_loss_fp, self.train_pos_loss)
        self.write_list_to_file(train_lap_fp, self.train_lap)
        self.write_list_to_file(train_mean_asa_fp, self.train_mean_asa)

        self.write_list_to_file(val_loss_fp, self.val_losses)
        self.write_list_to_file(val_recon_loss_fp, self.val_recon_loss)
        self.write_list_to_file(val_contrastive_loss_fp, self.val_contrastive_loss)
        self.write_list_to_file(val_othertwo_loss_fp, self.val_othertwo_loss)
        self.write_list_to_file(val_otherone_loss_fp, self.val_otherone_loss)
        self.write_list_to_file(val_pos_loss_fp, self.val_pos_loss)
        self.write_list_to_file(val_lap_fp, self.val_lap)
        self.write_list_to_file(val_mean_asa_fp, self.val_mean_asa)


    @staticmethod
    def write_list_to_file(file_path_and_name, list_to_write) -> None:
        textfile = open(file_path_and_name, "w")
        for element in list_to_write:
            textfile.write(str(element) + "\n")
        textfile.close()


class BaseEpochGraphs(Callback):
    def __init__(self, num_imgs=1, run_type: str = 'Train', num_sp=None, total_sp=None, model_name=None):
        self.num_imgs = num_imgs
        self.type = run_type
        # we can set which superpixels we want to watch, same with images, just need to modify later code with self._fixed_images
        self.random_superpixel_ids =[random.randint(0,total_sp-1) for _ in range(num_sp)] +[153]
        self._fixed_images = None
        self.model_name = model_name

    @torch.no_grad()
    def _epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", sp_ids, height, width, num_sp) -> None:
        if trainer.current_epoch % 10 != 0 and self.type != 'Test':
            return

        if self.type == 'Train' or self.type == 'TrainVal':
            attr_name = 'train_dataloader'
        elif self.type == 'Val':
            attr_name = 'val_dataloaders'
        elif self.type == 'Test':
            attr_name = 'test_dataloaders'
        else:
            raise Exception('wrong type passed')

        # Only fetch once
        if self._fixed_images is None and self.type == 'Val':

            val_loader = getattr(trainer, attr_name)

            selected_imgs = []
            selected_labels = []
            selected_flattened_gt = []
            selected_feat_label = []

            # (batch_idx -> img_idx within that batch)
            target_selections = {2: 0, 4: 5, 5: 0, 6: 0, 10: 5, 11: 7, 13:0, 16:6}
            max_batch = max(target_selections)

            for batch_idx, batch in enumerate(val_loader):
                if batch_idx > max_batch:
                    break
                if batch_idx not in target_selections:
                    continue

                img_idx = target_selections[batch_idx]
                if self.model_name == "cds":
                    img_input_batch, segmentation_labels_batch, flattened_gt_batch, feat_label_batch, sob = batch
                elif self.model_name == "scn":
                    img_input_batch, segmentation_labels_batch, flattened_gt_batch, feat_label_batch = batch
                elif self.model_name == "ainet":
                    img_input_batch, segmentation_labels_batch, flattened_gt_batch, feat_label_batch, patch_posi_batch, patch_label_batch = batch
                elif self.model_name == "ssm":
                    img_input_batch, segmentation_labels_batch, flattened_gt_batch, feat_label_batch, edge_batch = batch
                else:
                    raise Exception('wrong model_name passed')

                selected_imgs.append(img_input_batch[[img_idx]])
                selected_labels.append(segmentation_labels_batch[[img_idx]])
                selected_flattened_gt.append(flattened_gt_batch[[img_idx]])
                selected_feat_label.append(feat_label_batch[[img_idx]])

            # Stack into single tensors and cache
            img = torch.cat(selected_imgs, dim=0)
            seg_labels = torch.cat(selected_labels, dim=0)
            flattened_gt = torch.cat(selected_flattened_gt, dim=0)
            feat_label = torch.cat(selected_feat_label, dim=0)
            self._fixed_images = (img, seg_labels, flattened_gt, feat_label)


        # Use one of two paths depending on mode
        if self.type == 'Val':
            img_input, segmentation_labels, flattened_gt, feat_label = self._fixed_images
            img = img_input.to(pl_module.device)
            Q_9, _ = pl_module.model(img)
            curr_spixl_map = update_spixl_map(sp_ids[0:self.num_imgs], Q_9, device=Q_9.device)
            # we need to drop the batch size of sp id tracking matrix so that it matches the number of images we are using in val
        else:
            dataloader = getattr(trainer, attr_name)
            batch = next(iter(dataloader))
            if self.model_name == "cds":
                img_input, segmentation_labels, flattened_gt, feat_label, sob_input = batch
            elif self.model_name == "scn":
                img_input, segmentation_labels, flattened_gt, feat_label = batch
            elif self.model_name == "ssm":
                img_input, segmentation_labels, flattened_gt, feat_label, edge = batch
            elif self.model_name == "ainet":
                img_input, segmentation_labels, flattened_gt, feat_label, patch_posi, patch_label  = batch
            else:
                raise Exception('wrong model_name passed')
            img = img_input.to(pl_module.device)
            if self.model_name == "cds":
                sob = sob_input.to(pl_module.device)
                Q_9, prob_assit, align, mi, pix_embed = pl_module.model(img, sob)
            elif self.model_name == "scn":
                Q_9, _ = pl_module.model(img)
            elif self.model_name == "ssm":
                Q_9,_, _ = pl_module.model(img)
            elif self.model_name == "ainet":
                Q_9, _ = pl_module.model(img)
            else:
                raise Exception('wrong model_name passed')
            curr_spixl_map = update_spixl_map(sp_ids, Q_9, device=Q_9.device)

        segmentation_labels = segmentation_labels.to(pl_module.device)
        max_indices = rearrange(curr_spixl_map, ' b 1 h w -> b (h w)')
        # note: os.path.join did not work here and not sure why. the log_dir was always empty
        result_path = trainer.log_dir + f"/epochviz_{self.type}/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        pdf_path = result_path + f"{self.type}_gt_visualization_{trainer.current_epoch}_output.pdf"
        # create a PdfPages object
        pdf = PdfPages(pdf_path)

        for i in range(self.num_imgs):
            fig, ax = plt.subplots(2, 2, figsize=(10, 12))

            img_stan = rearrange(img[i], 'c h w -> h w c').cpu().numpy()[:,:,:3]
            mean = np.array([0.411, 0.432, 0.45])

            # Undo normalization
            img_denorm = (img_stan + mean) * 255
            rgb = np.clip(img_denorm, 0, 255).astype(np.uint8)
            gt_stan = segmentation_labels[i]
            gt_stan_labels = gt_stan.cpu().numpy()

            sp_masks_stan = rearrange(max_indices[i], '(h w) -> h w', h=height,w=width)
            sps = sp_masks_stan.contiguous().cpu().numpy()

            # Plot 1: Input Image
            ax[0, 0].imshow(rgb)
            ax[0, 0].set_title("Input Image")
            ax[0, 0].axis('off')  # Hide axes

            mark_bound_img = mark_boundaries(image=rgb, label_img=sps,mode='subpixel')
            norm_mark =  (mark_bound_img - np.min(mark_bound_img)) / (np.max(mark_bound_img) - np.min(mark_bound_img))

            ax[1, 0].imshow(norm_mark, interpolation='none')
            ax[1, 0].set_title('overlay sp')
            ax[1, 0].axis('off')

            # Plot 2: Ground Truth
            # Plot the image with masked values
            img_label = ax[0, 1].imshow(gt_stan_labels, cmap='jet', interpolation='none')
            # Get the unique values and their corresponding colors
            values = np.unique(gt_stan_labels)
            colors = [img_label.cmap(img_label.norm(value)) for value in values]

            # create a patch (proxy artist) for every color
            patches = [mpatches.Patch(color=colors[i], label="Class {l}".format(l=values[i])) for i in
                       range(len(values))]
            ax[0, 1].legend(handles=patches)
            ax[0, 1].set_title("gt")
            ax[0, 1].axis('off')

            # Plot 3: Superpixel Masks
            # Generate a random color for each unique integer in sps
            unique_values = np.unique(sps)
            num_unique_values = len(unique_values)
            # Generate random colors
            colors = np.random.rand(num_unique_values, 3)  # RGB values between 0 and 1
            random_cmap = mcolors.ListedColormap(colors)


            ax[1, 1].imshow(sps, cmap=random_cmap, interpolation='none')
            ax[1, 1].set_title("sp_masks")
            ax[1, 1].axis('off')

            plt.tight_layout()  # Adjust layout to make sure there is no overlap
            pdf.savefig(fig)

            plt.close()
            # code for doing enforce connectivity
            fig, ax = plt.subplots(1, 2,
                                   figsize=(10, 12))

            mark_bound_img = mark_boundaries(image=rgb, label_img=sps,mode='subpixel')
            norm_mark =  (mark_bound_img - np.min(mark_bound_img)) / (np.max(mark_bound_img) - np.min(mark_bound_img))

            ax[0].imshow(norm_mark, interpolation='none')
            ax[0].set_title('overlay sp')
            ax[0].axis('off')  # Hide the unused subplot

            segment_size = (height * width) / (600.0 * (height/320) * (width/420)) # so using the fact that the set is always 320x420 and that the scale here is constant
            # without that the segment_size is too small for train at 208
            #  https://github.com/fuy34/superpixel_fcn/blob/master/run_infer_bsds.py#L99
            min_size = int(0.06 * segment_size)
            max_size = int(3 * segment_size)
            spix_index = enforce_connectivity(sps.astype(np.intp)[None,:,:], min_size, max_size)[0]

            mark_bound_img = mark_boundaries(image=rgb, label_img=spix_index,mode='subpixel')
            norm_mark =  (mark_bound_img - np.min(mark_bound_img)) / (np.max(mark_bound_img) - np.min(mark_bound_img))

            ax[1].imshow(norm_mark, interpolation='none')
            ax[1].set_title('enforce connectivity sp')
            ax[1].axis('off')  # Hide the unused subplot

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

            # superpixel id plots
            for i_sp in self.random_superpixel_ids:
                mask_np = (sps == i_sp)

                # Now lets plot the labels as those points
                # Prepare an overlay for labels at these specific points
                # We create an array with the same shape initialized with NaNs, which are ignored in plots
                label_overlay = np.full(sps.shape, np.nan)
                label_overlay[mask_np] = gt_stan_labels[mask_np]
                fig, ax = plt.subplots(1, 1, figsize=(10, 12), dpi=400)

                # Set the grid spacing
                grid_spacing = height / num_sp

                # Set grid lines to appear on the plot
                ax.set_xticks(np.arange(0, height, grid_spacing))
                ax.set_yticks(np.arange(0, width, grid_spacing))

                # Enable the grid
                ax.grid(True, which='both', color='black', linestyle='-', linewidth=0.5)
                ax.set_aspect('equal')

                # Label each cell with its ID
                for row in range(num_sp):
                    for col in range(num_sp):
                        cell_id = row * num_sp + col
                        # Compute center of the cell
                        x_center = col * 16 + 16 / 2
                        y_center = row * 16 + 16 / 2

                        # Place the text with cell ID
                        ax.text(x_center, y_center, str(cell_id),
                                fontsize=8, color='black',
                                ha='center', va='center')


                ax.imshow(label_overlay)
                num_unique_labels = len(np.unique(label_overlay[~np.isnan(label_overlay)]))

                img_label_sp = ax.imshow(label_overlay, cmap='jet', interpolation='none')
                # Get the unique values and their corresponding colors
                values, class_counts = np.unique(label_overlay[~np.isnan(label_overlay)], return_counts=True)
                colors = [img_label_sp.cmap(img_label_sp.norm(value)) for value in values]
                # create a patch (proxy artist) for every color
                patches = [mpatches.Patch(color=colors[i], label="Class {l}, count {c}".format(l=values[i], c=class_counts[i])) for i in
                           range(len(values))]
                ax.legend(handles=patches,bbox_to_anchor=(1.04, 1), loc="upper left")

                if len(class_counts) == 0:
                    percent_same_label = 0
                else:
                    percent_same_label = (class_counts/ class_counts.sum()).max()

                ax.set_title(f"Labels at sp {i_sp}, num labels: {num_unique_labels}, {percent_same_label*100:.2f}% same label")
                fig.tight_layout()
                pdf.savefig(fig)  # Save the figure to a PDF page
                plt.close()

                ##### Viz code
                fig, ax = plt.subplots(1, 2, figsize=(10, 12), dpi=400)

                my_neighbors = get_3x3_spixel_ids(i_sp, N_w=num_sp, N_h=num_sp)
                colors = plt.cm.tab10(np.linspace(0, 1, len(my_neighbors)))

                overlay = np.full((*sps.shape, 4), 1.0)  # RGBA (default white/transparent)
                for idx,neighbor in enumerate(my_neighbors):
                    mask_np = (sps == neighbor)
                    overlay[mask_np] = colors[idx]

                # Plot
                ax[0].imshow(overlay, interpolation='none')

                # Create legend manually
                handles = [mpatches.Patch(color=colors[idx], label=f"Spixel {neighbor}") for idx, neighbor in
                           enumerate(my_neighbors)]
                ax[0].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
                ax[1].axis('off')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
        pdf.close()


class TrainEpochGraphs(BaseEpochGraphs):
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None: # if this is on_train_epoch_start then we get error with the log no val loss
        self._epoch_end(trainer=trainer, pl_module=pl_module, height=pl_module.train_height, width=pl_module.train_width, num_sp=pl_module.train_num_sp, sp_ids=pl_module.train_sp_ids)


class ValEpochGraphs(BaseEpochGraphs):
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._epoch_end(trainer=trainer, pl_module=pl_module,height=pl_module.val_height, width=pl_module.val_width, num_sp=pl_module.val_num_sp, sp_ids=pl_module.val_sp_ids)



class PrintLossesCallback(Callback):
    def __init__(self, train_type: str = 'Train'):
        self.train_type = train_type

    def on_validation_epoch_end(self, trainer, pl_module):
        # Access the logged values
        val_loss = trainer.callback_metrics.get('val_loss')
        train_loss = trainer.callback_metrics.get('train_loss')
        train_recon_loss = trainer.callback_metrics.get('train_recon_loss')
        train_contrastive_loss = trainer.callback_metrics.get('train_contrastive_loss')
        train_othertwo_loss = trainer.callback_metrics.get('train_othertwo_loss')
        train_otherone_loss = trainer.callback_metrics.get('train_otherone_loss')
        train_asa = trainer.callback_metrics.get('train_mean_asa')
        train_pos_loss = trainer.callback_metrics.get('train_pos_loss')
        val_recon_loss = trainer.callback_metrics.get('val_recon_loss')
        val_contrastive_loss = trainer.callback_metrics.get('val_contrastive_loss')
        val_othertwo_loss = trainer.callback_metrics.get('val_othertwo_loss')
        val_otherone_loss = trainer.callback_metrics.get('val_otherone_loss')
        val_pos_loss = trainer.callback_metrics.get('val_pos_loss')
        train_lap = trainer.callback_metrics.get('train_lap')
        val_asa = trainer.callback_metrics.get('val_mean_asa')
        val_lap = trainer.callback_metrics.get('val_lap')

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        if val_loss is not None and train_loss is not None:
            print(
                f"Epoch: {int(trainer.current_epoch)}, time: {str(current_time)}\n"
                f"\t{self.train_type:<10} Loss: {train_loss.item():<10.8f} "
                f"recon: {train_recon_loss.item():<10.8f} "
                f"contrastive: {train_contrastive_loss.item():<10.8f} "
                f"othertwo: {train_othertwo_loss.item():<10.8f} "
                f"otherone: {train_otherone_loss.item():<10.8f} "
                f"pos: {train_pos_loss.item():<10.8f} "
                f"lap: {train_lap.item():<10.8f} "
                f"no_enf_asa: {train_asa.item():<10.8f}\n"
                f"\t{'Val':<10} Loss: {val_loss.item():<10.8f} "
                f"recon: {val_recon_loss.item():<10.8f} "
                f"contrastive: {val_contrastive_loss.item():<10.8f} "
                f"othertwo: {val_othertwo_loss.item():<10.8f} "
                f"otherone: {val_otherone_loss.item():<10.8f} "
                f"pos: {val_pos_loss.item():<10.8f} "
                f"lap: {val_lap.item():<10.8f} "
                f"no_enf_asa: {val_asa.item():<10.8f}",
                flush=True
            )

def check_sp(num_sp, img_size):
    """
    This function will check the padding by looking at the height of the input and the sp height
    the code will error if the condition was not met.

    :param num_sp: the number of superpixels
    :param img_size: the size of the image. Note this is square.

    :return:
    """
    height_s = int(math.sqrt(num_sp * img_size / img_size))
    width_s = int(math.sqrt(num_sp * img_size / img_size))
    pad_x = (width_s - img_size % width_s) % width_s  # Yes ingore tha padding just make a rule
    pad_y = (height_s - img_size % height_s) % height_s
    assert pad_x == pad_y == 0, "The padding is not correct, needs to be zero"

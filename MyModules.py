from models.CDSpixel import get_cds_model, CLUB
from models.ssm.local_model import get_ssm_model
import lightning as pl
import torch
import torch.optim as optim
import torch.nn.functional as pf
from einops import rearrange
from utils.trainutils import compute_asa, calc_inner_sps_ids
from models.SCN import get_scn_model
from models.AInet import get_ainet_model
from utils.trainutils import init_spixel_grid_border_copy as init_spixel_grid, update_spixl_map, poolfeat, upfeat
from utils.util import check_sp
from utils.lossutils import EmbedLoss, trace_laplacian_all_windows, compute_alignment_loss, cross_entropy_loss_edge


class AISuperpixModel(pl.LightningModule, EmbedLoss):
    def __init__(self, learning_rate=None,  model_name=None, margin=None,random_sample_size=None, device=None,
                 loss_weights=None, train_height:int =-1, train_width: int=-1, val_height:int=-1, val_width:int=-1,
                 stride:int=-1, batch_size:int=-1, wd:float=None, loss_name:str=None,lr_decay_epoch=None):
        pl.LightningModule.__init__(self)
        EmbedLoss.__init__(self,margin=margin, random_sample_size=random_sample_size)
        self.model_name = model_name
        self.LR_decay_epoch = lr_decay_epoch
        self.loss_name = loss_name
        self.train_height = train_height
        self.train_width = train_width
        self.val_height = val_height
        self.val_width = val_width
        self.stride = stride
        self.batch_size = batch_size
        assert (self.train_height // self.stride)  == (self.train_width // self.stride)
        assert (self.val_height // self.stride) == (self.val_width // self.stride)
        self.train_num_sp = self.train_height // self.stride
        self.val_num_sp = (self.val_height // self.stride)
        self.total_train_num_sp = (self.train_height // self.stride)  * (self.train_width // self.stride)
        self.total_val_num_sp =  (self.val_height // self.stride) * (self.val_width // self.stride)
        check_sp(num_sp=self.total_train_num_sp, img_size=self.train_width)
        check_sp(num_sp=self.total_val_num_sp , img_size=self.val_width)


        if model_name == "scn":
            self.model = get_scn_model()
        elif model_name == "ainet":
            self.model = get_ainet_model(device=device, Train_flag=True)
        elif model_name == "cds":
            self.model = get_cds_model()
            self.mi_estimator = CLUB(x_dim=32, y_dim=32)
        elif model_name == "ssm":
            self.model = get_ssm_model()
        else:
            raise NotImplementedError

        self.lr = learning_rate
        self.recon_w = loss_weights['recon']
        self.compact_w = loss_weights['compact']
        self.contrastive_w = loss_weights['contrastive']
        self.lap_w = loss_weights['lap']
        self.wd = wd

        #### static tensors of the 9 superpixel pixel candidate ids for each pixel
        self.register_buffer('train_sp_ids', init_spixel_grid(
            img_height=self.train_height,
            img_width=self.train_width,
            device=device,
            batch_size= self.batch_size,
            downsize=self.stride
        ))

        self.register_buffer('val_sp_ids', init_spixel_grid(
            img_height=self.val_height,
            img_width=self.val_width,
            device=device,
            batch_size= self.batch_size,
            downsize=self.stride
        ))

        ##### XY coords
        coords_train = torch.stack(torch.meshgrid(torch.arange(self.train_height, device=device),
                                            torch.arange(self.train_width, device=device),  indexing="ij"), 0)
        coords_val = torch.stack(torch.meshgrid(torch.arange(self.val_height, device=device),
                                                torch.arange(self.val_width, device=device),  indexing="ij"), 0)

        coords_train =  coords_train[None].repeat(self.batch_size, 1, 1, 1).float()
        coords_val = coords_val[None].repeat(self.batch_size, 1, 1, 1).float()

        self.register_buffer('coords_train', coords_train)
        self.register_buffer('coords_val', coords_val)

        ######## Trace Lap Loss
        self.train_num_sp_pad = self.train_num_sp + 2 # extra block for height and width, since we only have square
        # images for training/val only need a single number here padded num sp H = padded num sp W
        self.val_num_sp_pad = self.val_num_sp + 2
        train_block_ids = calc_inner_sps_ids(padded_num_sp=self.train_num_sp_pad,device=self.device)
        val_block_ids = calc_inner_sps_ids(padded_num_sp=self.val_num_sp_pad,device=self.device)

        # The 9 channels in Q_p correspond in reverse order:
        #  channel 8 = top‐left, 7 = top, …, 0 = bottom‐right
        chan_idxs = torch.arange(9 - 1, -1, -1, device=self.device)  # [8,7,...,0]

        # Expand to full batch for advanced‐indexing:
        # We want to do Q_blocks[b, chan_idxs[j], block_ids[i,j], :, :]
        b_idx_train = torch.arange(self.batch_size, device=self.device)[:, None, None].expand(self.batch_size, self.total_train_num_sp, 9)
        c_idx_train = chan_idxs[None, None, :].expand(self.batch_size, self.total_train_num_sp, 9)
        n_idx_train = train_block_ids[None, :, :].expand(self.batch_size, self.total_train_num_sp, 9)
        ## Val
        b_idx_val = torch.arange(self.batch_size, device=self.device)[:, None, None].expand(self.batch_size, self.total_val_num_sp , 9)
        c_idx_val = chan_idxs[None, None, :].expand(self.batch_size, self.total_val_num_sp, 9)
        n_idx_val = val_block_ids[None, :, :].expand(self.batch_size, self.total_val_num_sp, 9)

        self.register_buffer('b_idx_train', b_idx_train)
        self.register_buffer('c_idx_train', c_idx_train)
        self.register_buffer('n_idx_train', n_idx_train)
        self.register_buffer('b_idx_val', b_idx_val)
        self.register_buffer('c_idx_val', c_idx_val)
        self.register_buffer('n_idx_val', n_idx_val)
        # loss logic selection
        if self.loss_name == "baseline_scn":
            self.train_loss = self._baseline_scn_loss
            self.val_loss = self._baseline_scn_loss
            assert self.recon_w >0 and self.compact_w >0
            assert self.contrastive_w == 0 and self.lap_w == 0
        elif self.loss_name == "novel_scn":
            assert self.recon_w > 0 and self.compact_w > 0 and self.contrastive_w > 0 and self.lap_w > 0
            self.train_loss = self._novel_scn_loss
            self.val_loss = self._novel_scn_loss
        elif self.loss_name == "baseline_ainet":
            self.train_loss = self._baseline_ainet_loss
            self.val_loss = self._baseline_ainet_loss
            assert self.recon_w >0 and self.compact_w >0
            assert self.contrastive_w == 0 and self.lap_w == 0
        elif self.loss_name == "novel_ainet":
            self.train_loss = self._novel_ainet_loss
            self.val_loss = self._novel_ainet_loss
            assert self.recon_w > 0 and self.compact_w > 0 and self.contrastive_w > 0 and self.lap_w > 0
        elif self.loss_name == "baseline_ssm":
            self.train_loss = self._baseline_ssm_train_loss
            self.val_loss = self._baseline_ssm_val_loss
            assert self.recon_w >0 and self.compact_w >0
            assert self.contrastive_w == 0 and self.lap_w == 0
        elif self.loss_name == "novel_ssm":
            assert self.recon_w > 0 and self.compact_w > 0 and self.contrastive_w > 0 and self.lap_w > 0
            self.train_loss = self._novel_ssm_train_loss
            self.val_loss = self._novel_ssm_val_loss
        elif self.loss_name == "baseline_cds":
            self.automatic_optimization = False # CDS as two weight update steps.
            assert self.recon_w > 0 and self.compact_w > 0
            assert self.contrastive_w == 0 and self.lap_w == 0
            self.train_loss = self._baseline_cds_pixel_train
            self.val_loss = self._baseline_cds_pixel_val
        elif self.loss_name == "novel_cds":
            self.automatic_optimization = False
            assert self.recon_w > 0 and self.compact_w > 0 and self.contrastive_w > 0 and self.lap_w > 0
            self.train_loss = self._novel_cds_pixel_train
            self.val_loss = self._novel_cds_pixel_val
        else:
            raise NotImplementedError


    def _make_all_windows_q_pad_zero(self, Q_p, b_idx, c_idx, n_idx):
        Qp = pf.pad(Q_p, pad=(self.stride, self.stride, self.stride, self.stride)) # adding block of padding all four sides
        Q_blocks = rearrange(Qp, 'b c (n1 p1) (n2 p2) -> b c (n1 n2) p1 p2', p1=self.stride, p2=self.stride) #[b,9, num_blocks, stride * stride]
        # Gather windows: shape [b,num_sp,9,stride,stride]
        windows = Q_blocks[b_idx, c_idx, n_idx]
        #Tile the 9 patches into one 48×48 window each
        node_values = rearrange(
            windows,
            'b n (hb wb) ph pw -> b n (hb ph) (wb pw)',
            hb=3, wb=3,
            ph=self.stride, pw=self.stride
        )
        # Each window is the 9 blocks (3x3) so the shape here is [b, num_sp,9,16,16] where the num_sp is the id of the superpixel we are at
        # meaning the center block of the 3x3 blocks. every value in the window is the probability of assignment to that superpixel. so a pixel will be all the windows (superpixels)
        # that it could be assigned to.
        return node_values

    def _make_all_windows_gt(self,one_hot_all, padded_num_sp ):
        oh_p = pf.pad(one_hot_all, pad=(self.stride,) * 4)  # [b,50,H+2s,W+2s]

        # tile into non-overlapping stride×stride blocks
        # [b,50, n_blocks, stride, stride]
        Q_blocks = rearrange(
            oh_p,
            'b c (nh ph) (nw pw) -> b c (nh nw) ph pw',
            ph=self.stride, pw=self.stride
        )

        # [b, n_blocks, 50, stride, stride]
        blocks = Q_blocks.permute(0, 2, 1, 3, 4)  # [b, n_blocks, C, ph, pw]

        block_ids = calc_inner_sps_ids(padded_num_sp=padded_num_sp,device=self.device)
        # gather the 9 patches for each of num_sp windows
        # [b, num_sp, 9, C, ph, pw]
        windows = blocks[:, block_ids, :, :, :]

        # tile the 3×3 grid (9) into one 3s×3s window along the spatial dims
        windows_all_gt = rearrange(
            windows,
            'b n (hb wb) c ph pw -> b n c (hb ph) (wb pw)',
            hb=3, wb=3,
            ph=self.stride, pw=self.stride
        )

        return windows_all_gt


    def _trace_loss(self, windows_qp):
        traces = trace_laplacian_all_windows(node_values=windows_qp)
        trace_normalized = traces / 17860
        return 1 - torch.mean(trace_normalized)


    def _scn_pos_loss_pool_sum(self, Q9_p, feat_to_recon):
        """
        Code modified from https://github.com/fuy34/superpixel_fcn/blob/master/loss.py
        :param Q9_p: Assigment probabilities
        :param feat_to_recon: the position coords
        :return:
        """
        pooled_labxy = poolfeat(feat_to_recon, Q9_p, self.stride, self.stride)
        reconstr_feat = upfeat(pooled_labxy, Q9_p, self.stride, self.stride)
        loss_map= reconstr_feat-feat_to_recon
        loss_pos = torch.norm(loss_map, p=2, dim=1).sum()/self.batch_size
        return loss_pos


    def _scn_recon_loss_pool_sum(self, Q9_p, feat_to_recon):
        """
        Code modified from https://github.com/fuy34/superpixel_fcn/blob/master/loss.py
        :param Q9_p: Assigment probabilities
        :param feat_to_recon: the gt labels
        :return:
        """
        pooled_labxy = poolfeat(feat_to_recon, Q9_p, self.stride, self.stride)
        reconstr_feat = upfeat(pooled_labxy, Q9_p, self.stride, self.stride)
        logit = torch.log(reconstr_feat + 1e-8)
        loss_sem = - torch.sum(logit * feat_to_recon)/ self.batch_size
        return loss_sem

    def _scn_weighted_recon_loss_pool_sum(self, Q9_p, feat_to_recon, num_sp_one_d):
        one_hot_group = rearrange(feat_to_recon, 'b c (n1 p1) (n2 p2) -> b c (n1 n2) (p1 p2)', p1=self.stride,
                                  p2=self.stride)


        # A class is present in a block if it has at least one nonzero pixel
        class_present = one_hot_group.sum(dim=-1) > 0  # [B, 50, num_sp]
        num_classes_per_block = class_present.sum(dim=1)  # [B, 400]

        # Step 2: Create a mask for blocks with a single class
        single_class_mask = num_classes_per_block == 1  # [B, 400]

        # Step 3: Broadcast to [B, 400, 256]
        weights = torch.where(
            single_class_mask.unsqueeze(-1),  # [B, 400, 1]
            torch.full_like(one_hot_group[:, 0], 0.1),  # 0.1 if single class
            torch.full_like(one_hot_group[:, 0], 1.0)  # 2.0 otherwise
        )

        # weights: [B, 400, 256] — flat within each 16x16 block
        # Step 1: reshape 256 → 16×16
        block_weights = rearrange(weights, 'b (h w) (p1 p2) -> b h w p1 p2', h=num_sp_one_d, w=num_sp_one_d, p1=self.stride, p2=self.stride)
        # Now shape is [B, 20, 20, 16, 16]

        # Step 2: rearrange to full image grid
        weight_map = rearrange(block_weights, 'b h w p1 p2 -> b (h p1) (w p2)')


        pooled_labxy = poolfeat(feat_to_recon, Q9_p, self.stride, self.stride)
        reconstr_feat = upfeat(pooled_labxy, Q9_p, self.stride, self.stride)
        logit = torch.log(reconstr_feat + 1e-8)

        loss_sem_pre_pixel_ce = - torch.sum(logit * feat_to_recon, dim=1) #sum og scn style

        w = weight_map.detach()
        B, H, W = w.shape
        den = w.flatten(1).sum(1)
        alpha = (H * W) / den
        w_norm = w * alpha.view(B, 1, 1)
        loss_sem = (loss_sem_pre_pixel_ce * w_norm).sum() / B

        return loss_sem



    @torch.no_grad()
    def _calc_asa_approx(self, sp_ids, Q_9, flat_gt, img_size, num_sp):
        curr_spixl_map = update_spixl_map(sp_ids, Q_9, device=Q_9.device)
        # ASA calc
        Q_hard_assign_id = rearrange(curr_spixl_map, ' b 1 h w -> b (h w)')
        mean_asa = compute_asa(assign_hard=Q_hard_assign_id, flattened_gt=flat_gt, enfoce_con=False, img_size=img_size, num_sp=num_sp)
        return mean_asa


    def _baseline_scn_loss(self,batch, coords_type, sps_ids, img_height,total_num_sp):
        """
        This is the loss used by SCN: https://github.com/fuy34/superpixel_fcn/blob/master/loss.py The code has been organized differently.
        """
        img, gt, flat_gt, feat_label = batch
        Q_9, pix_embed = self.model(img)
        recon_loss = self._scn_recon_loss_pool_sum(Q9_p=Q_9, feat_to_recon=feat_label)*self.recon_w * 0.005
        pos_loss = self._scn_pos_loss_pool_sum(Q9_p=Q_9, feat_to_recon=coords_type) * self.compact_w * 0.005
        combined_loss = recon_loss + pos_loss

        mean_asa = self._calc_asa_approx(sp_ids=sps_ids, Q_9=Q_9, flat_gt=flat_gt,
                                         img_size=img_height,
                                         num_sp=total_num_sp)

        return combined_loss, recon_loss, pos_loss, 0.0, 0.0, 0.0, 0.0, mean_asa

    def _baseline_ssm_train_loss(self,batch, coords_type, sps_ids, img_height,total_num_sp):
        """
        Code modified from https://github.com/jiaxhm/SSMamba/tree/main
        """
        img, gt, flat_gt, feat_label, edge = batch
        Q_9, pred_edge, pix_embed = self.model(img)
        recon_loss = self._scn_recon_loss_pool_sum(Q9_p=Q_9, feat_to_recon=feat_label)*self.recon_w * 0.005
        pos_loss = self._scn_pos_loss_pool_sum(Q9_p=Q_9, feat_to_recon=coords_type) * self.compact_w * 0.005
        edge_loss = cross_entropy_loss_edge(pred_edge, edge)

        combined_loss = recon_loss + pos_loss + edge_loss

        mean_asa = self._calc_asa_approx(sp_ids=sps_ids, Q_9=Q_9, flat_gt=flat_gt,
                                         img_size=img_height,
                                         num_sp=total_num_sp)

        return combined_loss, recon_loss, pos_loss, 0.0, 0.0, edge_loss, 0.0, mean_asa

    def _novel_ssm_train_loss(self, batch, coords_type,b_idx, c_idx, n_idx,num_sp_ond_type, sps_ids, img_height,total_num_sp):
        img, gt, flat_gt, feat_label, edge = batch
        Q_9, pred_edge, pix_embed = self.model(img)
        recon_loss = self._scn_weighted_recon_loss_pool_sum(Q9_p=Q_9, feat_to_recon=feat_label, num_sp_one_d=num_sp_ond_type)*self.recon_w*0.005
        pos_loss = self._scn_pos_loss_pool_sum(Q9_p=Q_9, feat_to_recon=coords_type) * self.compact_w * 0.005
        edge_loss = cross_entropy_loss_edge(pred_edge, edge)

        windows_qp = self._make_all_windows_q_pad_zero(Q_9, b_idx, c_idx, n_idx)
        lap_loss = self.lap_w * self._trace_loss(windows_qp=windows_qp)
        contrastive_loss = self.contrastive_w * self._contrastive_loss_random_sample_pairs(pixel_embeddings=pix_embed,
                                                                                           segmentation_labels=gt)

        combined_loss = recon_loss + pos_loss + edge_loss + lap_loss + contrastive_loss

        mean_asa = self._calc_asa_approx(sp_ids=sps_ids, Q_9=Q_9, flat_gt=flat_gt,
                                         img_size=img_height,
                                         num_sp=total_num_sp)

        return combined_loss, recon_loss, pos_loss, 0.0, contrastive_loss, edge_loss, lap_loss, mean_asa


    def _novel_ssm_val_loss(self, batch, coords_type,b_idx, c_idx, n_idx,num_sp_ond_type, sps_ids, img_height,total_num_sp):
        """
        This is the same as the SCN loss
        """
        img, gt, flat_gt, feat_label, edge = batch
        Q_9, pix_embed = self.model(img)
        recon_loss = self._scn_weighted_recon_loss_pool_sum(Q9_p=Q_9, feat_to_recon=feat_label, num_sp_one_d=num_sp_ond_type)*self.recon_w*0.005
        pos_loss = self._scn_pos_loss_pool_sum(Q9_p=Q_9, feat_to_recon=coords_type) * self.compact_w * 0.005
        windows_qp = self._make_all_windows_q_pad_zero(Q_9, b_idx, c_idx, n_idx)
        lap_loss = self.lap_w * self._trace_loss(windows_qp=windows_qp)
        contrastive_loss = self.contrastive_w * self._contrastive_loss_random_sample_pairs(pixel_embeddings=pix_embed,
                                                                                           segmentation_labels=gt)

        combined_loss = recon_loss + pos_loss + lap_loss + contrastive_loss

        mean_asa = self._calc_asa_approx(sp_ids=sps_ids, Q_9=Q_9, flat_gt=flat_gt,
                                         img_size=img_height,
                                         num_sp=total_num_sp)

        return combined_loss, recon_loss, pos_loss, 0.0, contrastive_loss, 0.0, lap_loss, mean_asa

    def _baseline_ssm_val_loss(self,batch, coords_type, sps_ids, img_height,total_num_sp):
        """
        Code modified from https://github.com/jiaxhm/SSMamba/tree/main

        This is the same as the SCN loss
        """

        img, gt, flat_gt, feat_label, edge = batch
        Q_9, pix_embed = self.model(img)
        recon_loss = self._scn_recon_loss_pool_sum(Q9_p=Q_9, feat_to_recon=feat_label)*self.recon_w * 0.005
        pos_loss = self._scn_pos_loss_pool_sum(Q9_p=Q_9, feat_to_recon=coords_type) * self.compact_w * 0.005

        combined_loss = recon_loss + pos_loss

        mean_asa = self._calc_asa_approx(sp_ids=sps_ids, Q_9=Q_9, flat_gt=flat_gt,
                                         img_size=img_height,
                                         num_sp=total_num_sp)

        return combined_loss, recon_loss, pos_loss, 0.0, 0.0, 0.0, 0.0, mean_asa

    def _novel_scn_loss(self, batch, coords_type,b_idx, c_idx, n_idx,num_sp_ond_type, sps_ids, img_height,total_num_sp):
        img, gt, flat_gt, feat_label = batch
        Q_9, pix_embed = self.model(img)
        recon_loss = self._scn_weighted_recon_loss_pool_sum(Q9_p=Q_9, feat_to_recon=feat_label, num_sp_one_d=num_sp_ond_type)*self.recon_w*0.005
        pos_loss = self._scn_pos_loss_pool_sum(Q9_p=Q_9, feat_to_recon=coords_type) * self.compact_w * 0.005
        windows_qp = self._make_all_windows_q_pad_zero(Q_9, b_idx, c_idx, n_idx)
        lap_loss = self.lap_w * self._trace_loss(windows_qp=windows_qp)
        contrastive_loss = self.contrastive_w * self._contrastive_loss_random_sample_pairs(pixel_embeddings=pix_embed,
                                                                                           segmentation_labels=gt)
        combined_loss = recon_loss + pos_loss +lap_loss + contrastive_loss

        mean_asa = self._calc_asa_approx(sp_ids=sps_ids, Q_9=Q_9, flat_gt=flat_gt,
                                         img_size=img_height,
                                         num_sp=total_num_sp)
        return combined_loss, recon_loss, pos_loss, 0.0,contrastive_loss, 0.0, lap_loss, mean_asa


    def _novel_cds_pixel_train(self, batch, coords_type, b_idx, c_idx, n_idx,num_sp_ond_type, sps_ids, img_height,total_num_sp ):
        opt_model, opt_mi = self.optimizers()
        img, gt, flat_gt, feat_label, sob = batch

        Q_9, prob_assit, align, mi, pix_embed = self.model(img, sob)

        # ====== MODEL STEP (update model only) ======
        # freeze MI params so sample_loss backprops only into the model
        self.mi_estimator.eval()
        for p in self.mi_estimator.parameters():
            p.requires_grad_(False)
        # recon_loss = self._scn_recon_loss_pool_sum(Q9_p=Q_9, feat_to_recon=feat_label) * self.recon_w * 0.005
        recon_loss = self._scn_weighted_recon_loss_pool_sum(Q9_p=Q_9, feat_to_recon=feat_label, num_sp_one_d=num_sp_ond_type)*self.recon_w * 0.005
        pos_loss = self._scn_pos_loss_pool_sum(Q9_p=Q_9, feat_to_recon=coords_type) * self.compact_w * 0.005
        combined_loss_slic = recon_loss + pos_loss

        recon_loss_assist = self._scn_recon_loss_pool_sum(Q9_p=prob_assit, feat_to_recon=feat_label)*self.recon_w * 0.005 # the assit always used the standard loss
        pos_loss_assist = self._scn_pos_loss_pool_sum(Q9_p=prob_assit, feat_to_recon=coords_type) * self.compact_w * 0.005
        combined_loss_assist =recon_loss_assist + pos_loss_assist
        edge_loss = self.mi_estimator(mi[0].squeeze(), mi[1].squeeze())

        ###### Novel stuff
        windows_qp = self._make_all_windows_q_pad_zero(Q_9, b_idx=b_idx, c_idx=c_idx, n_idx=n_idx)
        lap_loss = self.lap_w * self._trace_loss(windows_qp=windows_qp)
        contrastive_loss = self.contrastive_w * self._contrastive_loss_random_sample_pairs(pixel_embeddings=pix_embed,
                                                                                           segmentation_labels=gt)



        align_loss = compute_alignment_loss(align)

        combined_loss = combined_loss_slic + combined_loss_assist + edge_loss + align_loss + lap_loss + contrastive_loss #  all losses
        # combined_loss = combined_loss_slic + combined_loss_assist + edge_loss + align_loss + lap_loss #  no cl
        # combined_loss = combined_loss_slic + combined_loss_assist + edge_loss + align_loss + contrastive_loss # no lap
        # combined_loss = combined_loss_slic + combined_loss_assist + edge_loss + align_loss # no lap and no cl
        self.manual_backward(combined_loss)

        opt_model.step()
        opt_model.zero_grad()

        # ----- scheduler step: do it *after* optimizer.step() -----
        # If you returned exactly one scheduler for the model in configure_optimizers,
        # this gets it in a backend-safe way:
        scheds = self.lr_schedulers()  # can be a single scheduler or a list
        if not isinstance(scheds, (list, tuple)):
            scheds = [scheds]
        for s in scheds:
                s.step()

        mean_asa = self._calc_asa_approx(sp_ids=sps_ids, Q_9=Q_9, flat_gt=flat_gt,
                                         img_size=img_height,
                                         num_sp=total_num_sp)

        # ====== MI STEP (update MI only) ======
        # unfreeze MI, but DETACH features so model isn't updated
        self.mi_estimator.train()
        for p in self.mi_estimator.parameters():
            p.requires_grad_(True)
        mi_loss = self.mi_estimator.learning_loss(mi[0].squeeze().detach(), mi[1].squeeze().detach())
        self.manual_backward(mi_loss)
        opt_mi.step()
        opt_mi.zero_grad()

        # This is an hack fix. use this as the key to plots in the file
        return combined_loss, recon_loss, pos_loss, edge_loss,contrastive_loss, mi_loss, lap_loss, mean_asa # all losses
        # return combined_loss, recon_loss, pos_loss, edge_loss,0.0, mi_loss, lap_loss, mean_asa # no cl
        # return combined_loss, recon_loss, pos_loss, edge_loss,contrastive_loss, mi_loss, 0.0, mean_asa # no lap
        # return combined_loss, recon_loss, pos_loss, edge_loss,0.0, mi_loss, 0.0, mean_asa # no lap no cl



    def _baseline_cds_pixel_train(self, batch, coords_type, sps_ids, img_height,total_num_sp ):
        """
        Code modified from https://github.com/rookiie/CDSpixel/tree/main
        """
        opt_model, opt_mi = self.optimizers()
        img, gt, flat_gt, feat_label, sob = batch

        Q_9, prob_assit, align, mi, pix_embed = self.model(img, sob)
        # ====== MODEL STEP (update model only) ======
        # freeze MI params so sample_loss only updates the model
        self.mi_estimator.eval()
        for p in self.mi_estimator.parameters():
            p.requires_grad_(False)
        recon_loss = self._scn_recon_loss_pool_sum(Q9_p=Q_9, feat_to_recon=feat_label)*self.recon_w * 0.005
        pos_loss = self._scn_pos_loss_pool_sum(Q9_p=Q_9, feat_to_recon=coords_type) * self.compact_w * 0.005
        combined_loss_slic = recon_loss + pos_loss

        recon_loss_assist = self._scn_recon_loss_pool_sum(Q9_p=prob_assit, feat_to_recon=feat_label)*self.recon_w * 0.005
        pos_loss_assist = self._scn_pos_loss_pool_sum(Q9_p=prob_assit, feat_to_recon=coords_type) * self.compact_w * 0.005
        combined_loss_assist =recon_loss_assist + pos_loss_assist
        edge_loss = self.mi_estimator(mi[0].squeeze(), mi[1].squeeze())


        align_loss = compute_alignment_loss(align)

        combined_loss = combined_loss_slic + combined_loss_assist + edge_loss + align_loss
        self.manual_backward(combined_loss)

        opt_model.step()
        opt_model.zero_grad()

        # ----- scheduler step -----
        # If you returned exactly one scheduler for the model in configure_optimizers,
        scheds = self.lr_schedulers()  # can be a single scheduler or a list
        if not isinstance(scheds, (list, tuple)):
            scheds = [scheds]
        for s in scheds:
                s.step()

        mean_asa = self._calc_asa_approx(sp_ids=sps_ids, Q_9=Q_9, flat_gt=flat_gt,
                                         img_size=img_height,
                                         num_sp=total_num_sp)

        # ====== MI STEP (update MI only) ======
        # unfreeze MI, but DETACH features so model isn't updated
        self.mi_estimator.train()
        for p in self.mi_estimator.parameters():
            p.requires_grad_(True)
        mi_loss = self.mi_estimator.learning_loss(mi[0].squeeze().detach(), mi[1].squeeze().detach())
        self.manual_backward(mi_loss)
        opt_mi.step()
        opt_mi.zero_grad()

        return combined_loss, recon_loss, pos_loss, edge_loss,align_loss, mi_loss, 0.0, mean_asa

    def _baseline_cds_pixel_val(self, batch, coords_type, sps_ids, img_height,total_num_sp ):
        """
        Code modified from https://github.com/rookiie/CDSpixel/tree/main

        Same as SCN
        """
        img, gt, flat_gt, feat_label, sob = batch
        Q_9, pix_embed = self.model(img, sob)

        recon_loss = self._scn_recon_loss_pool_sum(Q9_p=Q_9, feat_to_recon=feat_label)*self.recon_w * 0.005
        pos_loss = self._scn_pos_loss_pool_sum(Q9_p=Q_9, feat_to_recon=coords_type) * self.compact_w * 0.005
        combined_loss_slic = recon_loss + pos_loss


        mean_asa = self._calc_asa_approx(sp_ids=sps_ids, Q_9=Q_9, flat_gt=flat_gt,
                                         img_size=img_height,
                                         num_sp=total_num_sp)

        return combined_loss_slic, recon_loss, pos_loss, 0.0,0.0, 0.0, 0.0, mean_asa


    def _novel_cds_pixel_val(self, batch, coords_type, sps_ids, img_height,total_num_sp, b_idx, c_idx,n_idx,num_sp_ond_type):
        img, gt, flat_gt, feat_label, sob = batch
        Q_9, pix_embed = self.model(img, sob)

        recon_loss = self._scn_weighted_recon_loss_pool_sum(Q9_p=Q_9, feat_to_recon=feat_label,num_sp_one_d=num_sp_ond_type )*self.recon_w * 0.005
        pos_loss = self._scn_pos_loss_pool_sum(Q9_p=Q_9, feat_to_recon=coords_type) * self.compact_w * 0.005

        windows_qp = self._make_all_windows_q_pad_zero(Q_9, b_idx=b_idx, c_idx=c_idx, n_idx=n_idx)
        lap_loss = self.lap_w * self._trace_loss(windows_qp=windows_qp)
        contrastive_loss = self.contrastive_w * self._contrastive_loss_random_sample_pairs(pixel_embeddings=pix_embed,
                                                                                           segmentation_labels=gt)

        combined_loss_slic = recon_loss + pos_loss + lap_loss + contrastive_loss
        mean_asa = self._calc_asa_approx(sp_ids=sps_ids, Q_9=Q_9, flat_gt=flat_gt,
                                         img_size=img_height,
                                         num_sp=total_num_sp)

        return combined_loss_slic, recon_loss, pos_loss, 0.0,contrastive_loss, 0.0, lap_loss, mean_asa


    def _baseline_ainet_loss(self,batch, coords_type, sps_ids, img_height,total_num_sp):
        """
        Code modified from https://github.com/YanFangCS/AINET/blob/main/loss.py
        """
        img, gt, flat_gt, feat_label, patch_posi, patch_label = batch
        Q_9, pix_embed = self.model(img)
        recon_loss = self._scn_recon_loss_pool_sum(Q9_p=Q_9, feat_to_recon=feat_label)*self.recon_w * 0.005
        pos_loss = self._scn_pos_loss_pool_sum(Q9_p=Q_9, feat_to_recon=coords_type) * self.compact_w * 0.005

        bdl=0.0
        if self.current_epoch >= 0:
            bdl = self._ainet_boundary_perceiving_loss(pix_embed, patch_posi, patch_label) * 0.005
            combined_loss = recon_loss + pos_loss + 0.5 * bdl
        else:
            combined_loss = recon_loss + pos_loss

        mean_asa = self._calc_asa_approx(sp_ids=sps_ids, Q_9=Q_9, flat_gt=flat_gt,
                                         img_size=img_height,
                                         num_sp=total_num_sp)

        return combined_loss, recon_loss, pos_loss, 0.0, 0.0, bdl, 0.0, mean_asa

    def _novel_ainet_loss(self,batch, coords_type, sps_ids, img_height,total_num_sp, num_sp_ond_type, b_idx, c_idx, n_idx):
        img, gt, flat_gt, feat_label, patch_posi, patch_label = batch
        Q_9, pix_embed = self.model(img)
        recon_loss = self._scn_weighted_recon_loss_pool_sum(Q9_p=Q_9, feat_to_recon=feat_label,
                                               num_sp_one_d=num_sp_ond_type) * self.recon_w * 0.005
        pos_loss = self._scn_pos_loss_pool_sum(Q9_p=Q_9, feat_to_recon=coords_type) * self.compact_w * 0.005
        windows_qp = self._make_all_windows_q_pad_zero(Q_9, b_idx, c_idx, n_idx)
        lap_loss = self.lap_w * self._trace_loss(windows_qp=windows_qp)
        contrastive_loss = self.contrastive_w * self._contrastive_loss_random_sample_pairs(pixel_embeddings=pix_embed,
                                                                                           segmentation_labels=gt)

        bdl=0.0
        if self.current_epoch >= 0:
            bdl = self._ainet_boundary_perceiving_loss(pix_embed, patch_posi, patch_label) * 0.005
            combined_loss = recon_loss + pos_loss + 0.5 * bdl + lap_loss + contrastive_loss
        else:
            combined_loss = recon_loss + pos_loss +  lap_loss + contrastive_loss

        mean_asa = self._calc_asa_approx(sp_ids=sps_ids, Q_9=Q_9, flat_gt=flat_gt,
                                         img_size=img_height,
                                         num_sp=total_num_sp)

        return combined_loss, recon_loss, pos_loss, 0.0, contrastive_loss, bdl, lap_loss, mean_asa

    def _ainet_boundary_perceiving_loss(self,feat_map, patch_posi, patch_label):
        """
        Code taken from https://github.com/YanFangCS/AINET/blob/main/loss.py
        """
        bs, c, h, w = feat_map.shape
        patch_loss = torch.tensor([0.]).to(self.device)
        for i in range(bs):
            label = patch_label[i]
            patches = patch_posi[i]
            feat = feat_map[i]

            patch_num = patches.shape[0]
            patches_i = []
            labels_i = []
            for k in range(patch_num):
                patch = patches[k]
                patch_label_i = label[k]
                feat_patch = torch.narrow(feat, 1, patch[0], patch[1])
                feat_patch = torch.narrow(feat_patch, 2, patch[2], patch[3])
                patches_i.append(feat_patch)
                labels_i.append(patch_label_i)

            patch_stack = torch.stack(patches_i, dim=0)
            label_stack = torch.stack(labels_i, dim=0)
            patch_loss_i = self._ainet_patch_classify(patch_stack, label_stack)
            patch_loss += patch_loss_i

        return patch_loss / bs

    def _ainet_patch_classify(self,feat, label):
        """
        Code taken from https://github.com/YanFangCS/AINET/blob/main/loss.py
        """
        def simi_func(anchor_emb, emb):
            norm = torch.sum(torch.abs(anchor_emb - emb), dim=-1)
            simi = 2.0 / (1 + torch.exp(norm).clamp(min=1e-8, max=1e15))

            return simi

        # feat: c x h x w
        # label: 4 x h x w
        patch_num, c, h, w = feat.shape
        label_num = torch.sum(torch.sum(label, dim=-1), dim=-1)
        feat1_1 = feat * label[:, 0:1]
        feat1_1 = torch.sum(torch.sum(feat1_1, dim=-1), dim=-1) / (label_num[:, 0:1] + 1)

        feat1_2 = feat * label[:, 1:2]
        feat1_2 = torch.sum(torch.sum(feat1_2, dim=-1), dim=-1) / (label_num[:, 1:2] + 1)

        feat2_1 = feat * label[:, 2:3]
        feat2_1 = torch.sum(torch.sum(feat2_1, dim=-1), dim=-1) / (label_num[:, 2:3] + 1)

        feat2_2 = feat * label[:, 3:4]
        feat2_2 = torch.sum(torch.sum(feat2_2, dim=-1), dim=-1) / (label_num[:, 3:4] + 1)

        same_simi1 = simi_func(feat1_1, feat1_2)
        same_simi2 = simi_func(feat2_1, feat2_2)

        cross_simi1 = simi_func(feat1_1, feat2_1)
        cross_simi2 = simi_func(feat1_2, feat2_2)

        same_loss = -(torch.log(same_simi1 + 1e-8) + torch.log(same_simi2 + 1e-8)) / 2.
        cross_loss = -(torch.log(1 - cross_simi1 + 1e-8) + torch.log(1 - cross_simi2 + 1e-8)) / 2.

        return torch.mean(same_loss) + torch.mean(cross_loss)

    def configure_optimizers(self):

        if self.model_name == "scn":

            param_groups = [
                {'params': self.model.bias_parameters(), 'weight_decay': 0.0},
                {'params': self.model.weight_parameters(), 'weight_decay': self.wd}
            ]
            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.LR_decay_epoch, gamma=0.5)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
        elif self.model_name == "cds":
            param_groups = [
                {'params': self.model.bias_parameters(), 'weight_decay': 0.0},
                {'params': self.model.weight_parameters(), 'weight_decay': self.wd}
            ]
            print("For Model")
            print("the weight decay is ",self.wd)
            print("the lr is: ",self.lr)

            optimizer_model = torch.optim.Adam(
                param_groups,
                lr=self.lr)

            ###### For MI

            opt_mi = torch.optim.Adam(self.mi_estimator.parameters(), lr=5e-4)
            print("learning rate for optimizer mi is ", str(5e-4))

            ##### LR Poly (
            power = 0.9
            add_steps = 100_000
            last_milestone = 50_000
            max_iter = add_steps + last_milestone

            def poly_lambda(current_step: int):
                frac = 1.0 - current_step / max_iter
                frac = max(frac, 0.0)
                return frac ** power

            scheduler_model = optim.lr_scheduler.LambdaLR(
                optimizer_model,
                lr_lambda=[poly_lambda, poly_lambda]
            )

            scheduler_cfg = {
                "scheduler": scheduler_model,
                "interval": "step",  # step every batch
                "frequency": 1,
                "name": "poly_lr",
            }

            return [optimizer_model, opt_mi], [scheduler_cfg]

        elif self.model_name == "ssm":
            param_groups = [
                {'params': self.model.bias_parameters(), 'weight_decay': 0.0},
                {'params': self.model.weight_parameters(), 'weight_decay': self.wd}
            ]

            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.lr)

            ##### LR Poly (
            power = 0.9
            add_steps = 100_000
            # last_milestone = 200_000
            last_milestone = 50_000
            max_iter = add_steps + last_milestone

            def poly_lambda(current_step: int):
                frac = 1.0 - current_step / max_iter
                frac = max(frac, 0.0)
                return frac ** power

            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=[poly_lambda, poly_lambda]
            )

            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
        elif self.model_name == "ainet":

            param_groups = [
                {'params': self.model.bias_parameters(), 'weight_decay': 0.0, 'lr': self.lr},
                {'params': self.model.weight_parameters(), 'weight_decay': self.wd, 'lr': self.lr}]

            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.LR_decay_epoch,
                                                  gamma=0.5)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
        else:
            raise NotImplementedError



    def training_step(self, batch, batch_idx):
        """
        You need to have only one self.train_loss uncommented. Are you doing baseline models or novel?
        """

        # Baseline
        # (combined_loss, reconstruct_loss_val, compact_loss, otherone_loss_val,
        #  contrastive_loss, othertwo_loss_val, lap_loss, mean_asa)= self.train_loss(batch=batch,coords_type= self.coords_train,
        #                                                                        sps_ids=self.train_sp_ids, img_height=self.train_height,total_num_sp=self.total_train_num_sp)

        # Novel
        (combined_loss, reconstruct_loss_val, compact_loss, otherone_loss_val,
         contrastive_loss, othertwo_loss_val, lap_loss, mean_asa)= self.train_loss(batch=batch,coords_type=self.coords_train,
                                                                               b_idx=self.b_idx_train, c_idx=self.c_idx_train,
                                                                               n_idx=self.n_idx_train,num_sp_ond_type=self.train_num_sp, sps_ids=self.train_sp_ids, img_height=self.train_height,total_num_sp=self.total_train_num_sp)


        self.log("train_loss", combined_loss, prog_bar=False, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log("train_recon_loss", reconstruct_loss_val, prog_bar=False, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log("train_contrastive_loss", contrastive_loss, prog_bar=False, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log("train_othertwo_loss", othertwo_loss_val, prog_bar=False, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log("train_otherone_loss", otherone_loss_val, prog_bar=False, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log("train_pos_loss", compact_loss, prog_bar=False, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log("train_lap", lap_loss, prog_bar=False, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log("train_mean_asa", mean_asa, prog_bar=False, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)

        return combined_loss

    def validation_step(self, batch, batch_idx):
        """
        You need to have only one self.val_loss uncommented. Are you doing baseline models or novel?
        """


        # Baseline SCN and CDS and AInet
        # (combined_loss, reconstruct_loss_val, compact_loss, otherone_loss_val,
        #  contrastive_loss, othertwo_loss_val, lap_loss, mean_asa)= self.val_loss(batch=batch,coords_type= self.coords_val,
        #                                                                        sps_ids=self.val_sp_ids, img_height=self.val_height,total_num_sp=self.total_val_num_sp)
        #
        # SCN and CDS novel
        (combined_loss, reconstruct_loss_val, compact_loss, otherone_loss_val,
         contrastive_loss, othertwo_loss_val, lap_loss, mean_asa)= self.val_loss(batch=batch,coords_type=self.coords_val,
                                                                               b_idx=self.b_idx_val, c_idx=self.c_idx_val,
                                                                               n_idx=self.n_idx_val,num_sp_ond_type=self.val_num_sp, sps_ids=self.val_sp_ids, img_height=self.val_height,total_num_sp=self.total_val_num_sp)


        self.log("val_loss", combined_loss, prog_bar=False, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log("val_recon_loss", reconstruct_loss_val, prog_bar=False, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log("val_contrastive_loss", contrastive_loss, prog_bar=False, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log("val_othertwo_loss", othertwo_loss_val, prog_bar=False, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log("val_otherone_loss", otherone_loss_val, prog_bar=False, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log("val_pos_loss", compact_loss, prog_bar=False, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log("val_lap", lap_loss, prog_bar=False, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        self.log("val_mean_asa", mean_asa, prog_bar=False, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)

        return combined_loss

"""
SIN code located here: https://github.com/yuanqqq/SIN

This code has been slightly modified.
"""
from torch.nn.init import kaiming_normal_, constant_

import torch
import torch.nn as nn
import torch.nn.functional as F

# *************************** my functions ****************************
def predict_param(in_planes, channel=3):
    return nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)


def predict_mask(in_planes, channel=9):
    return nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)



def predict_feat(in_planes, channel=20, stride=1):
    return nn.Conv2d(in_planes, channel, kernel_size=3, stride=stride, padding=1, bias=True)


def predict_prob(in_planes, channel=9):
    return nn.Sequential(
        nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True),
        nn.Softmax(1)
    )

# ***********************************************************************


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.LeakyReLU(0.1)
        )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1)
    )


def deconv_h(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=(1, 2), padding=(1, 1), bias=True),
        nn.LeakyReLU(0.1)
    )


def deconv_v(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=(2, 1), padding=(1, 1), bias=True),
        nn.LeakyReLU(0.1)
    )


def index_conv(x):
    # kernel_1 = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],\
    #                        [[0, 0, 0], [1, 0, 0], [0, 0, 0]],\
    #                        [[0, 0, 0], [0, 0, 0], [0, 1, 0]],\
    #                        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],\
    #                        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],\
    #                        [[0, 0, 0], [0, 0, 0], [1, 0, 0]],\
    #                        [[0, 0, 0], [0, 0, 0], [0, 0, 1]],\
    #                        [[0, 0, 1], [0, 0, 0], [0, 0, 0]],\
    #                        [[1, 0, 0], [0, 0, 0], [0, 0, 0]]]
    #                       )
    kernel = torch.tensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],\
                           [[0, 0, 1], [0, 0, 0], [0, 0, 0]],\
                           [[0, 0, 0], [0, 0, 0], [0, 0, 1]],\
                           [[0, 0, 0], [0, 0, 0], [1, 0, 0]],\
                           [[0, 1, 0], [0, 0, 0], [0, 0, 0]],\
                           [[0, 0, 0], [0, 0, 1], [0, 0, 0]],\
                           [[0, 0, 0], [0, 0, 0], [0, 1, 0]],\
                           [[0, 0, 0], [1, 0, 0], [0, 0, 0]],\
                           [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

    # kernel = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],\
    #                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],\
    #                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],\
    #                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],\
    #                        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],\
    #                        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],\
    #                        [[0, 0, 0], [0, 0, 0], [0, 1, 0]],\
    #                        [[0, 0, 0], [1, 0, 0], [0, 0, 0]],\
    #                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

    kernel = kernel.float().to(x.device)
    bz, c, h, w = x.shape
    # for odd size
    output = F.conv_transpose2d(x.float(), kernel.unsqueeze(0), stride=2, padding=0, output_padding=0)
    return output


def initialize_map(x):
    bz, c, h, w = x.shape
    h = (h+15)//16
    w = (w+15)//16
    device = x.device
    start_id = 1
    end_id = start_id + h*w
    map = torch.arange(start_id, end_id).reshape(h, w).float()
    batch_map = map.repeat(bz, 1, 1, 1).to(device)
    return batch_map


def update_map(prob, map):
    # prob: bz*9*h*w
    # map: bz*1*h'*w'
    map_ = index_conv(map)
    bz, c, h, w = prob.shape
    device = prob.device
    # map_one = (map_ != 0)
    map_one = -F.relu(-map_ + 1) + 1
    prob = prob * map_one
    index_map = torch.arange(0, c).reshape(1, c, 1, 1).to(device)
    max_prob, max_id = prob.max(dim=1, keepdim=True)
    assignment = F.relu(prob - max_prob - (index_map - max_id) * (index_map - max_id) + 1)

    # temp = torch.arange(0, c)
    # temp = temp.repeat(bz, h, w, 1).permute(0, 3, 1, 2).to(device)
    # assignment = torch.where(temp == max_id, torch.ones(bz, c, h, w).to(device), torch.zeros(bz, c, h, w).to(device))
    new_map_ = assignment.float() * map_
    new_map = torch.sum(new_map_, dim=1, keepdim=True)
    return prob, new_map


def update_h_map(prob, map):
    b, c, h, w = map.shape
    device = prob.device
    lr_map = map
    lr_map = F.interpolate(lr_map, (h, 2*w), mode='nearest')
    # lr_map = F.pad(lr_map, (1, 1, 0, 0), mode='replicate')
    left_p = lr_map[:, :, :, :-1]
    right_p = lr_map[:, :, :, 1:]
    lr_map = torch.cat((left_p, right_p), dim=1)

    index_map = torch.arange(0, 2).reshape(1, 2, 1, 1).to(device)
    max_prob, max_id = prob.max(dim=1, keepdim=True)
    assignment = F.relu(prob - max_prob - (index_map - max_id) * (index_map - max_id) + 1)
    new_map_ = assignment.float() * lr_map
    new_map = torch.sum(new_map_, dim=1, keepdim=True)
    return new_map


def update_v_map(prob, map):
    b, c, h, w = map.shape
    device = prob.device
    tb_map = map
    tb_map = F.interpolate(tb_map, (2*h, w), mode='nearest')
    # tb_map = F.pad(tb_map, (0, 0, 1, 1), mode='replicate')
    top_p = tb_map[:, :, :-1, :]
    bott_p = tb_map[:, :, 1:, :]
    tb_map = torch.cat((top_p, bott_p), dim=1)

    index_map = torch.arange(0, 2).reshape(1, 2, 1, 1).to(device)
    max_prob, max_id = prob.max(dim=1, keepdim=True)
    assignment = F.relu(prob - max_prob - (index_map - max_id) * (index_map - max_id) + 1)
    new_map_ = assignment.float() * tb_map
    new_map = torch.sum(new_map_, dim=1, keepdim=True)
    return new_map


def update_spixel_map_sin(img, prob0_v, prob0_h, prob1_v, prob1_h, prob2_v, prob2_h, prob3_v, prob3_h):
    initial_map = initialize_map(img)
    map3_h = update_h_map(prob3_h, initial_map)
    map3_v = update_v_map(prob3_v, map3_h)
    map2_h = update_h_map(prob2_h, map3_v)
    map2_v = update_v_map(prob2_v, map2_h)
    map1_h = update_h_map(prob1_h, map2_v)
    map1_v = update_v_map(prob1_v, map1_h)
    map0_h = update_h_map(prob0_h, map1_v)
    map0_v = update_v_map(prob0_v, map0_h)

    return map0_v



class SpixelNet(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(SpixelNet,self).__init__()

        self.batchNorm = batchNorm
        self.assign_ch = 9

        self.conv0a = conv(self.batchNorm, 3, 16, kernel_size=3, padding=1)
        self.conv0b = conv(self.batchNorm, 16, 16, kernel_size=3, padding=1)

        self.conv1a = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
        self.conv1b = conv(self.batchNorm, 32, 32, kernel_size=3, padding=1)

        self.conv2a = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        self.conv2b = conv(self.batchNorm, 64, 64, kernel_size=3, padding=1)

        self.conv3a = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv3b = conv(self.batchNorm, 128, 128, kernel_size=3, padding=1)

        self.conv4a = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv4b = conv(self.batchNorm, 256, 256, kernel_size=3, padding=1)

        self.deconv3 = deconv(256, 128)
        self.deconv3_h = deconv_h(256, 128)
        self.deconv3_v = deconv_v(128, 128)
        self.conv3_1 = conv(self.batchNorm, 256, 128, padding=1)
        self.pred_mask3 = predict_mask(128, self.assign_ch)
        self.pred_mask3_h = predict_mask(128, 2)
        self.pred_mask3_v = predict_mask(128, 2)

        self.deconv2 = deconv(128, 64)
        self.deconv2_h = deconv_h(128, 64)
        self.deconv2_v = deconv_v(64, 64)
        self.conv2_1 = conv(self.batchNorm, 128, 64, padding=1)
        self.pred_mask2 = predict_mask(64, self.assign_ch)
        self.pred_mask2_h = predict_mask(64, 2)
        self.pred_mask2_v = predict_mask(64, 2)

        self.deconv1 = deconv(64, 32)
        self.deconv1_h = deconv_h(64, 32)
        self.deconv1_v = deconv_v(32, 32)
        self.conv1_1 = conv(self.batchNorm, 64, 32, padding=1)
        self.pred_mask1 = predict_mask(32, self.assign_ch)
        self.pred_mask1_h = predict_mask(32, 2)
        self.pred_mask1_v = predict_mask(32, 2)

        self.deconv0 = deconv(32, 16)
        self.deconv0_h = deconv_h(32, 16)
        self.deconv0_v = deconv_v(16, 16)
        self.conv0_1 = conv(self.batchNorm, 32 , 16, padding=1)
        self.pred_mask0 = predict_mask(16, self.assign_ch)
        self.pred_mask0_h = predict_mask(16, 2)
        self.pred_mask0_v = predict_mask(16, 2)

        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out1 = self.conv0b(self.conv0a(x))  # 5*5
        out2 = self.conv1b(self.conv1a(out1))  # 11*11
        out3 = self.conv2b(self.conv2a(out2))  # 23*23
        out4 = self.conv3b(self.conv3a(out3))  # 47*47
        out5 = self.conv4b(self.conv4a(out4))  # 95*95

        out_deconv3_h = self.deconv3_h(out5)
        mask3_h = self.pred_mask3_h(out_deconv3_h)
        prob3_h = self.softmax(mask3_h)

        out_deconv3_v = self.deconv3_v(out_deconv3_h)
        mask3_v = self.pred_mask3_v(out_deconv3_v)
        prob3_v = self.softmax(mask3_v)

        out_deconv2_h = self.deconv2_h(out_deconv3_v)
        mask2_h = self.pred_mask2_h(out_deconv2_h)
        prob2_h = self.softmax(mask2_h)

        out_deconv2_v = self.deconv2_v(out_deconv2_h)
        mask2_v = self.pred_mask2_v(out_deconv2_v)
        prob2_v = self.softmax(mask2_v)

        out_deconv1_h = self.deconv1_h(out_deconv2_v)
        mask1_h = self.pred_mask1_h(out_deconv1_h)
        prob1_h = self.softmax(mask1_h)

        out_deconv1_v = self.deconv1_v(out_deconv1_h)
        mask1_v = self.pred_mask1_v(out_deconv1_v)
        prob1_v = self.softmax(mask1_v)

        out_deconv0_h = self.deconv0_h(out_deconv1_v)
        mask0_h = self.pred_mask0_h(out_deconv0_h)
        prob0_h = self.softmax(mask0_h)

        out_deconv0_v = self.deconv0_v(out_deconv0_h)
        mask0_v = self.pred_mask0_v(out_deconv0_v)
        prob0_v = self.softmax(mask0_v)

        return prob0_v, prob0_h, prob1_v, prob1_h, prob2_v, prob2_h, prob3_v, prob3_h

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]



def get_sin_model():
    mymodel = SpixelNet(batchNorm=True)
    return mymodel


if __name__ == '__main__':
    model= get_sin_model()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("the number of parameters in the model is ", pytorch_total_params)
    x_temp = torch.randn(2, 3, 208, 208, device='cpu')
    prob0_v, prob0_h, prob1_v, prob1_h, prob2_v, prob2_h, prob3_v, prob3_h = model(x_temp)
    print(prob0_v.shape)
"""
CDSpixel code located here: https://github.com/rookiie/CDSpixel

This code has been slightly modified.
"""
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_

## *************************** my functions ****************************

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

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=None):
    if padding is None:
        padding = (kernel_size - 1) // 2
    else:
        padding = padding
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
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1)
    )


class Embedder(nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()
        self.conv0a = conv(True, 3, 64, kernel_size=3)
        self.conv0b = conv(True, 64, 64, kernel_size=3)

        self.pool0 = nn.MaxPool2d(3, 2, 1)

        self.conv1a = conv(True, 64, 64, kernel_size=3)
        self.conv1b = conv(True, 64, 64, kernel_size=3)

        self.pool1 = nn.MaxPool2d(3, 2, 1)

        self.conv2a = conv(True, 64, 64, kernel_size=3)
        self.conv2b = conv(True, 64, 64, kernel_size=3)

        self.head0 = conv(True, 64 * 3, 32, kernel_size=3)
        self.head1 = nn.Sequential(
            nn.Conv2d(32 + 3, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        x0 = self.conv0b(self.conv0a(x))
        x1 = self.pool0(x0)
        x1 = self.conv1b(self.conv1a(x1))
        x2 = self.pool1(x1)
        x2 = self.conv2b(self.conv2a(x2))

        x1_up = F.interpolate(x1, scale_factor=2)
        x2_up = F.interpolate(x2, scale_factor=4)

        out = torch.cat([x0, x1_up, x2_up], 1)
        out = self.head0(out)
        out = self.head1(torch.cat([out, x], dim=1))
        return out


class water_diffusion(nn.Module):
    def __init__(self, dim=32):
        super(water_diffusion, self).__init__()
        self.conv_up = deconv(dim, dim)  # up ->2
        self.fusion = conv(True, dim * 2, dim, kernel_size=3)

    def forward(self, input, flow):
        flow = self.conv_up(flow)
        _, _, h, w = flow.shape
        inp = F.interpolate(input, size=(h, w), mode='bilinear')
        out = self.fusion(torch.cat([inp, flow], dim=1))
        return out


class Diffusion(nn.Module):
    def __init__(self, grid_size):
        super(Diffusion, self).__init__()
        self.diffusion_step = int(np.log2(grid_size))
        self.up = nn.ModuleList([water_diffusion() for i in range(self.diffusion_step)])

    def forward(self, x, spixel=None):
        b, c, h, w = x.shape
        if spixel is None:
            spixel = (h // 16, w // 16)
        flow = F.interpolate(x, size=spixel, mode='bilinear')
        for stage in self.up:
            flow = stage(x, flow)
        return flow


class Disentangle(nn.Module):
    def __init__(self, dim=32):  # when embedderv3, this is 64
        super(Disentangle, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(dim, dim, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(dim, dim, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        unique = x * (1 - y).expand_as(x)
        common = x * y.expand_as(x)

        return unique, common


class CDSpixelNet(nn.Module):
    expansion = 1

    def __init__(self, bn=True, grid_size=16, use_assist=True):
        super(CDSpixelNet, self).__init__()

        self.bn = bn
        self.assign_ch = 9
        self.use_assist = use_assist
        if self.use_assist and self.training:
            self.encoder_assit = Embedder()
            self.gap = nn.AdaptiveAvgPool2d((1))

        self.encoder = Embedder()
        self.MI = Disentangle()
        self.decoder = Diffusion(grid_size=grid_size)
        self.head = nn.Sequential(
            predict_mask(32, self.assign_ch),
            nn.Softmax(1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x, x_assit=None, spixel=None):
        if self.training:
            assert x_assit is not None, "Training use input with two modal."
            return self._forward_train(x, x_assit, spixel)
        else:
            return self._forward_inference(x, spixel)

    def _forward_train(self, x, x_assit=None, spixel=None):
        if self.use_assist:
            x_assit = self.encoder_assit(x_assit)
            x_assit, mi_assit = self.MI(x_assit)
            flow_assit = self.decoder(x_assit, spixel)
            prob_assit = self.head(flow_assit)

            x = self.encoder(x)
            x, mi = self.MI(x)
            flow = self.decoder(x, spixel)
            prob = self.head(flow)
            return prob, prob_assit, [x, x_assit], [self.gap(mi), self.gap(mi_assit)], flow
        else:
            raise Exception("Not tested.")
            x = self.encoder(x)
            flow = self.decoder(x, spixel)
            prob = self.head(flow)
            return prob

    def _forward_inference(self, x, spixel=None):
        x = self.encoder(x)
        if self.use_assist:
            x, _ = self.MI(x)
        flow = self.decoder(x, spixel)
        prob = self.head(flow)
        return prob, flow

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def get_cds_model():
    mymodel = CDSpixelNet(bn=True)
    return mymodel


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim]
    '''

    def __init__(self, x_dim, y_dim, hidden_size=8):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

if __name__ == '__main__':
    model= get_cds_model()
    model.eval()
    weight_path = "../weights/baselines/model_best_cdspixel.tar"
    weight_load = torch.load(weight_path, map_location=torch.device("cpu"))
    load_status = model.load_state_dict(weight_load['state_dict'], strict=False)
    if load_status.missing_keys or load_status.unexpected_keys:
        raise Exception(
            f"State dict loading mismatch:\n"
            f"Missing keys: {load_status.missing_keys}\n"
            f"Unexpected keys: {load_status.unexpected_keys}"
        )
    else:
        print("all keys loaded")
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("the number of parameters in the model is ", pytorch_total_params)
    x_temp = torch.randn(2, 3, 208, 208, device='cpu')
    out, other = model(x_temp)
    print(out.shape)
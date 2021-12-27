import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import codes.models.networks as networks
import codes.models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from .modules.loss import BorderLoss, FidelityLoss, GradientLoss, IsosurfaceSimilarityLoss
from .modules.Quantization import Quantization

logger = logging.getLogger('base')


class InvertibleDownsample(BaseModel):
    def __init__(self, opt):
        super(InvertibleDownsample, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt

        # self.out_channel = opt['network_G']['out_nc']
        self.out_channel = 1

        self.model = networks.define_IDN(opt).to(self.device)
        if opt['dist']:
            self.model = DistributedDataParallel(self.model, device_ids=[torch.cuda.current_device()])
        else:
            self.model = DataParallel(self.model)

        # print network
        self.print_network()
        self.load()

        # self.Quantization = Quantization()

        if self.is_train:
            self.model.train()

            # loss
            # TODO: impl our loss -- Fidelity loss, Gradient loss, Iso-surface loss
            self.FidelityLoss = FidelityLoss().cuda()
            # self.GradientLoss = GradientLoss(scale=self.train_opt.get('scale', 2)).cuda()
            self.IsosurfacesSimilarityLoss = IsosurfaceSimilarityLoss(scale=1, approx=True)
            # self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            # self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])

            # optimizers
            wd = train_opt['weight_decay'] if train_opt['weight_decay'] else 0
            optim_params = []
            for k, v in self.model.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_model = torch.optim.Adam(optim_params, lr=train_opt['lr'],
                                                    weight_decay=wd,
                                                    betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_model)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            elif train_opt['lr_scheme'] == 'ReduceLROnPlateau_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.ReduceLROnPlateau_Restart(
                            optimizer, factor=train_opt['lr_factor'],
                            patience=train_opt['lr_patience'],
                            threshold=train_opt['lr_threshold']
                        )
                    )
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        # self.ref_L = data['LQ'].to(self.device)  # LQ
        # print("feed GT size: {}".format(data['GT'].size()))
        self.real_H = data['GT'].to(self.device)  # GT

    # def gaussian_batch(self, dims):
    #     return torch.randn(tuple(dims)).to(self.device)

    def loss_forward(self, GT, LR, HR):
        # l_forw_Border = self.BorderLoss.forward(GT, HR, isLR=False)
        # l_forw_Fidelity_GTLR = self.FidelityLoss.forward(GT, LR, isLR=True)
        l_forw_Fidelity_ssim, l_forw_Fidelity_norm = self.FidelityLoss.forward(GT, HR, isLR=False, addSSIM=True)
        # l_forw_Gradient_GTLR = self.GradientLoss.forward(GT, LR, isLR=True)
        # l_forw_Gradient_GTHR = self.GradientLoss.forward(GT, HR, isLR=False)
        # print("BorderLoss: {}".format(l_forw_Border))
        # print("Gradient: LR: {}, HR: {}".format(l_forw_Gradient_GTLR, l_forw_Gradient_GTHR))
        self.IsosurfacesSimilarityLoss.setGTHR(GT, HR)
        l_forw_IsosurfaceSimilarity = self.IsosurfacesSimilarityLoss.forward()
        # l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
        # z = z.reshape([out.shape[0], -1])
        # l_forw_ce = self.train_opt['lambda_ce_forw'] * torch.sum(z ** 2) / z.shape[0]

        return l_forw_Fidelity_ssim, l_forw_Fidelity_norm, l_forw_IsosurfaceSimilarity

    def optimize_parameters(self, step):
        self.optimizer_model.zero_grad()

        # forward downscaling
        self.input = self.real_H
        # print("When optimize, GT size: {}".format(self.real_H.size()))
        self.output_LR, self.output_HR = self.model.forward(x=self.input)
        # print('self.output,self.out_channel',self.output.shape,self.out_channel)

        # zshape = self.output[:, self.out_channel:, :, :, :].shape
        # print('zshape',zshape)
        # TODO: compute loss here ----from here to
        # LR_ref = self.ref_L.detach()

        loss_Fidelity_ssim, loss_Fidelity_norm, loss_IsosurfaceSimilarity = \
            self.loss_forward(GT=self.input, LR=self.output_LR, HR=self.output_HR)

        # backward upscaling
        # LR = self.Quantization(self.output[:, :self.out_channel, :, :, :])
        # LR = self.output[:, :self.out_channel, :, :, :]
        # print('LR,LR_ref',LR.shape,LR_ref.shape)
        # gaussian_scale = self.train_opt['gaussian_scale'] if self.train_opt['gaussian_scale'] != None else 1
        # y_ = torch.cat((LR, gaussian_scale * self.gaussian_batch(zshape)), dim=1)

        # print('self.real_H, y_',self.real_H.shape, y_.shape)
        # l_back_rec = self.loss_backward(self.real_H, y_)

        # total loss
        loss = \
            self.train_opt['lambda_fidelity_GTHR'] * loss_Fidelity_ssim + self.train_opt['lambda_gradient_GTHR'] * loss_Fidelity_norm + self.train_opt['lambda_isosurface_similarity'] * loss_IsosurfaceSimilarity
        # self.train_opt['lambda_fidelity_GTLR'] * loss_Fidelity_GTLR \
        # + self.train_opt['lambda_gradient_GTLR'] * loss_Gradient_GTLR \
        # self.train_opt['lambda_border'] * loss_Border \
        loss.backward()

        # gradient clipping
        """
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.train_opt['gradient_clipping'])
        """

        self.optimizer_model.step()

        # set log
        # self.log_dict['loss_Fidelity_GTLR'] = loss_Fidelity_GTLR
        self.log_dict['loss_Fidelity_ssim'] = loss_Fidelity_ssim
        self.log_dict['loss_Fidelity_norm'] = loss_Fidelity_norm
        # self.log_dict['loss_Gradient_GTLR'] = loss_Gradient_GTLR
        # self.log_dict['loss_Gradient_GTHR'] = loss_Gradient_GTHR
        self.log_dict['loss_Isosurface_Similarity'] = loss_IsosurfaceSimilarity

        # print("After optimize, GT size: {}".format(self.input.size()))
        return loss

    def test(self):
        # Lshape = self.ref_L.shape

        # input_dim = Lshape[1]
        # self.input = self.real_H

        # zshape = [Lshape[0], input_dim * (self.opt['scale'] ** 2) - Lshape[1], Lshape[2], Lshape[3], Lshape[4]]

        # gaussian_scale = 1
        # if self.test_opt and self.test_opt['gaussian_scale'] != None:
        #     gaussian_scale = self.test_opt['gaussian_scale']

        self.model.eval()
        with torch.no_grad():
            # self.forw_L = self.netG(x=self.input)[:, :self.out_channel, :, :, :]
            # self.forw_L = self.Quantization(self.forw_L)
            # y_forw = torch.cat((self.forw_L, gaussian_scale * self.gaussian_batch(zshape)), dim=1)
            # self.fake_H = self.netG(x=y_forw, rev=True)[:, :self.out_channel, :, :, :]
            self.forw_LR, self.forw_HR = self.model.forward(x=self.input)

        self.model.train()

    def encode(self, GT):
        self.model.eval()
        with torch.no_grad():
            LR = self.model.encode(x=GT)
        self.model.train()

        return LR

    def decode(self, LR):
        # Lshape = LR.shape
        # zshape = [Lshape[0], Lshape[1] * (scale ** 2 - 1), Lshape[2], Lshape[3], Lshape[4]]
        # y_ = torch.cat((LR, gaussian_scale * self.gaussian_batch(zshape)), dim=1)

        self.model.eval()
        with torch.no_grad():
            HR = self.model.decode(x=LR)
        self.model.train()

        return HR

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        # print("When get visuals, GT size: {}".format(self.real_H.size()))
        out_dict = OrderedDict()
        # out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        # out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_LR.detach()[0].float().cpu()
        out_dict['HR'] = self.forw_HR.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        # print("LR shape: {} \nHR shape: {} \nGT shape: {}".format(
        #     self.forw_LR.size(), self.forw_HR.size(), self.real_H.size()))
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(self.model, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.model.__class__.__name__,
                                             self.model.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network IDN structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_IDN = self.opt['path']['pretrain_model_IDN']
        if load_path_IDN is not None:
            logger.info('Loading model for IDN [{:s}] ...'.format(load_path_IDN))
            self.load_network(load_path_IDN, self.model, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.model, 'IDN', iter_label)

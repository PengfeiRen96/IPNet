import os
import cv2
import shutil
import logging
import numpy as np

import torch.nn
from tqdm import tqdm
import random

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR

from tensorboardX import SummaryWriter

from config import opt
from dataloader import loader
from util.generateFeature import GFM
from model.IPNet import IPNet, IPNet_MANO
from model.IPNet import Conv_PointMLP,Conv_Point2
from model.loss import SmoothL1Loss
from util import vis_tool
from pointNet.pointMLP import PointMLP
from pointNet.point2_msg_sem import PointNet2SemSegMSG
import json

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.data_rt = self.config.root_dir + "/" + self.config.dataset
        if self.config.model_save == '':
            self.model_save = self.config.net + \
                              '_' + str(self.config.opt) + \
                              "_" + self.config.scheduler + \
                              '_ips' + str(self.config.input_size) + \
                              '_centerType' + self.config.center_type + \
                              '_' + self.config.loss_type + \
                              '_downsample' + str(self.config.downsample) + \
                              '_coord_weight_' + str(self.config.coord_weight) + \
                              '_deconv_weight_' + str(self.config.deconv_weight) + \
                              '_step_size_' + str(self.config.step_size) + \
                              '_CubeSize_' + str(self.config.cube_size[0])

            self.model_save = self.model_save + str(self.config.stage_type)
            self.model_save += '_'
            for index, feature in enumerate(self.config.feature_type):
                self.model_save += feature
            if self.config.finetune_dir != '':
                self.model_save = 'finetune_' + self.model_save
            if self.config.dataset == 'msra':
                self.model_dir = './checkpoint/' + self.config.dataset + '/' + self.model_save + '/' + str(self.config.test_id)
            else:
                self.model_dir = './checkpoint/' + self.config.dataset + '/' + self.model_save
            self.model_dir += self.config.add_info
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.model_dir + '/img')
            os.makedirs(self.model_dir + '/debug')
            os.makedirs(self.model_dir + '/files')

        # save core file
        shutil.copyfile('./train_ho.py', self.model_dir+'/files/train_ho.py')
        shutil.copyfile('./config.py', self.model_dir + '/files/config.py')
        shutil.copyfile('./model/IPNet.py', self.model_dir + '/files/IPNet.py')

        # save config
        with open(self.model_dir + '/config.txt', 'w') as f:
            for k, v in self.config.__class__.__dict__.items():
                if not k.startswith('_'):
                    print(str(k) + ":" + str(v))
                    f.writelines(str(k) + ":" + str(v) + '\n')

        torch.cuda.set_device(self.config.gpu_id)
        cudnn.benchmark = False
        self.dataset = 'hands'
        self.joint_num = self.config.joint_num

        if 'IPNet_MANO' in self.config.net:
            self.net = IPNet_MANO(self.config.net, self.joint_num, self.dataset, './MANO/', kernel_size=self.config.feature_para[0])
        elif 'IPNet' in self.config.net:
            self.net = IPNet(self.config.net, self.joint_num, kernel_size=self.config.feature_para[0], dataset=self.dataset)
        elif 'PointNet' in self.config.net:
            self.net = PointNet2SemSegMSG()
        elif 'PointMLP' in self.config.net:
            self.net = PointMLP(self.config.joint_num, pre_blocks=[4, 4, 4, 4], pos_blocks=[4, 4, 4, 4],embed_dim=64)
        elif 'mix-Point2' in self.config.net:
            self.net = Conv_Point2(self.config.net, self.joint_num)
        elif 'mix-PointMLP' in self.config.net:
            self.net = Conv_PointMLP(self.config.net, self.joint_num)
        else:
            print('Undefined Net !!')
            return 0
        self.net = self.net.cuda()
        print(self.net)
        self.GFM_ = GFM()

        optimList = [{"params": self.net.parameters(), "initial_lr": self.config.lr}]
        # init optimizer
        if self.config.opt == 'sgd':
            self.optimizer = SGD(optimList, lr=self.config.lr, momentum=0.9, weight_decay=1e-4)
        elif self.config.opt == 'adam':
            self.optimizer = Adam(optimList, lr=self.config.lr)
        elif self.config.opt == 'adamw':
            self.optimizer = AdamW(optimList, lr=self.config.lr, weight_decay=0.01)

        self.L1Loss = SmoothL1Loss().cuda()
        self.L2Loss = torch.nn.MSELoss().cuda()
        self.start_epoch = 0

        # load model
        if self.config.load_model != '':
            print('loading model from %s' % self.config.load_model)
            checkpoint = torch.load(self.config.load_model, map_location=lambda storage, loc: storage)
            checkpoint_model = checkpoint['model']
            model_dict = self.net.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint_model.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict)

        # fine-tune model
        if self.config.finetune_dir != '':
            print('loading model from %s' % self.config.finetune_dir)
            checkpoint = torch.load(self.config.finetune_dir, map_location=lambda storage, loc: storage)
            checkpoint_model = checkpoint['model']
            model_dict = self.net.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint_model.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.net.load_state_dict(model_dict)

        # init scheduler
        self.scheduler = StepLR(self.optimizer, step_size=self.config.step_size, gamma=0.1, last_epoch=self.start_epoch)

        if self.config.dataset == 'dexycb':
            if self.config.phase == 'train':
                print(self.config.root_dir)
                self.trainData = loader.DexYCBDataset(self.config.dexycb_setup, 'train', self.config.root_dir, aug_para=self.config.augment_para)
                self.trainLoader = DataLoader(self.trainData, batch_size=self.config.batch_size, shuffle=True, num_workers=4)
            self.testData = loader.DexYCBDataset(self.config.dexycb_setup, 'test', self.config.root_dir)
            self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False, num_workers=4)

        if self.config.dataset == 'ho3d':
            if 'train' in self.config.phase:
                self.trainData = loader.HO3D('train_all', self.config.root_dir,
                                                      dataset_version=config.ho3d_version,
                                                      aug_para=self.config.augment_para,
                                                      img_size=self.config.input_size,
                                                      cube_size=self.config.cube_size,
                                                      center_type='joint_mean')
                self.trainLoader = DataLoader(self.trainData, batch_size=self.config.batch_size, shuffle=True, num_workers=4)
            self.testData = loader.HO3D('test', self.config.root_dir, dataset_version=config.ho3d_version, img_size=self.config.input_size, cube_size=self.config.cube_size, aug_para=[0, 0, 0])
            self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False, num_workers=4)

            self.evalData = loader.HO3D('eval', self.config.root_dir, dataset_version=config.ho3d_version, img_size=self.config.input_size, cube_size=self.config.cube_size, aug_para=[0, 0, 0])
            self.evalLoader = DataLoader(self.evalData, batch_size=self.config.batch_size, shuffle=False, num_workers=4)

        self.test_error = 10000
        self.min_error = 100

        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                            filename=os.path.join(self.model_dir, 'train.log'), level=logging.INFO)
        logging.info('======================================================')
        self.min_error = 100
        self.writer = SummaryWriter('runs/'+self.config.dataset+'-'+self.config.add_info)

    def train(self):
        self.phase = 'train'
        for epoch in range(self.start_epoch, self.config.max_epoch):
            self.net.train()

            for ii, data in tqdm(enumerate(self.trainLoader)):
                joint_xyz_list = []
                joint_uvd_list = []
                img, pcl, xyz_gt, uvd_gt, center, M, cube, cam_para, mano_para_gt, mano_mesh_gt = data
                img, pcl, uvd_gt, xyz_gt, cam_para = img.cuda(), pcl.cuda(), uvd_gt.cuda(), xyz_gt.cuda(), cam_para.cuda()
                center, M, cube = center.cuda(), M.cuda(), cube.cuda()
                mano_para_gt, mano_mesh_gt = mano_para_gt.cuda(), mano_mesh_gt.cuda()
                self.optimizer.zero_grad()
                if 'PointNet' == self.config.net or 'PointMLP' == self.config.net:
                    results = self.net(pcl.permute(0,2,1))
                else:
                    results = self.net(img, pcl, self.trainData, center, M, cube, cam_para, 0.8)
                loss = 0
                iter_num = ii + (self.trainData.__len__()//self.config.batch_size)*epoch

                for index, stage_type in enumerate(self.config.stage_type):
                    if stage_type == 0:# Regress-uvd
                        joint_uvd = results[index]
                        joint_xyz = self.trainData.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para)
                        loss_coord = self.L1Loss(joint_uvd, uvd_gt)*100
                        loss += loss_coord
                        joint_xyz_list.append(joint_xyz)
                        joint_uvd_list.append(joint_uvd)
                        batch_joint_error = self.xyz2error(joint_xyz, xyz_gt, center, cube)
                        error = np.mean(batch_joint_error)
                    elif stage_type == 1: # pixel-wise pixel-uvd
                        pixel_pd = results[index] #B x 5J x FS x FS
                        feature_size = pixel_pd.size(-1)
                        pixel_gt = self.GFM_.joint2feature(uvd_gt, img, self.config.feature_para, feature_size,self.config.feature_type)
                        joint_uvd = self.GFM_.feature2joint(img, pixel_pd, self.config.feature_type,self.config.feature_para)
                        joint_xyz = self.trainData.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para)
                        loss_pixel = self.L1Loss(pixel_pd[:, :pixel_gt.size(1)], pixel_gt) * self.config.deconv_weight
                        loss_coord = self.L1Loss(joint_uvd, uvd_gt) * self.config.coord_weight
                        loss += (loss_pixel + loss_coord)

                        joint_xyz_list.append(joint_xyz)
                        joint_uvd_list.append(joint_uvd)
                        batch_joint_error = self.xyz2error(joint_xyz, xyz_gt, center, cube)
                        error = np.mean(batch_joint_error)
                        self.writer.add_scalar('loss_pixel', loss_pixel, global_step=iter_num)
                        self.writer.add_scalar('loss_coord', loss_coord, global_step=iter_num)
                    elif stage_type == 2: # Regress-XYZ
                        joint_xyz = results[index]
                        joint_uvd = self.trainData.xyz_nl2uvdnl_tensor(joint_xyz, center, M, cube, cam_para)
                        loss_coord = self.L1Loss(joint_xyz, xyz_gt) * self.config.coord_weight
                        loss += loss_coord

                        joint_xyz_list.append(joint_xyz)
                        joint_uvd_list.append(joint_uvd)
                        batch_joint_error = self.xyz2error(joint_xyz, xyz_gt, center, cube)
                        error = np.mean(batch_joint_error)
                    elif stage_type == 3: # PCL-XYZ point-wise 1024 x 5J
                        pcl, pcl_result = results[index]
                        pcl_gt = self.GFM_.pcl_joint2offset(xyz_gt, pcl, self.config.feature_para[0])
                        joint_xyz = self.GFM_.pcl_offset2joint_weight(pcl_result, pcl, self.config.feature_para[0])
                        joint_uvd = self.trainData.xyz_nl2uvdnl_tensor(joint_xyz, center, M, cube, cam_para)
                        loss_pixel = self.L1Loss(pcl_result[:, :, :pcl_gt.size(-1)], pcl_gt) * self.config.deconv_weight
                        loss_coord = self.L1Loss(joint_xyz, xyz_gt) * self.config.coord_weight
                        loss += (loss_pixel + loss_coord)

                        joint_xyz_list.append(joint_xyz)
                        joint_uvd_list.append(joint_uvd)
                        batch_joint_error = self.xyz2error(joint_xyz, xyz_gt, center, cube)
                        error = np.mean(batch_joint_error)
                    elif stage_type == 4:# PCL-UVD
                        pcl_uvd, pcl_result = results[index]
                        pcl_gt = self.GFM_.pcl_joint2offset(uvd_gt, pcl_uvd, self.config.feature_para[0])
                        joint_uvd = self.GFM_.pcl_offset2joint_weight(pcl_result, pcl_uvd, self.config.feature_para[0])
                        joint_xyz = self.trainData.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube)
                        loss_pixel = self.L1Loss(pcl_result[:, :, :pcl_gt.size(-1)], pcl_gt) * self.config.deconv_weight
                        loss_coord = self.L1Loss(joint_uvd, uvd_gt) * self.config.coord_weight
                        loss += (loss_pixel + loss_coord)

                        joint_xyz_list.append(joint_xyz)
                        joint_uvd_list.append(joint_uvd)
                        batch_joint_error = self.xyz2error(joint_xyz, xyz_gt, center, cube)
                        error = np.mean(batch_joint_error)
                    elif stage_type == 5:
                        mano_mesh, joint_xyz, beta = results[index]
                        joint_uvd = self.trainData.xyz_nl2uvdnl_tensor(joint_xyz, center, M, cube, cam_para)
                        loss_mesh = self.L1Loss(mano_mesh, mano_mesh_gt) * 10
                        loss_joint = self.L1Loss(joint_xyz, xyz_gt) * 10
                        loss_beta = torch.mean(torch.abs(beta[:, 48:58]))*1
                        loss += (loss_joint + loss_mesh + loss_beta)

                        joint_xyz_list.append(joint_xyz)
                        joint_uvd_list.append(joint_uvd)
                        batch_joint_error = self.xyz2error(joint_xyz, xyz_gt, center, cube)
                        error = np.mean(batch_joint_error)
                    self.writer.add_scalar('error_{}'.format(index), error, global_step=iter_num)
                    if iter_num % 100 == 0:
                        img_show = vis_tool.draw_2d_pose(img[0], joint_uvd[0], self.dataset)
                        self.writer.add_image('img_{}'.format(index), np.transpose(img_show, (2, 0, 1)) / 255.0, global_step=iter_num)
                loss.backward()
                self.optimizer.step()

            test_error = self.test(epoch)
            if test_error <= self.min_error:
                self.min_error = test_error
                save = {
                    "model": self.net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch
                }
                torch.save(
                    save,
                    self.model_dir + "/best.pth"
                )
            save = {
                "model": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(
                save,
                self.model_dir + "/latest.pth"
            )

            if self.config.scheduler == 'auto':
                self.scheduler.step(test_error)
            elif self.config.scheduler == 'step':
                self.scheduler.step(epoch)
            elif self.config.scheduler == 'multi_step':
                self.scheduler.step()

    @torch.no_grad()
    def test(self, epoch=-1):
        self.phase = 'test'
        self.result_file_list = [ ]
        for index in range(len(self.config.stage_type)):
            self.result_file_list.append(open(self.model_dir + '/test_%d.txt'%(index), 'w'))
        self.id_file = open(self.model_dir + '/id.txt', 'w')
        self.mano_file = open(self.model_dir + '/eval_mano.txt', 'w')
        self.net.eval()
        batch_num = 0
        error_list = [0] * len(self.config.stage_type)
        for ii, data in tqdm(enumerate(self.testLoader)):
            img, pcl, xyz_gt, uvd_gt, center, M, cube, cam_para, mano_para, seg = data
            img, pcl, uvd_gt, xyz_gt, cam_para = img.cuda(), pcl.cuda(), uvd_gt.cuda(), xyz_gt.cuda(), cam_para.cuda()
            center, M, cube = center.cuda(), M.cuda(), cube.cuda()
            if 'PointNet' == self.config.net or 'PointMLP' == self.config.net:
                results = self.net(pcl.permute(0, 2, 1))
            else:
                results = self.net(img, pcl, self.testData, center, M, cube, cam_para, 0.8)
            batch_num += 1
            joint_error_list = []
            for index, stage_type in enumerate(self.config.stage_type):
                if stage_type == 0:
                    joint_uvd = results[index]
                    joint_xyz = self.testData.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para)
                    joint_errors = self.xyz2error(joint_xyz, xyz_gt, center, cube, self.result_file_list[index])
                    batch_errors = np.mean(joint_errors, axis=-1)
                elif stage_type == 1:
                    pixel_pd = results[index]
                    joint_uvd = self.GFM_.feature2joint(img, pixel_pd, self.config.feature_type,self.config.feature_para)
                    joint_xyz = self.testData.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para)
                    joint_errors = self.xyz2error(joint_xyz, xyz_gt, center, cube, self.result_file_list[index])
                    batch_errors = np.mean(joint_errors, axis=-1)
                elif stage_type == 2:
                    joint_xyz = results[index]
                    joint_uvd = self.testData.xyz_nl2uvdnl_tensor(joint_xyz, center, M, cube, cam_para)
                    joint_errors = self.xyz2error(joint_xyz, xyz_gt, center, cube, self.result_file_list[index])
                    batch_errors = np.mean(joint_errors, axis=-1)
                elif stage_type == 3:
                    pcl, pcl_result = results[index]
                    joint_xyz = self.GFM_.pcl_offset2joint_weight(pcl_result, pcl, self.config.feature_para[0])
                    joint_uvd = self.testData.xyz_nl2uvdnl_tensor(joint_xyz, center, M, cube, cam_para)
                    joint_errors = self.xyz2error(joint_xyz, xyz_gt, center, cube, self.result_file_list[index])
                    batch_errors = np.mean(joint_errors, axis=-1)
                elif stage_type == 4:
                    pcl_uvd, pixel_pd = results[index]
                    joint_uvd = self.GFM_.pcl_offset2joint_weight(pixel_pd, pcl_uvd, self.config.feature_para[0])
                    joint_xyz = self.testData.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para)
                    joint_errors = self.xyz2error(joint_xyz, xyz_gt, center, cube, self.result_file_list[index])
                    batch_errors = np.mean(joint_errors, axis=-1)
                    pixel_pd = pixel_pd.permute(0, 2, 1)
                elif stage_type == 5:# MANO
                    mano_mesh, joint_xyz, beta = results[index]
                    joint_uvd = self.testData.xyz_nl2uvdnl_tensor(joint_xyz, center, M, cube, cam_para)
                    joint_errors = self.xyz2error(joint_xyz, xyz_gt, center, cube, self.result_file_list[index])
                    batch_errors = np.mean(joint_errors, axis=-1)

                joint_error_list.append(joint_errors)
                error = np.mean(batch_errors)
                error_list[index] += error
                joint_xyz_world = joint_xyz * cube.unsqueeze(1) / 2 + center.unsqueeze(1)
                joint_xyz_world = joint_xyz_world.detach().cpu()
                np.savetxt(self.result_file_list[index], joint_xyz_world.reshape([-1, 21 * 3]), fmt='%.3f')
            if 5 in self.config.stage_type:
                np.savetxt(self.mano_file, beta.detach().cpu().reshape([-1, 62]), fmt='%.3f')
            if ii % 100 == 0:
                vis_tool.debug_2d_pose(img, joint_uvd, ii, self.config.dataset, self.model_dir + '/img', 'pd', self.config.batch_size, True)
        error_info = ''
        for index in range(len(error_list)):
            print("[mean_Error %.3f]"% (error_list[index] / batch_num))
            error_info += ' error' + str(index) + ": %.3f" % (error_list[index] / batch_num) + ' '
        logging.info(error_info)
        return error_list[-1] / batch_num

    @torch.no_grad()
    def evalution(self, epoch=-1):
        self.phase = 'evalutioin'
        self.net.eval()
        joint_list = []
        mesh_list = []
        MANO2HO3D = [0,
                     1, 2, 3,
                     4, 5, 6,
                     7, 8, 9,
                     10, 11, 12,
                     13, 14, 15,
                     20, 16, 17, 19, 18]
        self.result_file_list = [ ]
        for index in range(len(self.config.stage_type)):
            self.result_file_list.append(open(self.model_dir + '/test_%d.txt'%(index), 'w'))
        self.joint_file = open(self.model_dir + '/eval_joint.txt', 'w')
        self.mesh_file = open(self.model_dir + '/eval_mesh.txt', 'w')
        self.mano_file = open(self.model_dir + '/eval_mano.txt', 'w')

        for ii, data in tqdm(enumerate(self.evalLoader)):
            img, pcl, xyz_gt, uvd_gt, center, M, cube, cam_para, mano_para, seg = data
            img, pcl, uvd_gt, xyz_gt, cam_para = img.cuda(), pcl.cuda(), uvd_gt.cuda(), xyz_gt.cuda(), cam_para.cuda()
            center, M, cube = center.cuda(), M.cuda(), cube.cuda()
            if 'PointNet' == self.config.net or 'DGCNN' == self.config.net or 'PointMLP' == self.config.net:
                results = self.net(pcl.permute(0, 2, 1))
            else:
                results = self.net(img, pcl, self.evalData, center, M, cube, cam_para, 0.8)
            batch_size = img.size(0)
            mano_mesh = torch.zeros([img.size(0), 779, 3]).to(img.device)
            joint_xyz_list = []
            for index, stage_type in enumerate(self.config.stage_type):
                if stage_type == 0:
                    joint_uvd = results[index]
                    joint_xyz = self.evalData.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para)
                elif stage_type == 1:
                    pixel_pd = results[index]
                    joint_uvd = self.GFM_.feature2joint(img, pixel_pd, self.config.feature_type,self.config.feature_para)
                    joint_xyz = self.evalData.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para)
                elif stage_type == 2:
                    joint_xyz = results[index]
                    joint_uvd = self.evalData.xyz_nl2uvdnl_tensor(joint_xyz, center, M, cube, cam_para)
                elif stage_type == 3:
                    pcl, pcl_result = results[index]
                    joint_xyz = self.GFM_.pcl_offset2joint_weight(pcl_result, pcl, self.config.feature_para[0])
                    joint_uvd = self.evalData.xyz_nl2uvdnl_tensor(joint_xyz, center, M, cube, cam_para)
                elif stage_type == 4:
                    pcl_uvd, pixel_pd = results[index]
                    joint_uvd = self.GFM_.pcl_offset2joint_weight(pixel_pd, pcl_uvd, self.config.feature_para[0])
                    joint_xyz = self.testData.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para)
                elif stage_type == 5:# MANO
                    mano_mesh, joint_xyz, beta = results[index]
                    joint_uvd = self.evalData.xyz_nl2uvdnl_tensor(joint_xyz, center, M, cube, cam_para)
                joint_xyz_world = joint_xyz * cube.unsqueeze(1) / 2 + center.unsqueeze(1)
                joint_xyz_world = joint_xyz_world.detach().cpu()
                np.savetxt(self.result_file_list[index], joint_xyz_world.reshape([-1, 21 * 3]), fmt='%.3f')
                joint_xyz_list.append(joint_xyz)

            np.savetxt(self.mano_file, beta.detach().cpu().reshape([-1, 62]), fmt='%.3f')
            if ii % 100 == 0:
                vis_tool.debug_2d_pose(img, joint_uvd, ii, self.config.dataset, self.model_dir + '/img', 'pd', self.config.batch_size, True)
            joint_xyz_world = joint_xyz_list[5] * cube.unsqueeze(1) / 2 + center.unsqueeze(1)
            mesh_xyz_world = mano_mesh * cube.unsqueeze(1) / 2 + center.unsqueeze(1)
            mesh_xyz_world = mesh_xyz_world.detach().cpu()
            joint_xyz_world = joint_xyz_world.detach().cpu().numpy()[:, MANO2HO3D, :]
            np.savetxt(self.joint_file, joint_xyz_world.reshape([-1, 21 * 3]), fmt='%.3f')
            joint_xyz_world *= np.array([1, -1, -1]) / 1000
            mesh_xyz_world *= np.array([1, -1, -1]) / 1000
            joint_list = joint_list + np.split(joint_xyz_world, batch_size, axis=0)
            mesh_list = mesh_list + np.split(mesh_xyz_world, batch_size, axis=0)
        self.dump(self.model_dir+'/pred.json', joint_list, mesh_list)
        return 0

    @torch.no_grad()
    def xyz2error(self, output, joint, center, cube_size, write_file=None):
        output = output.detach().cpu().numpy()
        joint = joint.detach().cpu().numpy()
        center = center.detach().cpu().numpy()
        cube_size = cube_size.detach().cpu().numpy()
        batchsize, joint_num, _ = output.shape
        center = np.tile(center.reshape(batchsize, 1, -1), [1, joint_num, 1])
        cube_size = np.tile(cube_size.reshape(batchsize, 1, -1), [1, joint_num, 1])

        joint_xyz = output * cube_size / 2 + center
        joint_world_select = joint * cube_size / 2 + center

        errors = (joint_xyz - joint_world_select) * (joint_xyz - joint_world_select)
        errors = np.sqrt(np.sum(errors, axis=2))

        return errors

    def vis_hardcase(self, error, img, joint, ii, heatmap=None, t=15, name=''):
        B, J, _ = joint.size()
        hard_index = (error > t).sum(-1)
        hard_index = np.where(hard_index > 0)[0]
        hard_joint = joint[hard_index, ...]
        hard_img = img[hard_index, ...]
        if hard_index.sum() > 0:
            vis_tool.debug_2d_pose_select(hard_img, hard_joint, ii, self.dataset, self.model_dir + '/img', 'joint_' + name, self.config.batch_size, hard_index, True)
        return 0

    def dump(self, pred_out_path, xyz_pred_list, verts_pred_list):
        """ Save predictions into a json file. """
        # make sure its only lists
        xyz_pred_list = [x[0].tolist() for x in xyz_pred_list]
        verts_pred_list = [x[0].tolist() for x in verts_pred_list]

        # save to a json
        with open(pred_out_path, 'w') as fo:
            json.dump(
                [
                    xyz_pred_list,
                    verts_pred_list
                ], fo)
        print('Dumped %d joints and %d verts predictions to %s' % (
        len(xyz_pred_list), len(verts_pred_list), pred_out_path))



if __name__ == '__main__':
    # set_seed(0)
    Trainer = Trainer(opt)
    if 'train' in Trainer.config.phase:
        Trainer.train()
        Trainer.test()
        Trainer.evalution()
    elif Trainer.config.phase == 'test':
        Trainer.test()
        # Trainer.result_file.close()
    elif Trainer.config.phase == 'eval':
        Trainer.evalution()
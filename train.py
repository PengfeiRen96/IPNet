import os
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
from model.IPNet import IPNet_MANO,IPNet
from model.loss import SmoothL1Loss
from util import vis_tool

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
        shutil.copyfile('./train.py', self.model_dir+'/files/train.py')
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
        self.dataset = 'nyu_all' if 'nyu' in self.config.dataset else 'hands'
        self.joint_num = 23 if 'nyu' in self.config.dataset else self.config.joint_num

        if 'IPNet_MANO' in self.config.net:
            self.net = IPNet_MANO(self.config.net, self.joint_num, self.dataset, './MANO/', kernel_size=self.config.feature_para[0])
        elif 'IPNet' in self.config.net:
            self.net = IPNet(self.config.net, self.joint_num, kernel_size=self.config.feature_para[0], dataset=self.dataset)
        else:
            print('Undefined Net !!')
            return 0
        self.net = self.net.cuda()
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

        if self.config.phase == 'train':
            self.trainData = loader.nyu_loader(self.data_rt, 'train', aug_para=self.config.augment_para,
                                               img_size=self.config.input_size,
                                               cube_size=self.config.cube_size,
                                               center_type=self.config.center_type)
            self.trainLoader = DataLoader(self.trainData, batch_size=self.config.batch_size, shuffle=True,
                                          num_workers=4)
        self.testData = loader.nyu_loader(self.data_rt, 'test', img_size=self.config.input_size,
                                          cube_size=self.config.cube_size,
                                          center_type=self.config.center_type, aug_para=[0, 0, 0])

        self.testLoader = DataLoader(self.testData, batch_size=self.config.batch_size, shuffle=False, num_workers=4)
        self.test_error = 1e8
        self.min_error = 1e8

        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                            filename=os.path.join(self.model_dir, 'train.log'), level=logging.INFO)
        logging.info('======================================================')
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

                if 'PointMLP' in self.config.net:
                    results = self.net(pcl.permute(0, 2, 1))
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
                        mano_mesh, joint_xyz, mano_para = results[index]
                        joint_uvd = self.trainData.xyz_nl2uvdnl_tensor(joint_xyz, center, M, cube, cam_para)
                        loss_mesh = self.L1Loss(mano_mesh, mano_mesh_gt) * 10
                        loss_joint = self.L1Loss(joint_xyz, convert_nyu2mano(xyz_gt)) * 0
                        loss_beta = torch.mean(torch.abs(mano_para[:, 48:58]))*1
                        loss_scale = torch.mean(torch.abs(torch.min(mano_para[:, 61:62], torch.zeros_like(mano_para[:, 61:62]).to(img.device))))

                        loss += (loss_mesh + loss_beta + loss_joint + loss_scale)

                        joint_xyz_list.append(joint_xyz)
                        joint_uvd_list.append(joint_uvd)
                        batch_joint_error = self.xyz2error(joint_xyz, convert_nyu2mano(xyz_gt), center, cube)
                        error = np.mean(batch_joint_error)
                    self.writer.add_scalar('error_{}'.format(index), error, global_step=iter_num)
                if ii % 20 == 0:
                    for joint_list_index, joint_uvd in enumerate(joint_uvd_list):
                        if joint_uvd.size(1) == 23:
                            img_show = vis_tool.draw_2d_pose(img[0], joint_uvd[0], self.dataset)
                        else:
                            img_show = vis_tool.draw_2d_pose(img[0], joint_uvd[0], 'mano')
                        self.writer.add_image('pd-%d'%(joint_list_index), np.transpose(img_show, (2, 0, 1)) / 255.0, global_step=ii)
                    img_show = vis_tool.draw_2d_pose(img[0], uvd_gt[0], self.dataset)
                    self.writer.add_image('gt', np.transpose(img_show, (2, 0, 1)) / 255.0, global_step=ii)

                loss.backward()
                self.optimizer.step()

            test_error = self.min_error

            if not 'hands' in self.config.dataset:
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
            self.result_file_list.append(open(self.model_dir + '/result_%d.txt'%(index), 'w'))
        self.id_file = open(self.model_dir + '/id.txt', 'w')
        shutil.rmtree(self.model_dir + '/img')
        os.mkdir(self.model_dir + '/img')
        self.mano_file = open(self.model_dir + '/eval_mano.txt', 'w')
        self.net.eval()
        batch_num = 0
        error_list = [0] * len(self.config.stage_type)
        for ii, data in tqdm(enumerate(self.testLoader)):
            img, pcl, xyz_gt, uvd_gt, center, M, cube, cam_para, mano_para_gt, mano_mesh_gt = data
            img, pcl, uvd_gt, xyz_gt, cam_para = img.cuda(), pcl.cuda(), uvd_gt.cuda(), xyz_gt.cuda(), cam_para.cuda()
            center, M, cube = center.cuda(), M.cuda(), cube.cuda()
            if 'Point' in self.config.net:
                results = self.net(pcl.permute(0,2,1))
            else:
                results = self.net(img, pcl, self.testData, center, M, cube, cam_para, 0.8)
            batch_num += 1
            joint_error_list = []
            joint_uvd_list = []
            for index, stage_type in enumerate(self.config.stage_type):
                #AWR dense((21*4)*64*64) joint(21*3)
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
                elif stage_type == 5:
                    mano_mesh, joint_xyz, beta = results[index]
                    joint_uvd = self.testData.xyz_nl2uvdnl_tensor(joint_xyz, center, M, cube, cam_para)
                    joint_errors = self.xyz2error(joint_xyz, convert_nyu2mano(xyz_gt), center, cube, self.result_file_list[index])
                    batch_errors = np.mean(joint_errors, axis=-1)
                joint_uvd_list.append(joint_uvd)
                joint_error_list.append(joint_errors)
                error = np.mean(batch_errors)
                error_list[index] += error
            if (batch_errors > 20).sum() != 0:
                img_id = np.arange(img.size(0))[batch_errors > 20]
                img_id = self.config.batch_size * ii + img_id
                np.savetxt(self.id_file, img_id, fmt='%d')
            np.savetxt(self.mano_file, beta.detach().cpu().reshape([-1, 62]), fmt='%.3f')

        error_sum = 0
        error_info = ''
        for index in range(len(error_list)):
            print("[mean_Error %.3f]"% (error_list[index] / batch_num))
            error_info += ' error' + str(index) + ": %.3f" % (error_list[index] / batch_num) + ' '
            error_sum += error_list[index]
        logging.info(error_info)
        return error_sum / batch_num

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
        if joint_num == 23:
            calculate = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 21, 22, 20]
            errors = np.sqrt(np.sum(errors[:, calculate, :], axis=2))
        else:
            errors = np.sqrt(np.sum(errors, axis=2))
        if self.phase == 'test' and write_file is not None:
            np.savetxt(write_file, self.testData.joint3DToImg(joint_xyz).reshape([batchsize, joint_num * 3]), fmt='%.3f')

        return errors


def convert_nyu2mano(joint):
    select_joint = joint.clone()
    select_joint[:, 1, :] = joint[:, 1, :] + (joint[:, 2, :] - joint[:, 1, :]) * 0.3
    select_joint[:, 5, :] = joint[:, 5, :] + (joint[:, 6, :] - joint[:, 5, :]) * 0.3
    select_joint[:, 9, :] = joint[:, 9, :] + (joint[:, 10, :] - joint[:, 9, :]) * 0.3
    select_joint[:, 13, :] = joint[:, 13, :] + (joint[:, 14, :] - joint[:, 13, :]) * 0.3
    select_joint[:, 17, :] = joint[:, 17, :] + (joint[:, 18, :] - joint[:, 17, :]) * 0.2

    select_joint[:, 0, :] = joint[:, 0, :] - (joint[:, 1, :] - joint[:, 0, :]) * 0.3
    select_joint[:, 4, :] = joint[:, 4, :] - (joint[:, 5, :] - joint[:, 4, :]) * 0.3
    select_joint[:, 8, :] = joint[:, 8, :] - (joint[:, 9, :] - joint[:, 8, :]) * 0.3
    select_joint[:, 12, :] = joint[:, 12, :] - (joint[:, 13, :] - joint[:, 12, :]) * 0.3
    select_joint[:, 16, :] = joint[:, 16, :] - (joint[:, 17, :] - joint[:, 16, :]) * 0.3

    select_joint[:, 3, :] = joint[:, 3, :] - (joint[:, 3, :] - joint[:, 2, :]) * 0.1
    select_joint[:, 7, :] = joint[:, 7, :] - (joint[:, 7, :] - joint[:, 6, :]) * 0.1
    select_joint[:, 11, :] = joint[:, 11, :] - (joint[:, 11, :] - joint[:, 10, :]) * 0.2
    select_joint[:, 15, :] = joint[:, 15, :] - (joint[:, 15, :] - joint[:, 14, :]) * 0.3

    NYU2MANO = [22,
                15,14,13,
                11,10,9,
                3,2,1,
                7,6,5,
                19,18,17,
                12,8,0,4,16]
    return select_joint[:, NYU2MANO, :]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    set_seed(0)
    Trainer = Trainer(opt)
    if Trainer.config.phase == 'train':
        Trainer.train()
        Trainer.test()
    else:
        Trainer.test()

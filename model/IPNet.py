import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from util.vis_tool import get_hierarchy_sketch, get_hierarchy_mapping, get_sketch_setting
from util.graph_util import adj_mx_from_edges
from semGCN.sem_gcn import SimpleSemGCN
from semGCN.ph_gcn import SPGCN
from model.resnetUnet import OfficialResNetUnet
from convNeXT.resnetUnet import convNeXTUnet
from pointNet.pointMLP import PointMLP_refine
from pointNet.point2_msg_sem import PointNet2SemSegMSG_refine
import numpy as np
from render.obman_mano import Render
from pointnet2_ops import pointnet2_utils

BN_MOMENTUM = 0.1


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class NodeSimpleSemGCN(nn.Module):
    def __init__(self, dim, joint_num, num_layers=3, dataset='nyu_all'):
        super().__init__()
        adjs = adj_mx_from_edges(joint_num, get_sketch_setting(dataset), sparse=False, eye=True)
        self.gcn = SimpleSemGCN(adjs, dim, num_layers=num_layers)

    def forward(self, x):
        x = self.gcn(x)
        return x


class PoolSPGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, joint_num, dataset='nyu_all'):
        super().__init__()
        adjs = []
        for ajd in get_hierarchy_sketch(dataset):
            bone_num = np.array(ajd).max() + 1
            adjs.append(adj_mx_from_edges(bone_num, ajd, sparse=False, eye=True))
        node_maps = get_hierarchy_mapping(dataset)
        self.gcn = SPGCN(adjs, node_maps, in_dim, hid_dim, out_dim)

    def forward(self, x):
        x = self.gcn(x)
        return x


class RelationFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp, topk=3):
        super(RelationFeaturePropagation, self).__init__()
        self.topk = topk
        self.relation_conv = nn.Conv2d(3+1, 1, kernel_size=(1, 1), stride=(1, 1))
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel*2
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, C]
            xyz2: sampled input points position data, [B, S, C]
            points1: input points data, [B, N, D]
            points2: input points data, [B, S, D]
        Return:
            new_points: upsampled points data, [B, N, D']
        """
        dists = square_distance(xyz1, xyz2)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :self.topk], idx[:, :, :self.topk]
        offset = xyz1.unsqueeze(-2) - index_points(xyz2, idx)
        relation_prior = torch.cat((dists.unsqueeze(-1), offset), dim=-1).permute(0, 3, 1, 2)

        weight = self.relation_conv(relation_prior).squeeze(1)
        weight = torch.sigmoid(weight)
        interpolated_points = torch.mean(index_points(points2, idx)*weight.unsqueeze(-1), dim=2)
        new_points = torch.cat([points1, interpolated_points], dim=-1)

        # relation_prior = self.relation_conv(relation_prior)
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points.permute(0, 2, 1)


class DESA(nn.Module):
    def __init__(self, in_channel, mlp, S=[64, 64, 64], radius=[0.1, 0.2, 0,4]):
        super(DESA, self).__init__()
        self.S = S
        self.radius = radius
        self.scale_num = len(radius)
        self.groupers = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        self.conv_l0_blocks = nn.ModuleList()
        self.conv_f0_blocks = nn.ModuleList()
        self.bn_l0_blocks = nn.ModuleList()
        self.bn_f0_blocks = nn.ModuleList()

        for i in range(self.scale_num):
            self.conv_l0_blocks.append(nn.Conv2d(3, mlp[0], 1))
            self.conv_f0_blocks.append(nn.Conv2d(in_channel, mlp[0], 1))
            self.bn_l0_blocks.append(nn.BatchNorm2d(mlp[0]))
            self.bn_f0_blocks.append(nn.BatchNorm2d(mlp[0]))
            last_channel = mlp[0]
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            for out_channel in mlp[1:]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius[i], S[i], use_xyz=True))

        self.fusion = nn.Sequential(
                        nn.Conv1d(in_channel+mlp[-1]*self.scale_num, in_channel, 1),
                        nn.BatchNorm1d(in_channel),
                        nn.ReLU()
        )

    def forward(self, pcl_feat, node_feat, pcl_xyz, node_xyz):
        B, J, C = node_feat.size()
        pcl_xyz_expand = torch.cat((pcl_xyz, node_xyz), dim=1)
        pcl_feat_expand = torch.cat((pcl_feat, node_feat), dim=1)
        grouped_feat_list = []
        for i in range(self.scale_num):
            radius = self.radius[i]
            grouper = self.groupers[i]
            grouped_feat = grouper(pcl_xyz_expand, node_xyz, pcl_feat_expand.transpose(2, 1).contiguous())
            grouped_xyz, grouped_feat = torch.split(grouped_feat, [3, C], dim=1)

            group_xyz_norm = grouped_xyz / radius
            grouped_feat = grouped_feat - node_feat.permute(0, 2, 1).view(B, C, J, 1)

            # init layer
            loc = self.bn_l0_blocks[i](self.conv_l0_blocks[i](group_xyz_norm))
            feat = self.bn_f0_blocks[i](self.conv_f0_blocks[i](grouped_feat))
            grouped_feat = loc + feat
            grouped_feat = F.relu(grouped_feat)

            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_feat = F.relu(bn(conv(grouped_feat)))
            grouped_feat = torch.max(grouped_feat, dim=-1)[0]
            grouped_feat_list.append(grouped_feat)

        grouped_feat_list.append(node_feat.permute(0, 2, 1))
        grouped_feat_concat = torch.cat(grouped_feat_list, dim=1)
        out = self.fusion(grouped_feat_concat)
        return out.permute(0, 2, 1)


class Block(nn.Module):
    def __init__(self, joint_num, dim, kernel_size=0.8, dataset='nyu_all'):
        super().__init__()
        self.kernel_size = kernel_size
        self.joint_num = joint_num
        self.FA = DESA(dim, [128, 128], [64, 64, 64], [0.1, 0.2, 0.4])
        self.FP = RelationFeaturePropagation(dim, [dim], topk=4)
        self.joint_attn = NodeSimpleSemGCN(dim, joint_num, num_layers=2, dataset=dataset)
        self.token_reg = nn.Linear(dim, 5*joint_num)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, pcl_feat, joint_feat, pcl_xyz, joint_xyz):
        joint_feat = self.FA(pcl_feat, joint_feat, pcl_xyz, joint_xyz)
        joint_feat = self.joint_attn(joint_feat)
        pcl_feat = self.FP(pcl_xyz, joint_xyz, pcl_feat, joint_feat)
        token_offset = self.token_reg(pcl_feat)

        return pcl_feat, token_offset


class IPNet(nn.Module):
    def __init__(self, net, joint_num, dataset, kernel_size=1):
        super(IPNet, self).__init__()
        self.joint_num = joint_num
        self.kernel_size = kernel_size
        self.dim = 128
        self.num_stages = 1
        if 'convnext' in net:
            self.backbone = convNeXTUnet(net, joint_num, pretrain='1k', deconv_dim=self.dim, out_dim_list=[joint_num*3, joint_num, joint_num])
        elif 'resnet' in net:
            self.backbone = OfficialResNetUnet(net, joint_num, pretrain=True, deconv_dim=self.dim, out_dim_list=[joint_num*3, joint_num, joint_num])

        self.softmax = nn.Softmax(dim=-1)

        self.pcl_feat_emb = nn.Sequential(nn.Conv1d(self.dim, self.dim, 1), nn.BatchNorm1d(self.dim))
        self.pcl_xyz_emb = nn.Sequential(nn.Conv1d(3, self.dim, 1), nn.BatchNorm1d(self.dim))
        self.pcl_pose_emb = nn.Sequential(nn.Conv1d(self.joint_num * 5, self.dim, 1), nn.BatchNorm1d(self.dim))
        self.joint_feat_emb = nn.Sequential(nn.Conv1d(self.dim, self.dim, 1), nn.BatchNorm1d(self.dim))
        self.joint_xyz_emb = nn.Sequential(nn.Conv1d(3, self.dim, 1), nn.BatchNorm1d(self.dim))

        for i in range(self.num_stages):
            block = Block(joint_num=joint_num, dim=self.dim, kernel_size=self.kernel_size, dataset=dataset)
            # split emd
            pcl_feat_emb = nn.Sequential(nn.Conv1d(self.dim, self.dim, 1), nn.BatchNorm1d(self.dim))
            pcl_xyz_emb = nn.Sequential(nn.Conv1d(3, self.dim, 1), nn.BatchNorm1d(self.dim))
            pcl_pose_emb = nn.Sequential(nn.Conv1d(self.joint_num * 5, self.dim, 1), nn.BatchNorm1d(self.dim))
            joint_feat_emb = nn.Sequential(nn.Conv1d(self.dim, self.dim, 1), nn.BatchNorm1d(self.dim))
            joint_xyz_emb = nn.Sequential(nn.Conv1d(3, self.dim, 1), nn.BatchNorm1d(self.dim))

            setattr(self, f"block{i + 1}", block)
            setattr(self, f"pcl_feat_emb{i + 1}", pcl_feat_emb)
            setattr(self, f"pcl_xyz_emb{i + 1}", pcl_xyz_emb)
            setattr(self, f"pcl_pose_emb{i + 1}", pcl_pose_emb)
            setattr(self, f"joint_feat_emb{i + 1}", joint_feat_emb)
            setattr(self, f"joint_xyz_emb{i + 1}", joint_xyz_emb)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.001)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=0.001)

    def forward(self, img, pcl, loader, center, M, cube, cam_para, kernel):
        img_offset, img_feat = self.backbone(img)   # img_offset: B×C×W×H , C=3(direct vector)+1(heatmap)+1(weight)

        result = [img_offset]
        # joint_uvd = offset2joint_weight(img_offset, img, kernel)
        # B, C, H, W = img_feat.size()
        # img_down = F.interpolate(img, [H, W])

        # joint_uvd = joint_uvd.detach()
        # img_offset = img_offset.detach()
        # B, N, _ = pcl.size()
        # J = self.joint_num
        #
        # joint_xyz = loader.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para).detach()
        # pcl_offset_xyz = pcl_joint2offset(joint_xyz, pcl, 0.8)  # 反向传播信息的时，采用固定的范围
        #
        # pcl_closeness, pcl_index = loader.img2pcl_index(pcl, img_down, center, M, cube, cam_para, select_num=4)
        # pcl_feat_index = pcl_index.view(B, 1, -1).repeat(1, C, 1)   # B*128*(K*1024)
        # pcl_feat = torch.gather(img_feat.view(B, C, -1), -1, pcl_feat_index).view(B, C, N, -1)
        # pcl_feat = torch.sum(pcl_feat*pcl_closeness.unsqueeze(1), dim=-1).permute(0, 2, 1)
        #
        # """index token coordinate"""
        # pcl_index_weight = pcl_index.view(B, 1, -1).repeat(1, J, 1)
        # pcl_weight = torch.gather(img_offset[:, J*4:, :, :].view(B, J, -1), -1, pcl_index_weight).view(B, J, N, -1)
        # pcl_weight = torch.sum(pcl_weight*pcl_closeness.unsqueeze(1), dim=-1).permute(0, 2, 1)
        #
        # pcl_weight = pcl_weight.detach()            # B S N
        # pcl_offset_xyz = pcl_offset_xyz.detach()    # B S N*4(offset+dis)
        #
        # ################## ADD #####################
        # pcl_feat = self.pcl_xyz_emb(pcl.permute(0, 2, 1)).permute(0, 2, 1)\
        #            + self.pcl_feat_emb(pcl_feat.permute(0, 2, 1)).permute(0, 2, 1)\
        #            + self.pcl_pose_emb(torch.cat((pcl_weight, pcl_offset_xyz), dim=-1).permute(0, 2, 1)).permute(0, 2, 1)
        # pcl_feat = F.relu(pcl_feat)
        # attention = F.softmax(pcl_weight.permute(0, 2, 1), dim=-1)
        # joint_feat = torch.matmul(attention, pcl_feat)
        # joint_feat = self.joint_feat_emb(joint_feat.permute(0, 2, 1)).permute(0, 2, 1)\
        #            + self.joint_xyz_emb(joint_xyz.permute(0, 2, 1)).permute(0, 2, 1)
        # joint_feat = F.relu(joint_feat)

        # for i in range(self.num_stages):
        #     block = getattr(self, f"block{i + 1}")
        #
        #     pcl_feat, pcl_offset = block(pcl_feat, joint_feat, pcl, joint_xyz)
        #     if i < self.num_stages-1:
        #         pcl_weight = pcl_offset[:, :, J * 4:].view(B, N, J)
        #         # #################  ADD ##################
        #         attention = F.softmax(pcl_weight.permute(0, 2, 1), dim=-1)
        #         joint_feat_attn = torch.matmul(attention, pcl_feat)
        #         joint_xyz = pcl_offset2joint_weight(pcl_offset, pcl, self.kernel_size).detach()
        #
        #         # joint feat split emd
        #         joint_feat_emb = getattr(self, f"joint_feat_emb{i + 1}")
        #         joint_xyz_emb = getattr(self, f"joint_xyz_emb{i + 1}")
        #         joint_feat = joint_feat_emb(joint_feat_attn.permute(0, 2, 1)).permute(0, 2, 1) + \
        #                      joint_xyz_emb(joint_xyz.permute(0, 2, 1)).permute(0, 2, 1)
        #         joint_feat = F.relu(joint_feat)
        #
        #         # pcl feat split emd
        #         pcl_offset_remap = pcl_joint2offset(joint_xyz, pcl, 0.8)
        #         pcl_feat_emb = getattr(self, f"pcl_feat_emb{i + 1}")
        #         pcl_xyz_emb = getattr(self, f"pcl_xyz_emb{i + 1}")
        #         pcl_pose_emb = getattr(self, f"pcl_pose_emb{i + 1}")
        #         pcl_feat = pcl_feat_emb(pcl_feat.permute(0, 2, 1)).permute(0, 2, 1) + \
        #                    pcl_xyz_emb(pcl.permute(0, 2, 1)).permute(0, 2, 1) + \
        #                    pcl_pose_emb(torch.cat((pcl_weight, pcl_offset_remap), dim=-1).permute(0, 2, 1)).permute(0,
        #                                                                                                             2,
        #                                                                                                             1)
        #         pcl_feat = F.relu(pcl_feat)
        #
        #         result.append([pcl, pcl_offset])
        return result


class Block_MANO(nn.Module):
    def __init__(self, joint_num, dim,  kernel_size=0.8, dataset='nyu_all'):
        super().__init__()
        self.kernel_size = kernel_size
        self.joint_num = joint_num
        self.FA = DESA(dim, [128, 128], [64, 64, 64], [0.1, 0.2, 0.4])
        self.FP = RelationFeaturePropagation(dim, [dim], topk=4)
        self.joint_attn = NodeSimpleSemGCN(dim, joint_num, num_layers=3, dataset=dataset)
        self.PoolGCN = PoolSPGCN(dim, dim, dim, joint_num, dataset=dataset)
        self.mano_reg = nn.Sequential(nn.Linear(dim, 3 + 45 + 10 + 4))

        self.mesh_xyz_emb = nn.Sequential(nn.Conv1d(3, dim, 1), nn.BatchNorm1d(dim))
        self.mesh_pose_emb = nn.Sequential(nn.Conv1d(joint_num * 4, dim, 1), nn.BatchNorm1d(dim))
        self.token_reg = nn.Linear(dim, 5*joint_num)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, pcl_mesh_feat, joint_feat, pcl_mesh_xyz, joint_xyz, mano_feat, render, N):
        iter_num = 100
        tic = time.time()
        for i in range(iter_num + 1):
            joint_feat = self.FA(pcl_mesh_feat, joint_feat, pcl_mesh_xyz, joint_xyz)
        print('FA Time:%.3f' % ((time.time() - tic) / iter_num * 1000))

        tic = time.time()
        for i in range(iter_num + 1):
            if mano_feat is not None:
                mano_feat = self.PoolGCN(joint_feat) + mano_feat
            else:
                mano_feat = self.PoolGCN(joint_feat)
        print('PoolGCN Time:%.3f' % ((time.time() - tic) / iter_num * 1000))

        mano_para = self.mano_reg(mano_feat)

        tic = time.time()
        for i in range(iter_num + 1):
            joint_feat = self.joint_attn(joint_feat)
        print('SemGCN Time:%.3f' % ((time.time() - tic) / iter_num * 1000))

        tic = time.time()
        for i in range(iter_num + 1):
            # Add hand mesh
            mano_mesh, mano_joint = render.get_mesh(mano_para)
            mesh_center = torch.mean(mano_joint, dim=1, keepdim=True)
            joint_center = torch.mean(joint_xyz, dim=1, keepdim=True)
            mano_mesh = mano_mesh - mesh_center + joint_center.detach()
            mano_joint = mano_joint - mesh_center + joint_center.detach()
        print('MANO Time:%.3f' % ((time.time() - tic) / iter_num * 1000))

        tic = time.time()
        for i in range(iter_num + 1):
            # Add hand feat
            mesh_xyz = mano_mesh.detach()
            mesh_offset = pcl_joint2offset(joint_xyz, mesh_xyz, 0.8)
            mesh_feat = self.mesh_xyz_emb(mesh_xyz.permute(0, 2, 1)).permute(0, 2, 1) + \
                        self.mesh_pose_emb(mesh_offset.permute(0, 2, 1).detach()).permute(0, 2, 1)
            mesh_feat = F.relu(mesh_feat)
            pcl_feat = torch.split(pcl_mesh_feat, N, dim=1)[0]
            pcl = torch.split(pcl_mesh_xyz, N, dim=1)[0]
            pcl_mesh_xyz = torch.cat((pcl, mesh_xyz), dim=1)
            pcl_mesh_feat = torch.cat((pcl_feat, mesh_feat), dim=1)
        print('MeshFeat Time:%.3f' % ((time.time() - tic) / iter_num * 1000))


        tic = time.time()
        for i in range(iter_num + 1):
            pcl_mesh_feat = self.FP(pcl_mesh_xyz, joint_xyz, pcl_mesh_feat, joint_feat)
        print('FP Time:%.3f' % ((time.time() - tic) / iter_num * 1000))

        pcl_mesh_offset = self.token_reg(pcl_mesh_feat)

        return pcl_mesh_xyz, pcl_mesh_offset, \
               pcl_mesh_feat, joint_feat, \
               mano_para, mano_mesh, mano_joint, mano_feat


class IPNet_MANO(nn.Module):
    def __init__(self, net, joint_num, dataset, mano_dir, kernel_size=1):
        super(IPNet_MANO, self).__init__()
        self.joint_num = joint_num
        self.kernel_size = kernel_size
        self.dim = 128
        self.num_stages = 3
        if 'convnext' in net:
            self.backbone = convNeXTUnet(net, joint_num, pretrain='1k', deconv_dim=self.dim, out_dim_list=[joint_num*3, joint_num, joint_num])
        elif 'resnet' in net:
            self.backbone = OfficialResNetUnet(net, joint_num, pretrain=False, deconv_dim=self.dim, out_dim_list=[joint_num*3, joint_num, joint_num])

        self.render = Render(mano_dir, dataset)
        self.softmax = nn.Softmax(dim=-1)

        self.pcl_feat_emb = nn.Sequential(nn.Conv1d(self.dim, self.dim, 1), nn.BatchNorm1d(self.dim))
        self.pcl_xyz_emb = nn.Sequential(nn.Conv1d(3, self.dim, 1), nn.BatchNorm1d(self.dim))
        self.pcl_pose_emb = nn.Sequential(nn.Conv1d(self.joint_num * 5, self.dim, 1), nn.BatchNorm1d(self.dim))

        self.joint_feat_emb = nn.Sequential(nn.Conv1d(self.dim, self.dim, 1), nn.BatchNorm1d(self.dim))
        self.joint_xyz_emb = nn.Sequential(nn.Conv1d(3, self.dim, 1), nn.BatchNorm1d(self.dim))

        for i in range(self.num_stages):
            block = Block_MANO(joint_num=joint_num, dim=self.dim, kernel_size=self.kernel_size, dataset=dataset)
            # split emd
            pcl_feat_emb = nn.Sequential(nn.Conv1d(self.dim, self.dim, 1), nn.BatchNorm1d(self.dim))
            pcl_xyz_emb = nn.Sequential(nn.Conv1d(3, self.dim, 1), nn.BatchNorm1d(self.dim))
            pcl_pose_emb = nn.Sequential(nn.Conv1d(self.joint_num * 5, self.dim, 1), nn.BatchNorm1d(self.dim))
            joint_feat_emb = nn.Sequential(nn.Conv1d(self.dim, self.dim, 1), nn.BatchNorm1d(self.dim))
            joint_xyz_emb = nn.Sequential(nn.Conv1d(3, self.dim, 1), nn.BatchNorm1d(self.dim))
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"pcl_feat_emb{i + 1}", pcl_feat_emb)
            setattr(self, f"pcl_xyz_emb{i + 1}", pcl_xyz_emb)
            setattr(self, f"pcl_pose_emb{i + 1}", pcl_pose_emb)
            setattr(self, f"joint_feat_emb{i + 1}", joint_feat_emb)
            setattr(self, f"joint_xyz_emb{i + 1}", joint_xyz_emb)
    def forward(self, img, pcl, loader, center, M, cube, cam_para, kernel):
        img_offset, img_feat = self.backbone(img)   # img_offset: B×C×W×H , C=3(direct vector)+1(heatmap)+1(weight)
        joint_uvd = offset2joint_weight(img_offset, img, kernel)
        B, C, H, W = img_feat.size()
        img_down = F.interpolate(img, [H, W])
        result = [img_offset]
        joint_uvd = joint_uvd.detach()
        img_offset = img_offset.detach()

        # 提取点云特征
        B, N, _ = pcl.size()
        J = self.joint_num

        joint_xyz = loader.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para)
        pcl_offset_xyz = pcl_joint2offset(joint_xyz, pcl, 0.8)

        pcl_closeness, pcl_index = loader.img2pcl_index(pcl, img_down, center, M, cube, cam_para, select_num=4)
        pcl_feat_index = pcl_index.view(B, 1, -1).repeat(1, C, 1)   # B*128*(K*1024)
        pcl_feat = torch.gather(img_feat.view(B, C, -1), -1, pcl_feat_index).view(B, C, N, -1)
        pcl_feat = torch.sum(pcl_feat*pcl_closeness.unsqueeze(1), dim=-1).permute(0, 2, 1)

        """index token coordinate"""
        pcl_index_weight = pcl_index.view(B, 1, -1).repeat(1, J, 1)
        pcl_weight = torch.gather(img_offset[:, J*4:, :, :].view(B, J, -1), -1, pcl_index_weight).view(B, J, N, -1)
        pcl_weight = torch.sum(pcl_weight*pcl_closeness.unsqueeze(1), dim=-1).permute(0, 2, 1)

        pcl_weight = pcl_weight.detach()            # B S N
        pcl_offset_xyz = pcl_offset_xyz.detach()    # B S N*4(offset+dis)

        # 解耦式生成点云特征
        pcl_feat = self.pcl_feat_emb(pcl_feat.permute(0, 2, 1)).permute(0, 2, 1)+\
                   self.pcl_xyz_emb(pcl.permute(0, 2, 1)).permute(0, 2, 1)+ \
                   self.pcl_pose_emb(torch.cat((pcl_weight, pcl_offset_xyz), dim=-1).permute(0, 2, 1)).permute(0, 2, 1)
        pcl_feat = F.relu(pcl_feat)

        attention = F.softmax(pcl_weight.permute(0, 2, 1), dim=-1)
        joint_feat = torch.matmul(attention, pcl_feat)

        joint_feat = self.joint_feat_emb(joint_feat.permute(0, 2, 1)).permute(0, 2, 1)+\
                   self.joint_xyz_emb(joint_xyz.detach().permute(0, 2, 1)).permute(0, 2, 1)
        joint_feat = F.relu(joint_feat)

        mano_feat = None
        pcl_mesh_feat = pcl_feat
        pcl_mesh_xyz = pcl
        for i in range(self.num_stages):
            block = getattr(self, f"block{i + 1}")
            iter_num = 100
            tic = time.time()
            for i in range(iter_num + 1):
                pcl_mesh_xyz, pcl_mesh_offset, pcl_mesh_feat, joint_feat, mano_para, mano_mesh, mano_joint, mano_feat = \
                    block(pcl_mesh_feat, joint_feat, pcl_mesh_xyz, joint_xyz.detach(), mano_feat, self.render, N)
            print('Block Time:%.3f'%((time.time() - tic)/iter_num*1000))
            # pcl_mesh_xyz, pcl_mesh_offset, pcl_mesh_feat, joint_feat, mano_para, mano_mesh, mano_joint, mano_feat = \
            #     block(pcl_mesh_feat, joint_feat, pcl_mesh_xyz, joint_xyz.detach(), mano_feat, self.render, N)

            if i < self.num_stages - 1:
                pcl_mesh_weight = pcl_mesh_offset[:, :, J * 4:].view(B, -1, J)
                joint_xyz = pcl_offset2joint_weight(pcl_mesh_offset, pcl_mesh_xyz, self.kernel_size).detach()
                result.append([pcl_mesh_xyz, pcl_mesh_offset])

                # joint feat split emd
                joint_feat_emb = getattr(self, f"joint_feat_emb{i + 1}")
                joint_xyz_emb = getattr(self, f"joint_xyz_emb{i + 1}")
                attention = F.softmax(pcl_mesh_weight.permute(0, 2, 1), dim=-1)
                joint_feat_attn = torch.matmul(attention, pcl_mesh_feat)
                joint_feat = joint_feat_emb(joint_feat_attn.permute(0, 2, 1)).permute(0, 2, 1) + \
                             joint_xyz_emb(joint_xyz.permute(0, 2, 1).detach()).permute(0, 2, 1)
                joint_feat = F.relu(joint_feat)

                # pcl feat split emd
                pcl_mesh_offset_remap = pcl_joint2offset(joint_xyz, pcl_mesh_xyz, 0.8)
                pcl_feat_emb = getattr(self, f"pcl_feat_emb{i + 1}")
                pcl_xyz_emb = getattr(self, f"pcl_xyz_emb{i + 1}")
                pcl_pose_emb = getattr(self, f"pcl_pose_emb{i + 1}")
                pcl_mesh_feat = pcl_feat_emb(pcl_mesh_feat.permute(0, 2, 1)).permute(0, 2, 1) + \
                                pcl_xyz_emb(pcl_mesh_xyz.permute(0, 2, 1).detach()).permute(0, 2, 1) + \
                                pcl_pose_emb(torch.cat((pcl_mesh_weight, pcl_mesh_offset_remap), dim=-1).permute(0, 2,
                                                                                                                 1).detach()).permute(
                                    0, 2, 1)
                pcl_mesh_feat = F.relu(pcl_mesh_feat)

                result.append([mano_mesh, mano_joint, mano_para])
        return result


class Conv_PointMLP(nn.Module):
    def __init__(self, net, joint_num, kernel_size=1):
        super(Conv_PointMLP, self).__init__()
        self.joint_num = joint_num
        self.kernel_size = kernel_size
        self.dim = 128
        self.num_stages = 1
        if 'convnext' in net:
            self.backbone = convNeXTUnet(net, joint_num, pretrain='1k', deconv_dim=self.dim, out_dim_list=[joint_num*3, joint_num, joint_num])
        elif 'resnet' in net:
            self.backbone = OfficialResNetUnet(net, joint_num, pretrain=True, deconv_dim=self.dim, out_dim_list=[joint_num*3, joint_num, joint_num])

        self.emd_dim = 128
        self.pcl_feat_emb = nn.Sequential(nn.Conv1d(self.dim, self.emd_dim, 1), nn.BatchNorm1d(self.emd_dim))
        self.pcl_xyz_emb = nn.Sequential(nn.Conv1d(3, self.emd_dim, 1), nn.BatchNorm1d(self.emd_dim))
        self.pcl_pose_emb = nn.Sequential(nn.Conv1d(self.joint_num * 5, self.emd_dim, 1), nn.BatchNorm1d(self.emd_dim))
        self.joint_feat_emb = nn.Sequential(nn.Conv1d(self.dim, self.emd_dim, 1), nn.BatchNorm1d(self.emd_dim))
        self.joint_xyz_emb = nn.Sequential(nn.Conv1d(3, self.emd_dim, 1), nn.BatchNorm1d(self.emd_dim))

        self.pointMLP = PointMLP_refine(joint_num, embed_dim=self.emd_dim, dim_expansion=[1, 1, 2, 2])

    def forward(self, img, pcl, loader, center, M, cube, cam_para, kernel):
        img_offset, img_feat = self.backbone(img)   # img_offset: B×C×W×H , C=3(direct vector)+1(heatmap)+1(weight)
        joint_uvd = offset2joint_weight(img_offset, img, kernel)
        B, C, H, W = img_feat.size()
        img_down = F.interpolate(img, [H, W])

        result = [img_offset]
        joint_uvd = joint_uvd.detach()
        img_offset = img_offset.detach()

        # 提取点云特征
        B, N, _ = pcl.size()
        J = self.joint_num

        joint_xyz = loader.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para).detach()
        pcl_offset_xyz = pcl_joint2offset(joint_xyz, pcl, 0.8)  # 反向传播信息的时，采用固定的范围

        # pcl_closeness, pcl_index = loader.img2pcl_index_softmax(pcl, img_down, center, M, cube, cam_para, select_num=8,scale=10)
        pcl_closeness, pcl_index = loader.img2pcl_index(pcl, img_down, center, M, cube, cam_para, select_num=4)
        # pcl_closeness, pcl_index = loader.pcl2img_index(pcl, H, center, M, cube, cam_para, select_num=4)
        pcl_feat_index = pcl_index.view(B, 1, -1).repeat(1, C, 1)   # B*128*(K*1024)
        pcl_feat = torch.gather(img_feat.view(B, C, -1), -1, pcl_feat_index).view(B, C, N, -1)
        pcl_feat = torch.sum(pcl_feat*pcl_closeness.unsqueeze(1), dim=-1).permute(0, 2, 1)

        """index token coordinate"""
        pcl_index_weight = pcl_index.view(B, 1, -1).repeat(1, J, 1)
        pcl_weight = torch.gather(img_offset[:, J*4:, :, :].view(B, J, -1), -1, pcl_index_weight).view(B, J, N, -1)
        pcl_weight = torch.sum(pcl_weight*pcl_closeness.unsqueeze(1), dim=-1).permute(0, 2, 1)

        pcl_weight = pcl_weight.detach()            # B S N
        # pcl_feat = pcl_feat.detach()                # B S C
        pcl_offset_xyz = pcl_offset_xyz.detach()    # B S N*4(offset+dis)

        ################## ADD #####################
        pcl_feat = self.pcl_xyz_emb(pcl.permute(0, 2, 1)).permute(0, 2, 1)\
                   + self.pcl_feat_emb(pcl_feat.permute(0, 2, 1)).permute(0, 2, 1)\
                   + self.pcl_pose_emb(torch.cat((pcl_weight, pcl_offset_xyz), dim=-1).permute(0, 2, 1)).permute(0, 2, 1)
        pcl_feat = F.relu(pcl_feat)

        pcl_offset = self.pointMLP(pcl, pcl_feat.permute(0, 2, 1))
        result.append([pcl, pcl_offset])
        return result


class Conv_Point2(nn.Module):
    def __init__(self, net, joint_num, kernel_size=1):
        super(Conv_Point2, self).__init__()
        self.joint_num = joint_num
        self.kernel_size = kernel_size
        self.dim = 128
        self.num_stages = 1
        if 'convnext' in net:
            self.backbone = convNeXTUnet(net, joint_num, pretrain='1k', deconv_dim=self.dim, out_dim_list=[joint_num*3, joint_num, joint_num])
        elif 'resnet' in net:
            self.backbone = OfficialResNetUnet(net, joint_num, pretrain=True, deconv_dim=self.dim, out_dim_list=[joint_num*3, joint_num, joint_num])

        self.emd_dim = 128
        self.pcl_feat_emb = nn.Sequential(nn.Conv1d(self.dim, self.emd_dim, 1), nn.BatchNorm1d(self.emd_dim))
        self.pcl_xyz_emb = nn.Sequential(nn.Conv1d(3, self.emd_dim, 1), nn.BatchNorm1d(self.emd_dim))
        self.pcl_pose_emb = nn.Sequential(nn.Conv1d(self.joint_num * 5, self.emd_dim, 1), nn.BatchNorm1d(self.emd_dim))
        self.joint_feat_emb = nn.Sequential(nn.Conv1d(self.dim, self.emd_dim, 1), nn.BatchNorm1d(self.emd_dim))
        self.joint_xyz_emb = nn.Sequential(nn.Conv1d(3, self.emd_dim, 1), nn.BatchNorm1d(self.emd_dim))

        self.point2 = PointNet2SemSegMSG_refine()

    def forward(self, img, pcl, loader, center, M, cube, cam_para, kernel):
        img_offset, img_feat = self.backbone(img)   # img_offset: B×C×W×H , C=3(direct vector)+1(heatmap)+1(weight)
        joint_uvd = offset2joint_weight(img_offset, img, kernel)
        B, C, H, W = img_feat.size()
        img_down = F.interpolate(img, [H, W])

        result = [img_offset]
        joint_uvd = joint_uvd.detach()
        img_offset = img_offset.detach()

        # 提取点云特征
        B, N, _ = pcl.size()
        J = self.joint_num

        joint_xyz = loader.uvd_nl2xyznl_tensor(joint_uvd, center, M, cube, cam_para).detach()
        pcl_offset_xyz = pcl_joint2offset(joint_xyz, pcl, 0.8)  # 反向传播信息的时，采用固定的范围

        # pcl_closeness, pcl_index = loader.img2pcl_index_softmax(pcl, img_down, center, M, cube, cam_para, select_num=8,scale=10)
        pcl_closeness, pcl_index = loader.img2pcl_index(pcl, img_down, center, M, cube, cam_para, select_num=4)
        # pcl_closeness, pcl_index = loader.pcl2img_index(pcl, H, center, M, cube, cam_para, select_num=4)
        pcl_feat_index = pcl_index.view(B, 1, -1).repeat(1, C, 1)   # B*128*(K*1024)
        pcl_feat = torch.gather(img_feat.view(B, C, -1), -1, pcl_feat_index).view(B, C, N, -1)
        pcl_feat = torch.sum(pcl_feat*pcl_closeness.unsqueeze(1), dim=-1).permute(0, 2, 1)

        """index token coordinate"""
        pcl_index_weight = pcl_index.view(B, 1, -1).repeat(1, J, 1)
        pcl_weight = torch.gather(img_offset[:, J*4:, :, :].view(B, J, -1), -1, pcl_index_weight).view(B, J, N, -1)
        pcl_weight = torch.sum(pcl_weight*pcl_closeness.unsqueeze(1), dim=-1).permute(0, 2, 1)

        pcl_weight = pcl_weight.detach()            # B S N
        # pcl_feat = pcl_feat.detach()                # B S C
        pcl_offset_xyz = pcl_offset_xyz.detach()    # B S N*4(offset+dis)

        ################## ADD #####################
        pcl_feat = self.pcl_xyz_emb(pcl.permute(0, 2, 1)).permute(0, 2, 1)\
                   + self.pcl_feat_emb(pcl_feat.permute(0, 2, 1)).permute(0, 2, 1)\
                   + self.pcl_pose_emb(torch.cat((pcl_weight, pcl_offset_xyz), dim=-1).permute(0, 2, 1)).permute(0, 2, 1)
        pcl_feat = F.relu(pcl_feat)

        pcl_offset = self.point2(pcl, pcl_feat.permute(0, 2, 1).contiguous()).permute(0,2,1)
        # print(pcl.size())
        result.append([pcl, pcl_offset])
        return result


def img2pcl(img):
    B, _, W, H = img.size()
    device = img.device
    mesh_x = 2.0 * (torch.arange(W).unsqueeze(1).expand(W, W).float() + 0.5) / W - 1.0
    mesh_y = 2.0 * (torch.arange(W).unsqueeze(0).expand(W, W).float() + 0.5) / W - 1.0
    coords = torch.stack((mesh_y, mesh_x), dim=0)
    coords = torch.unsqueeze(coords, dim=0).repeat(B, 1, 1, 1).to(device)
    img_uvd = torch.cat((coords, img), dim=1).view(B, 3, H * W).permute(0, 2, 1)
    return img_uvd


def joint2offset(joint, img, kernel_size, feature_size):
    device = joint.device
    batch_size, _, img_height, img_width = img.size()
    img = F.interpolate(img, size=[feature_size, feature_size])
    _, joint_num, _ = joint.view(batch_size, -1, 3).size()
    joint_feature = joint.reshape(joint.size(0), -1, 1, 1).repeat(1, 1, feature_size, feature_size)
    mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
    mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
    coords = torch.stack((mesh_y, mesh_x), dim=0)
    coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
    coords = torch.cat((coords, img), dim=1).repeat(1, joint_num, 1, 1)
    offset = joint_feature - coords
    offset = offset.view(batch_size, joint_num, 3, feature_size, feature_size)
    dist = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=2))
    offset_norm = (offset / (dist.unsqueeze(2)))

    heatmap = (kernel_size - dist) / kernel_size

    mask = heatmap.ge(0).float() * img.lt(0.99).float().view(batch_size, 1, feature_size, feature_size)
    offset_norm_mask = (offset_norm * mask.unsqueeze(2)).view(batch_size, -1, feature_size, feature_size)
    heatmap_mask = heatmap * mask.float()
    return torch.cat((offset_norm_mask, heatmap_mask), dim=1)


def offset2joint_weight(offset, depth, kernel_size):
    device = offset.device
    batch_size, joint_num, feature_size, feature_size = offset.size()
    joint_num = int(joint_num / 5)
    if depth.size(-1) != feature_size:  # 下采样深度图
        depth = F.interpolate(depth, size=[feature_size, feature_size])

    offset_unit = offset[:, :joint_num * 3, :, :].contiguous()  # b * (3*J) * fs * fs
    heatmap = offset[:, joint_num * 3:joint_num * 4, :, :].contiguous()
    weight = offset[:, joint_num * 4:, :, :].contiguous()

    mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size,feature_size).float() + 0.5) / feature_size - 1.0
    mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size,feature_size).float() + 0.5) / feature_size - 1.0
    coords = torch.stack((mesh_y, mesh_x), dim=0)
    coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
    coords = torch.cat((coords, depth), dim=1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 3, -1)

    mask = depth.lt(0.99).float()
    offset_mask = (offset_unit * mask).view(batch_size, joint_num, 3, -1)   # 截取深度图中有值的部分
    heatmap_mask = (heatmap * mask).view(batch_size, joint_num, -1)
    weight_mask = weight.masked_fill(depth.gt(0.99), -1e8)
    normal_weight = F.softmax(weight_mask.view(batch_size, joint_num, -1), dim=-1)  # b * J * fs^2

    if torch.is_tensor(kernel_size):
        kernel_size = kernel_size.to(device)
        dist = kernel_size.view(1, joint_num, 1) - heatmap_mask * kernel_size.view(1, joint_num, 1)
    else:
        dist = kernel_size - heatmap_mask * kernel_size

    joint = torch.sum((offset_mask * dist.unsqueeze(2).repeat(1, 1, 3, 1) + coords) * normal_weight.unsqueeze(2).repeat(1, 1, 3, 1), dim=-1)
    return joint


def pcl_joint2offset(joint, pcl, kernel_size):
    """
    :param: joint BxJx3--xyz坐标
    :param: pcl BxNx3
    """
    B, J, _ = joint.size()
    N = pcl.size(1)
    device = joint.device
    offset = joint.unsqueeze(2) - pcl.unsqueeze(1)  # B J 1 3 - B 1 N 3 -> B J N 3
    dis = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=-1))   # B J N
    offset_norm = offset / (dis.unsqueeze(-1) + 1e-8)
    offset_norm = offset_norm.permute(0, 1, 3, 2).reshape(B, J * 3, N)

    if torch.is_tensor(kernel_size):
        kernel_size = kernel_size.to(device)
        dis = (kernel_size.view(1, J, 1) - dis) / kernel_size.view(1, J, 1)
    else:
        dis = (kernel_size - dis) / kernel_size

    mask = dis.ge(0).float() * pcl[:, :, 2].lt(0.99).float().unsqueeze(1)
    dis = dis * mask    # closeness map
    offset_norm = offset_norm * mask.view(B, J, 1, N).repeat(1, 1, 3, 1).reshape(B, -1, N)  # 3D directional unit vector
    return torch.cat((offset_norm, dis), dim=1).to(device).permute(0, 2, 1)


def pcl_offset2joint_weight(pcl_result, pcl, kernel_size):
    """
    :param: pcl_result BxNx(5*J)
    :param: pcl BxNx3
    """
    assert pcl.size(2) == 3
    pcl_result = pcl_result.permute(0, 2, 1)
    B, J, N = pcl_result.size()
    J = int(J / 5)
    device = pcl.device

    coords = pcl.permute(0, 2, 1).reshape(B, 1, 3, N)
    offset = pcl_result[:, :J * 3, :].view(B, J, 3, N)
    heatmap = pcl_result[:, J * 3:J * 4, :].view(B, J, 1, N)
    weight = pcl_result[:, J * 4:, :].view(B, J, 1, N)

    mask = pcl[:, :, 2].gt(0.99).view(B, 1, 1, N)
    weight_mask = torch.masked_fill(weight, mask, -1e8)
    normal_weight = F.softmax(weight_mask, dim=-1)

    if torch.is_tensor(kernel_size):
        kernel_size = kernel_size.to(device)
        dist = kernel_size.view(1, J, 1) - heatmap * kernel_size.view(1, J, 1)
    else:
        dist = kernel_size - heatmap * kernel_size

    joint = torch.sum((offset * dist + coords) * normal_weight, dim=-1)
    return joint


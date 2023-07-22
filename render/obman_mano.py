import torch
import torch.nn as nn
import numpy as np
from manopth.manolayer import ManoLayer
from pytorch3d.structures.meshes import Meshes
import pytorch3d.ops as ops

OBMAN2MANO = [
    0,
    5,6,7,
    9,10,11,
    17,18,19,
    13,14,15,
    1,2,3,
    8,12,20,16,4
]

def xyz2sphere(xyz, normalize=True):
    """
    Convert XYZ to Spherical Coordinate

    reference: https://en.wikipedia.org/wiki/Spherical_coordinate_system

    :param xyz: [B, N, 3] / [B, N, G, 3]
    :return: (rho, theta, phi) [B, N, 3] / [B, N, G, 3]
    """
    rho = torch.sqrt(torch.sum(torch.pow(xyz, 2), dim=-1, keepdim=True))
    rho = torch.clamp(rho, min=0)  # range: [0, inf]
    theta = torch.acos(xyz[..., 2, None] / rho)  # range: [0, pi]
    phi = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])  # range: [-pi, pi]
    # check nan
    idx = rho == 0
    theta[idx] = 0

    if normalize:
        theta = theta / np.pi  # [0, 1]
        phi = phi / (2 * np.pi) + .5  # [0, 1]
    out = torch.cat([rho, theta, phi], dim=-1)
    return out

def resort_points(points, idx):
    """
    Resort Set of points along G dim

    """
    device = points.device
    N, G, _ = points.shape

    view_shape = [N, 1]
    repeat_shape = [1, G]
    n_indices = torch.arange(N, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[n_indices, idx, :]

    return new_points


class Render(nn.Module):
    def __init__(self, mano_path, dataset, flat=True, use_pca=False):
        super(Render, self).__init__()
        self.mano_layer = ManoLayer(mano_root=mano_path, flat_hand_mean=flat, use_pca=use_pca, side='right')
        self.cam_extr = torch.from_numpy(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])).view(1, 3, 3).float()
        if 'hands' in dataset:
            self.transfer = OBMAN2MANO
        elif 'nyu' in dataset:
            self.transfer = OBMAN2MANO
        else:
            self.transfer = range(21)
        self.register_umbrella()
        self.register_finger_mesh()

    def get_mesh(self, model_paras, s=125):
        B = model_paras.size(0)
        theta = model_paras[:, :48]
        beta = model_paras[:, 48:58]
        trans = model_paras[:, 58:61]
        scale = model_paras[:, 61:62]
        hand_mesh, hand_joint = self.mano_layer(th_pose_coeffs=theta, th_betas=beta)
        hand_mesh = torch.matmul(hand_mesh, self.cam_extr.to(model_paras.device))
        hand_joint = torch.matmul(hand_joint, self.cam_extr.to(model_paras.device))

        center = torch.mean(hand_joint, dim=1, keepdim=True)
        hand_mesh = (hand_mesh - center) / s
        hand_joint = (hand_joint - center) / s
        hand_mesh = hand_mesh * scale.view(B, 1, 1) + trans.view(B, 1, 3)
        hand_joint = hand_joint * scale.view(B, 1, 1) + trans.view(B, 1, 3)

        return hand_mesh, hand_joint[:, self.transfer, :]

    def sample_finger_pcl_from_mesh(self, hand_mesh, num_samples, pcl=None):
        batch_size = hand_mesh.size(0)
        finger_mesh = torch.index_select(hand_mesh, dim=1, index=self.finger_index)
        pytorch3d_mesh = Meshes(finger_mesh, self.finger_face.unsqueeze(0).repeat(batch_size, 1, 1))
        pcl_sample = ops.sample_points_from_meshes(pytorch3d_mesh, num_samples=num_samples)
        return pcl_sample

    def sample_pcl_from_mesh(self, hand_mesh, num_samples, pcl=None):
        batch_size = hand_mesh.size(0)
        pytorch3d_mesh = Meshes(hand_mesh, self.mano_layer.th_faces.unsqueeze(0).repeat(batch_size, 1, 1))
        pcl_sample = ops.sample_points_from_meshes(pytorch3d_mesh, num_samples=num_samples)
        return pcl_sample

    def sample_pcl_from_para(self, model_paras, center3d, cube_size, num_samples, pcl=None):
        batch_size = model_paras.size(0)
        pose_dim = model_paras.size(-1) - 17
        quat = model_paras[:, :3]
        theta = model_paras[:, 3:3+pose_dim]
        beta = model_paras[:, 3+pose_dim:3+pose_dim+10]
        cam = model_paras[:, 3 + pose_dim + 10:]
        hand_mesh, hand_joint = self.mano_layer.get_mano_vertices(quat, theta, beta, cam)
        hand_mesh = (hand_mesh - center3d.unsqueeze(1)) / cube_size.unsqueeze(1) * 2
        hand_joint = (hand_joint - center3d.unsqueeze(1)) / cube_size.unsqueeze(1) * 2
        pytorch3d_mesh = Meshes(hand_mesh, self.mano_layer.th_faces.unsqueeze(0).repeat(batch_size, 1, 1))
        pcl_sample = ops.sample_points_from_meshes(pytorch3d_mesh, num_samples=num_samples)
        return pcl_sample

    def sample_finger_pcl_from_para(self, model_paras, center3d, cube_size, num_samples, pcl=None):
        batch_size = model_paras.size(0)
        pose_dim = model_paras.size(-1) - 17
        quat = model_paras[:, :3]
        theta = model_paras[:, 3:3+pose_dim]
        beta = model_paras[:, 3+pose_dim:3+pose_dim+10]
        cam = model_paras[:, 3 + pose_dim + 10:]
        hand_mesh, hand_joint = self.mano_layer.get_mano_vertices(quat, theta, beta, cam)
        hand_mesh = (hand_mesh - center3d.unsqueeze(1)) / cube_size.unsqueeze(1) * 2
        hand_joint = (hand_joint - center3d.unsqueeze(1)) / cube_size.unsqueeze(1) * 2
        finger_mesh = torch.index_select(hand_mesh, dim=-1, index=self.finger_index)
        pytorch3d_mesh = Meshes(finger_mesh, self.finger_face.unsqueeze(0).repeat(batch_size, 1, 1))
        pcl_sample = ops.sample_points_from_meshes(pytorch3d_mesh, num_samples=num_samples)
        return pcl_sample

    def register_finger_mesh(self):
        weights = self.mano_layer.th_weights.clone()
        V, J = weights.size()
        vertex_seg = torch.argmax(weights, dim=-1)

        joint_vertex_list = []
        for index in range(1, J):
            joint_vertex_list.append(vertex_seg.eq(index).nonzero().squeeze(-1))

        # joint_vertex_list = []
        # for index in range(1, J):
        #     joint_vertex_list.append(weights[:, index].gt(0.1).nonzero().squeeze())

        joint_vertex = torch.cat(joint_vertex_list, dim=0)
        joint_face_list = []
        for face in self.mano_layer.th_faces:
            if face[0] in joint_vertex and face[1] in joint_vertex and face[2] in joint_vertex:
                a = (joint_vertex == face[0]).nonzero().item()
                b = (joint_vertex == face[1]).nonzero().item()
                c = (joint_vertex == face[2]).nonzero().item()
                joint_face_list.append(torch.Tensor([a, b, c]))
        joint_face = torch.stack(joint_face_list, dim=0).long()

        self.register_buffer('finger_index', joint_vertex)
        self.register_buffer('finger_face', joint_face)

    def register_umbrella(self):
        v_umbrella_list = []
        mesh = self.mano_layer.th_v_template.squeeze(0)
        face = self.mano_layer.th_faces
        max_face = 0
        min_face = 100
        for v_index in range(mesh.size(0)):
            f_id = (v_index == face).nonzero()[:, 0]
            f = face[f_id, :].view(-1)
            f = f[f != v_index].view(-1, 2)
            f_unique = torch.unique(f)
            f_num = torch.zeros_like(f_unique)
            for ii, f_index in enumerate(f_unique):
                f_num[ii] = (f == f_index).sum()
            if f_num[f_num == 1].sum() > 0:
                start_id = f_unique[f_num == 1][0]
            else:
                start_id = f_unique[0]

            face_num = f.size(0)
            sort_idx_list = [start_id]
            for ii in range(face_num):
                face_index = torch.arange(0, f.size(0))[(f == start_id).sum(-1).bool()][0].long()
                face_select = f[face_index]
                start_id = face_select[face_select != start_id]
                sort_idx_list.append(start_id.item())
                f = torch.cat((f[:face_index], f[face_index+1:]), dim=0)

            sort_idx = []
            for ii in range(len(sort_idx_list)-1):
                sort_idx.append(torch.Tensor([v_index, sort_idx_list[ii], sort_idx_list[ii+1]]))
            v_umbrella_list.append(torch.stack(sort_idx, dim=0))
            if (len(sort_idx_list)-1) > max_face:
                max_face = len(sort_idx_list)-1
            if (len(sort_idx_list)-1) < min_face:
                min_face = len(sort_idx_list)-1

        v_umbrella_expand_list = []
        mask_list = []

        for ii, face in enumerate(v_umbrella_list):
            face_num = face.size(0)
            n = max_face - face_num
            face_expand = torch.cat((face, torch.ones(n, 3)*ii), dim=0)
            # n = max_face // face_num
            # m = max_face - face_num * n
            # face_expand = face.unsqueeze(0).repeat(n, 1, 1).view(n*face_num, 3)
            # face_expand = torch.cat((face_expand, face[:m, :]), dim=0)
            v_umbrella_expand_list.append(face_expand)
            mask_ones = torch.ones([face_num])
            mask_zeros = torch.zeros([n])
            mask = torch.cat((mask_ones, mask_zeros), dim=0)
            mask_list.append(mask)
        # print(v_umbrella_list)
        umbrella_idx = torch.stack(v_umbrella_expand_list, dim=0).long()
        umbrella_mask = torch.stack(mask_list, dim=0)

        self.register_buffer('umbrella_idx', umbrella_idx)
        self.register_buffer('umbrella_mask', umbrella_mask)

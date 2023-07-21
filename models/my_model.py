from datasets import points_utils
from models import my_base_model
from models.backbone.pointnet_new import Pointnet_Backbone, MiniPointNet, SegPointNet
from utils.metrics import estimateOverlap, estimateAccuracy
import torch
from torch import nn
import torch.nn.functional as F
from ipdb import set_trace
from nuscenes.utils.data_classes import LidarPointCloud
from datasets.data_classes import PointCloud, Box
import numpy as np
import os
import pickle
import copy
from models.head.attention import SelfAttention

class MyModel(my_base_model.BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        
        self.T = 41
        self.feature_num = 512
        self.save_flag = False
        self.save_path = "train_save_bbox/"
        self.output = []
        self.index = 0
        self.search_scale = getattr(config, 'bb_scale', 1.0)
        self.search_offset = getattr(config, 'bb_offset', 0)
        self.max_frame_num = getattr(config, 'max_frame_num', 0)
        self.max_point_num = getattr(config, 'max_point_num', 0)
        self.point_sample_size = getattr(config, 'point_sample_size', 0)
        
        self.seg_pointnet = SegPointNet(input_channel=3 + 1,
                                        per_point_mlp1=[64, 64, 64, 128, 1024],
                                        per_point_mlp2=[512, 256, 128, 128],
                                        output_size=2)
        
        self.pc_pointnet = MiniPointNet(input_channel=3 + 1,
                                          per_point_mlp=[64, 128, 256, 512],
                                          hidden_mlp=[512, 256],
                                          output_size=-1)
        
        self.bbox_pointnet = MiniPointNet(input_channel=3 + 2,
                                          per_point_mlp=[64, 64, 128, 512],
                                          hidden_mlp=[128, 128],
                                          output_size=-1)

        self.selfattn = SelfAttention(n_head=8, d_k=256, d_v=256, d_x=256, d_o=256)
        
        self.point_mlp = nn.Sequential(nn.Linear(256, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 4))
        
    def forward(self, input_dict):
        """
        Args:
            input_dict: {
            "points": (B,N,3+1)
            }
        """
        output_dict = {}
        
        x = input_dict["pc_data"].cuda()
        B, T, N, _ = x.shape
        x = x.reshape(B, -1, 4)
        # seg
        x = x.transpose(1, 2)
        seg_out = self.seg_pointnet(x)
        seg_logits = seg_out[:, :2, :]  # B,2,N
        output_dict['seg_logits'] = seg_logits
        pred_cls = torch.argmax(seg_logits, dim=1, keepdim=True)  # B,1,N
        seg_points = x[:, :4, :] * pred_cls
        seg_points = seg_points.reshape(-1, 4, 1024)
        
        # encode point feature
        point_feature = self.pc_pointnet(seg_points)
        point_feature = point_feature.view(B, T, -1)
        attn, point_feature = self.selfattn(point_feature)
        point_feature = point_feature.view(-1, 256)
        
        point_output = self.point_mlp(point_feature)
        
        point_output[:, 3] = 0
        output_dict['point_output'] = point_output
        
        return output_dict
    
    def compute_loss(self, data, output):
        loss_total = 0.0
        loss_dict = {}
        point_output = output['point_output'] # B,4
        seg_logits = output['seg_logits']
        with torch.no_grad():
            box_label = data['gt_delta'].view(-1, 4)
            frame_num = data['frame_num']
            center_label = box_label[:, :3]
            angle_label = box_label[:, 3]
            bbox_mask = data['bbox_mask'].view(-1, 1)
            seg_label = data['seg_label']
        
        point_output = point_output * bbox_mask
        center_label = center_label * bbox_mask
        bbox_mask = bbox_mask.squeeze()
        point_output = point_output[bbox_mask>0]
        center_label = center_label[bbox_mask>0]
        loss_center = F.smooth_l1_loss(point_output[:, :3], center_label)

        loss_seg = F.cross_entropy(seg_logits, seg_label, weight=torch.tensor([0.5, 2.0]).cuda())
        loss_total = 0.1 * loss_seg + loss_center
        
        loss_dict['loss_total'] = loss_total
        loss_dict["loss_center"] = loss_center
        loss_dict['loss_seg'] = loss_seg
        
        return loss_dict
        
    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss_dict = self.compute_loss(batch, output)
        loss = loss_dict['loss_total']
        
        log_dict = {k: v.item() for k, v in loss_dict.items()}

        self.logger.experiment.add_scalars('loss', log_dict,
                                           global_step=self.global_step)
        return loss
    
    def evaluate_one_sequence(self, sequence):
        """
        :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
        :return:
        """
        ious = []
        distances = []
        results_bbs = []
        
        bbox_label = sequence['gt_track']
        
        if self.save_flag:
            bbox_corners = [bbs.corners() for bbs in sequence['track']]
            np.save(self.save_path + str(self.index) + '_input_bbox.npy', np.array(bbox_corners))
            input_centers = [bbs.center for bbs in sequence['track']]
            np.save(self.save_path + str(self.index) + '_input_center.npy', np.array(input_centers))
        
        input_dict = self.build_input_dict(sequence)
        ref_box = input_dict['ref_box']
        output = self(input_dict)
        offset = output['point_output'][:, :3].cpu()
        
        if self.save_flag:
            seg_logits = output['seg_logits']
            pred_cls = torch.argmax(seg_logits, dim=1, keepdim=True).view(-1)
            seg_points = input_dict['pc_data'].view(-1,4)[pred_cls == 1][:, :3].cpu().numpy()
            seg_pc = PointCloud(points=seg_points.T)
            seg_pc.rotate(ref_box.rotation_matrix)
            seg_pc.translate(ref_box.center)
            np.save(self.save_path + str(self.index) + '_seg_point.npy', seg_pc.points.T)
        
        
        for i in range(len(bbox_label)):
            if (i == 0):
                output_bbox = bbox_label[i]
            else:
                w, l, h = bbox_label[i].wlh
                estimation_box = input_dict['track_bbox'][0][i, :3].cpu() + offset[i] / np.array((1, 1, 2)) * np.array((l, w, h))
                output_bbox = points_utils.getOffsetBB(ref_box, estimation_box, degrees=self.config.degrees,
                                                 use_z=self.config.use_z,
                                                 limit_box=self.config.limit_box)
            
            this_overlap = estimateOverlap(bbox_label[i], output_bbox, dim=self.config.IoU_space,
                                           up_axis=self.config.up_axis)
            this_accuracy = estimateAccuracy(bbox_label[i], output_bbox, dim=self.config.IoU_space,
                                             up_axis=self.config.up_axis)
            ious.append(this_overlap)
            distances.append(this_accuracy)
            results_bbs.append(output_bbox)
        
        # save result
        if self.save_flag:
            pc = input_dict['origin_pc'].view(-1, 3).cpu()
            np.save( + str(self.index) + '_pc.npy', np.array(pc))
            gt_corners = [bbs.corners() for bbs in bbox_label]
            result_corners = [bbs.corners() for bbs in results_bbs]
            np.save(self.save_path + str(self.index) + '_gt_bbox.npy', np.array(gt_corners))
            np.save(self.save_path + str(self.index) + '_output_bbox.npy', np.array(result_corners))
            gt_centers = [bbs.center for bbs in bbox_label]
            np.save(self.save_path + str(self.index) + '_gt_center.npy', np.array(gt_centers))
            self.index = self.index + 1
        
        return ious, distances, results_bbs

    def validation_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        ious, distances, *_ = self.evaluate_one_sequence(sequence)
        # update metrics
        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))
        self.log('success/test', self.success, on_step=True, on_epoch=True)
        self.log('precision/test', self.prec, on_step=True, on_epoch=True)

    def validation_epoch_end(self, outputs):
        self.logger.experiment.add_scalars('metrics/test',
                                           {'success': self.success.compute(),
                                            'precision': self.prec.compute()},
                                           global_step=self.global_step)
    
    def test_step(self, batch, batch_idx):
        sequence = batch[0]  # unwrap the batch with batch size = 1
        ious, distances, result_bbs = self.evaluate_one_sequence(sequence)           
        # update metrics
        self.success(torch.tensor(ious, device=self.device))
        self.prec(torch.tensor(distances, device=self.device))
        self.log('success/test', self.success, on_step=True, on_epoch=True)
        self.log('precision/test', self.prec, on_step=True, on_epoch=True)
        if self.save_flag:
            sequence['track'] = result_bbs
            self.output.append(sequence)
        
        return result_bbs

    def test_epoch_end(self, outputs):
        if self.save_flag:
            save_output_path = '/home/zhangxq/datasets/nuscenes/nuscenes_track_car_test1.pkl'
            with open(save_output_path, 'wb') as f:
                pickle.dump(self.output, f, 0)
        self.logger.experiment.add_scalars('metrics/test',
                                        {'success': self.success.compute(),
                                         'precision': self.prec.compute()},
                                        global_step=self.global_step)
    
    
    
    def build_input_dict(self, info):
        gt_track = info['gt_track']
        track = info['track']
        pc_paths = info['lidar_path']
        frame_num = info['frame_num']
        pc_data = []
        origin_pc_data = []
        centers = []
        track_bbox = []
        track_corners = []
        begin_sot_box = gt_track[0]

        # define ref_center and ref_rot_mat
        for box in track:
            centers.append(box.center)
        centers = np.array(centers)
        
        center_range = np.stack((np.min(centers, axis=0), np.max(centers, axis=0)), axis = 0)
        # ref_center = np.mean(center_range, axis=0)
        # ref_box = Box(center=ref_center, size=begin_sot_box.wlh, orientation=begin_sot_box.orientation)
        ref_box = begin_sot_box
        ref_center = ref_box.center
        # ref_box_theta = ref_box.orientation.radians * ref_box.orientation.axis[-1]
        # ref_bbox = np.append(ref_box.center, ref_box_theta).astype('float32')
        center_range -= ref_center
        range_len = center_range[1, :] - center_range[0, :]
        
        # get all frames point cloud data
        for i, pc_path in enumerate(pc_paths):
            pc = LidarPointCloud.from_file(pc_path)
            # coordinate trans: lidar to ego
            pc.rotate(info['lidar2ego_rotation'][i].rotation_matrix)
            pc.translate(info['lidar2ego_translation'][i])
            # coordinate trans: ego to global
            pc.rotate(info['ego2global_rotation'][i].rotation_matrix)
            pc.translate(info['ego2global_translation'][i])

            pc = PointCloud(points=pc.points)
            crop_pc = points_utils.generate_subwindow(pc, track[i], scale=self.search_scale, offset=self.search_offset)
            # assert crop_pc.nbr_points() > 20, 'not enough search points'
            crop_pc.rotate(track[i].rotation_matrix)
            crop_pc.translate(track[i].center)
            origin_pc = copy.deepcopy(crop_pc).points.T
            crop_pc = points_utils.transform_pc(crop_pc, ref_box)
            points, _ = points_utils.regularize_pc(crop_pc.points.T, self.point_sample_size)
            timestamp = np.full((points.shape[0], 1), fill_value = i)
            pc_data.append(np.concatenate([points, timestamp], axis=-1))
            origin_pc_data.append(origin_pc)

            track[i] = points_utils.transform_box(track[i], ref_box)
            theta = track[i].orientation.radians * track[i].orientation.axis[-1]
            box = np.append(track[i].center, theta)
            box = np.append(box, (i + 1) * 0.01).astype('float32')
            track_bbox.append(box)
            track_corners.append(track[i].corners())
        
        # padding
        pc_data = np.array(pc_data).astype('float32')
        pc_data = np.pad(np.array(pc_data), ((0, self.max_frame_num - frame_num), (0, 0), (0, 0)), 'constant', constant_values = (0, 0))
        origin_pc_data = np.concatenate(origin_pc_data).astype('float32')
        track_bbox = np.array(track_bbox)
        track_bbox = np.pad(track_bbox, ((0, self.max_frame_num - frame_num), (0, 0)), 'constant', constant_values = (0, 0))
        
        input_dict = {
            'origin_pc': torch.tensor(origin_pc_data[None, ...], device=self.device),
            'pc_data': torch.tensor(pc_data[None, ...], device=self.device),
            'track_bbox': torch.tensor(track_bbox[None, ...], device=self.device),
            'ref_box': ref_box,
            'center_range': center_range,
            'range_len': range_len
        }
        return input_dict
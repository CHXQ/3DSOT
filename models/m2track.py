"""
m2track.py
Created by zenn at 2021/11/24 13:10
"""
from datasets import points_utils
from models import base_model
from models.backbone.pointnet import MiniPointNet, SegPointNet

import torch
from torch import nn
import torch.nn.functional as F

from utils.metrics import estimateOverlap, estimateAccuracy
from torchmetrics import Accuracy
from ipdb import set_trace

import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from datasets.data_classes import PointCloud, Box


class M2TRACK(base_model.MotionBaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        
        # add
        self.bb_scale = getattr(config, 'bb_scale', 1.0)
        self.bb_offset = getattr(config, 'bb_offset', 0)
        self.point_sample_size = getattr(config, 'point_sample_size', 1024)
        
        self.seg_acc = Accuracy(num_classes=2, average='none')

        self.box_aware = getattr(config, 'box_aware', False)
        self.use_motion_cls = getattr(config, 'use_motion_cls', True)
        self.use_second_stage = getattr(config, 'use_second_stage', True)
        self.use_prev_refinement = getattr(config, 'use_prev_refinement', True)
        self.seg_pointnet = SegPointNet(input_channel=3 + 1 + 1 + (9 if self.box_aware else 0),
                                        per_point_mlp1=[64, 64, 64, 128, 1024],
                                        per_point_mlp2=[512, 256, 128, 128],
                                        output_size=2 + (9 if self.box_aware else 0))
        self.mini_pointnet = MiniPointNet(input_channel=3 + 1 + (9 if self.box_aware else 0),
                                          per_point_mlp=[64, 128, 256, 512],
                                          hidden_mlp=[512, 256],
                                          output_size=-1)
        if self.use_second_stage:
            self.mini_pointnet2 = MiniPointNet(input_channel=3 + (9 if self.box_aware else 0),
                                               per_point_mlp=[64, 128, 256, 512],
                                               hidden_mlp=[512, 256],
                                               output_size=-1)

            self.box_mlp = nn.Sequential(nn.Linear(256, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 4))
        if self.use_prev_refinement:
            self.final_mlp = nn.Sequential(nn.Linear(256, 128),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Linear(128, 128),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(),
                                           nn.Linear(128, 4))
        if self.use_motion_cls:
            self.motion_state_mlp = nn.Sequential(nn.Linear(256, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 2))
            self.motion_acc = Accuracy(num_classes=2, average='none')

        self.motion_mlp = nn.Sequential(nn.Linear(256, 128),
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
            "points": (B,N,3+1+1)
            "candidate_bc": (B,N,9)

        }

        Returns: B,4

        """
        
        output_dict = {}
        x = input_dict["points"].transpose(1, 2)
        if self.box_aware:
            candidate_bc = input_dict["candidate_bc"].transpose(1, 2)
            x = torch.cat([x, candidate_bc], dim=1)

        B, _, N = x.shape

        seg_out = self.seg_pointnet(x)
        seg_logits = seg_out[:, :2, :]  # B,2,N
        pred_cls = torch.argmax(seg_logits, dim=1, keepdim=True)  # B,1,N
        mask_points = x[:, :4, :] * pred_cls
        mask_xyz_t0 = mask_points[:, :3, :N // 2]  # B,3,N//2
        mask_xyz_t1 = mask_points[:, :3, N // 2:]
        if self.box_aware:
            pred_bc = seg_out[:, 2:, :]
            mask_pred_bc = pred_bc * pred_cls
            # mask_pred_bc_t0 = mask_pred_bc[:, :, :N // 2]  # B,9,N//2
            # mask_pred_bc_t1 = mask_pred_bc[:, :, N // 2:]
            mask_points = torch.cat([mask_points, mask_pred_bc], dim=1)
            output_dict['pred_bc'] = pred_bc.transpose(1, 2)

        point_feature = self.mini_pointnet(mask_points)

        # motion state prediction
        motion_pred = self.motion_mlp(point_feature)  # B,4
        if self.use_motion_cls:
            motion_state_logits = self.motion_state_mlp(point_feature)  # B,2
            motion_mask = torch.argmax(motion_state_logits, dim=1, keepdim=True)  # B,1
            motion_pred_masked = motion_pred * motion_mask
            output_dict['motion_cls'] = motion_state_logits
        else:
            motion_pred_masked = motion_pred
        # previous bbox refinement
        if self.use_prev_refinement:
            prev_boxes = self.final_mlp(point_feature)  # previous bb, B,4
            output_dict["estimation_boxes_prev"] = prev_boxes[:, :4]
        else:
            prev_boxes = torch.zeros_like(motion_pred)

        # 1st stage prediction
        aux_box = points_utils.get_offset_box_tensor(prev_boxes, motion_pred_masked)

        # 2nd stage refinement
        if self.use_second_stage:
            mask_xyz_t0_2_t1 = points_utils.get_offset_points_tensor(mask_xyz_t0.transpose(1, 2),
                                                                     prev_boxes[:, :4],
                                                                     motion_pred_masked).transpose(1, 2)  # B,3,N//2
            mask_xyz_t01 = torch.cat([mask_xyz_t0_2_t1, mask_xyz_t1], dim=-1)  # B,3,N

            # transform to the aux_box coordinate system
            mask_xyz_t01 = points_utils.remove_transform_points_tensor(mask_xyz_t01.transpose(1, 2),
                                                                       aux_box).transpose(1, 2)

            if self.box_aware:
                mask_xyz_t01 = torch.cat([mask_xyz_t01, mask_pred_bc], dim=1)
            output_offset = self.box_mlp(self.mini_pointnet2(mask_xyz_t01))  # B,4
            output = points_utils.get_offset_box_tensor(aux_box, output_offset)
            output_dict["estimation_boxes"] = output
        else:
            output_dict["estimation_boxes"] = aux_box
        output_dict.update({"seg_logits": seg_logits,
                            "motion_pred": motion_pred,
                            'aux_estimation_boxes': aux_box,
                            })

        return output_dict

    def compute_loss(self, data, output):
        loss_total = 0.0
        loss_dict = {}
        aux_estimation_boxes = output['aux_estimation_boxes']  # B,4
        motion_pred = output['motion_pred']  # B,4
        seg_logits = output['seg_logits']
        with torch.no_grad():
            seg_label = data['seg_label']
            box_label = data['box_label']
            box_label_prev = data['box_label_prev']
            motion_label = data['motion_label']
            motion_state_label = data['motion_state_label']
            center_label = box_label[:, :3]
            angle_label = torch.sin(box_label[:, 3])
            center_label_prev = box_label_prev[:, :3]
            angle_label_prev = torch.sin(box_label_prev[:, 3])
            center_label_motion = motion_label[:, :3]
            angle_label_motion = torch.sin(motion_label[:, 3])

        loss_seg = F.cross_entropy(seg_logits, seg_label, weight=torch.tensor([0.5, 2.0]).cuda())
        if self.use_motion_cls:
            motion_cls = output['motion_cls']  # B,2
            loss_motion_cls = F.cross_entropy(motion_cls, motion_state_label)
            loss_total += loss_motion_cls * self.config.motion_cls_seg_weight
            loss_dict['loss_motion_cls'] = loss_motion_cls

            loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion, reduction='none')
            loss_center_motion = (motion_state_label * loss_center_motion.mean(dim=1)).sum() / (
                    motion_state_label.sum() + 1e-6)
            loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion, reduction='none')
            loss_angle_motion = (motion_state_label * loss_angle_motion).sum() / (motion_state_label.sum() + 1e-6)
        else:
            loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion)
            loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion)

        if self.use_second_stage:
            estimation_boxes = output['estimation_boxes']  # B,4
            loss_center = F.smooth_l1_loss(estimation_boxes[:, :3], center_label)
            loss_angle = F.smooth_l1_loss(torch.sin(estimation_boxes[:, 3]), angle_label)
            loss_total += 1 * (loss_center * self.config.center_weight + loss_angle * self.config.angle_weight)
            loss_dict["loss_center"] = loss_center
            loss_dict["loss_angle"] = loss_angle
        if self.use_prev_refinement:
            estimation_boxes_prev = output['estimation_boxes_prev']  # B,4
            loss_center_prev = F.smooth_l1_loss(estimation_boxes_prev[:, :3], center_label_prev)
            loss_angle_prev = F.smooth_l1_loss(torch.sin(estimation_boxes_prev[:, 3]), angle_label_prev)
            loss_total += (loss_center_prev * self.config.center_weight + loss_angle_prev * self.config.angle_weight)
            loss_dict["loss_center_prev"] = loss_center_prev
            loss_dict["loss_angle_prev"] = loss_angle_prev

        loss_center_aux = F.smooth_l1_loss(aux_estimation_boxes[:, :3], center_label)

        loss_angle_aux = F.smooth_l1_loss(torch.sin(aux_estimation_boxes[:, 3]), angle_label)

        loss_total += loss_seg * self.config.seg_weight \
                      + 1 * (loss_center_aux * self.config.center_weight + loss_angle_aux * self.config.angle_weight) \
                      + 1 * (
                              loss_center_motion * self.config.center_weight + loss_angle_motion * self.config.angle_weight)
        loss_dict.update({
            "loss_total": loss_total,
            "loss_seg": loss_seg,
            "loss_center_aux": loss_center_aux,
            "loss_center_motion": loss_center_motion,
            "loss_angle_aux": loss_angle_aux,
            "loss_angle_motion": loss_angle_motion,
        })
        if self.box_aware:
            prev_bc = data['prev_bc']
            this_bc = data['this_bc']
            bc_label = torch.cat([prev_bc, this_bc], dim=1)
            pred_bc = output['pred_bc']
            loss_bc = F.smooth_l1_loss(pred_bc, bc_label)
            loss_total += loss_bc * self.config.bc_weight
            loss_dict.update({
                "loss_total": loss_total,
                "loss_bc": loss_bc
            })

        return loss_dict

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch: {
            "points": stack_frames, (B,N,3+9+1)
            "seg_label": stack_label,
            "box_label": np.append(this_gt_bb_transform.center, theta),
            "box_size": this_gt_bb_transform.wlh
        }
        Returns:

        """
        output = self(batch)
        loss_dict = self.compute_loss(batch, output)
        loss = loss_dict['loss_total']

        # log
        seg_acc = self.seg_acc(torch.argmax(output['seg_logits'], dim=1, keepdim=False), batch['seg_label'])
        self.log('seg_acc_background/train', seg_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('seg_acc_foreground/train', seg_acc[1], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if self.use_motion_cls:
            motion_acc = self.motion_acc(torch.argmax(output['motion_cls'], dim=1, keepdim=False),
                                         batch['motion_state_label'])
            self.log('motion_acc_static/train', motion_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log('motion_acc_dynamic/train', motion_acc[1], on_step=True, on_epoch=True, prog_bar=False,
                     logger=True)

        log_dict = {k: v.item() for k, v in loss_dict.items()}

        self.logger.experiment.add_scalars('loss', log_dict,
                                           global_step=self.global_step)
        return loss



    # def evaluate_one_sequence(self, sequence):
    #     """
    #     :param sequence: a sequence of annos {"pc": pc, "3d_bbox": bb, 'meta': anno}
    #     :return:
    #     """
    #     ious = []
    #     distances = []
    #     results_bbs = []
        
    #     bbox_label = sequence['gt_track']
        
    #     # bbox_corners = [bbs.corners() for bbs in sequence['track']]
    #     # np.save('bus_bbox0.npy', np.array(bbox_corners))
        
    #     input_dict = self.build_input_dict(sequence)
    #     ref_box = input_dict['ref_box']
    #     # output = self(input_dict)
    #     # estimation_box = output['estimation_boxes']
    #     # estimation_box_cpu = estimation_box.cpu()
    #     # estimation_box_cpu[:, 3] = input_dict['track_bbox'][0][1, 3]
    #     estimation_box_cpu = input_dict['track_bbox'][0].cpu()
        
    #     for i in range(len(bbox_label)):
            
    #         if (i == 0):
    #             output_bbox = bbox_label[i]
    #         else:
    #             output_bbox = points_utils.getOffsetBB(ref_box, estimation_box_cpu[i-1], degrees=self.config.degrees,
    #                                              use_z=self.config.use_z,
    #                                              limit_box=self.config.limit_box)
            
    #         this_overlap = estimateOverlap(bbox_label[i], output_bbox, dim=self.config.IoU_space,
    #                                        up_axis=self.config.up_axis)
    #         this_accuracy = estimateAccuracy(bbox_label[i], output_bbox, dim=self.config.IoU_space,
    #                                          up_axis=self.config.up_axis)
    #         ious.append(this_overlap)
    #         distances.append(this_accuracy)
    #         results_bbs.append(output_bbox)
        
    #     # save result
    #     # gt_corners = [bbs.corners() for bbs in bbox_label]
    #     # result_corners = [bbs.corners() for bbs in results_bbs]
    #     # np.save('bus_gt_3d_bbox0.npy', np.array(gt_corners))
    #     # np.save('result_bbs0.npy', np.array(result_corners))
        
    #     return ious, distances, results_bbs

    # def validation_step(self, batch, batch_idx):
    #     sequence = batch[0]  # unwrap the batch with batch size = 1
    #     ious, distances, *_ = self.evaluate_one_sequence(sequence)
    #     # update metrics
    #     self.success(torch.tensor(ious, device=self.device))
    #     self.prec(torch.tensor(distances, device=self.device))
    #     self.log('success/test', self.success, on_step=True, on_epoch=True)
    #     self.log('precision/test', self.prec, on_step=True, on_epoch=True)

    # def validation_epoch_end(self, outputs):
    #     self.logger.experiment.add_scalars('metrics/test',
    #                                        {'success': self.success.compute(),
    #                                         'precision': self.prec.compute()},
    #                                        global_step=self.global_step)

    # def build_input_dict(self, info):
    #     gt_track = info['gt_track']
    #     track = info['track']
    #     pc_paths = info['lidar_path']
    #     frame_num = info['frame_num']
    #     pc_data = []
    #     centers = []
    #     track_bbox = []
    #     track_corners = []
    #     begin_sot_box = gt_track[0]

    #     # define ref_center and ref_rot_mat
    #     for box in track:
    #         centers.append(box.center)
    #     centers = np.array(centers)
        
    #     center_range = np.stack((np.min(centers, axis=0), np.max(centers, axis=0)), axis = 0)
    #     # ref_center = np.mean(center_range, axis=0)
    #     # ref_box = Box(center=ref_center, size=begin_sot_box.wlh, orientation=begin_sot_box.orientation)
    #     ref_box = begin_sot_box
    #     ref_center = ref_box.center
    #     # ref_box_theta = ref_box.orientation.radians * ref_box.orientation.axis[-1]
    #     # ref_bbox = np.append(ref_box.center, ref_box_theta).astype('float32')
    #     center_range -= ref_center
    #     range_len = center_range[1, :] - center_range[0, :]
        
    #     # get all frames point cloud data
    #     for i, pc_path in enumerate(pc_paths):
    #         pc = LidarPointCloud.from_file(pc_path)
    #         # coordinate trans: lidar to ego
    #         pc.rotate(info['lidar2ego_rotation'][i].rotation_matrix)
    #         pc.translate(info['lidar2ego_translation'][i])
    #         # coordinate trans: ego to global
    #         pc.rotate(info['ego2global_rotation'][i].rotation_matrix)
    #         pc.translate(info['ego2global_translation'][i])

    #         pc = PointCloud(points=pc.points)
    #         crop_pc = points_utils.generate_subwindow(pc, track[i], scale=self.bb_scale, offset=self.bb_offset)
    #         # assert crop_pc.nbr_points() > 20, 'not enough search points'
    #         crop_pc.rotate(track[i].rotation_matrix)
    #         crop_pc.translate(track[i].center)
    #         crop_pc = points_utils.transform_pc(crop_pc, ref_box)
    #         points, _ = points_utils.regularize_pc(crop_pc.points.T, self.point_sample_size)
    #         timestamp = np.full((points.shape[0], 1), fill_value = i)
    #         pc_data.append(np.concatenate([points, timestamp], axis=-1))

    #         track[i] = points_utils.transform_box(track[i], ref_box)
    #         theta = track[i].orientation.radians * track[i].orientation.axis[-1]
    #         # box = np.append((track[i].center - center_range[0]) / range_len, theta).astype('float32')
    #         box = np.append(track[i].center, theta).astype('float32')
    #         track_bbox.append(box)
    #         track_corners.append(track[i].corners())
        
    #     # padding
    #     # pc_data = np.concatenate(pc_data).astype('float32')
    #     # pc_data = np.pad(pc_data, ((0, self.max_point_num - len(pc_data)), (0, 0)), 'edge')
    #     track_bbox = np.array(track_bbox)
    #     # track_bbox = np.pad(track_bbox, ((0, self.max_frame_num - frame_num), (0, 0)), 'constant', constant_values = (0, 0))
        
    #     # begin_sot_box = points_utils.transform_box(begin_sot_box, ref_box)
    #     # begin_sot_box_theta = begin_sot_box.orientation.radians * begin_sot_box.orientation.axis[-1]
    #     # begin_sot_bbox = np.append(begin_sot_box.center, begin_sot_box_theta).astype('float32')
        
    #     # pc_data, _ = points_utils.regularize_pc(pc_data, self.point_sample_size)
    #     input_dict = {
    #         # 'pc_data': torch.tensor(pc_data[None, ...], device=self.device),
    #         'track_bbox': torch.tensor(track_bbox[None, ...], device=self.device),
    #         'ref_box': ref_box,
    #         # 'center_range': center_range,
    #         # 'range_len': range_len
    #     }
    #     return input_dict
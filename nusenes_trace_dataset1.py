import numpy as np
import torch
import argparse
import os
import pickle
from torch.utils.data import Dataset
from nuscenes.utils.data_classes import LidarPointCloud
from datasets.data_classes import PointCloud, Box
from datasets import points_utils
from torch.utils.data import DataLoader
from tqdm import tqdm
from nuscenes.utils import geometry_utils
import torch.nn.functional as F


class NuscenceTraceDataset(Dataset):
    '''
        Nuscenes Trace Dataset for SOT
    '''

    def __init__(self, data_root, ann_file, search_scale = 1.25,
                 search_offset = 2, max_frame_num = 41, max_point_num = 41984, point_sample_size = 1024):
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.search_scale = search_scale
        self.search_offset = search_offset
        self.max_frame_num = max_frame_num
        self.max_point_num = max_point_num
        self.point_sample_size = point_sample_size

        # load annotations
        self.data_infos = self.load_annotations(self.ann_file)

    
    def load_annotations(self, ann_file):
        with open(ann_file, 'rb') as f:
            data_infos = pickle.load(f)
        return data_infos

    def get_data_info(self, index):
        info = self.data_infos[index]
        gt_track = info['gt_track']
        track = info['track']
        pc_paths = info['lidar_path']
        frame_num = info['frame_num']
        pc_data = []
        begin_sot_box = gt_track[0]
        centers = []
        track_bbox = []
        track_gt_bbox = []
        track_corners = []
        track_gt_corners = []
        seg_label = []
        
        bbox_mask = np.zeros(self.max_frame_num).astype('float32')

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
        
        flag = False
        for i, pc_path in enumerate(pc_paths):
            if flag:
                break
            dis = np.linalg.norm(gt_track[i].center - track[i].center)
            if dis > 1:
                frame_num = i + 1
                flag = True
            
            bbox_mask[i] = 1
            pc = LidarPointCloud.from_file(pc_path)
            # coordinate trans: lidar to ego
            pc.rotate(info['lidar2ego_rotation'][i].rotation_matrix)
            pc.translate(info['lidar2ego_translation'][i])
            # coordinate trans: ego to global
            pc.rotate(info['ego2global_rotation'][i].rotation_matrix)
            pc.translate(info['ego2global_translation'][i])

            pc = PointCloud(points=pc.points)
            crop_pc = points_utils.generate_subwindow(pc, track[i], scale=self.search_scale, offset=self.search_offset)
            crop_pc.rotate(track[i].rotation_matrix)
            crop_pc.translate(track[i].center)
            crop_pc = points_utils.transform_pc(crop_pc, ref_box)
            points, _ = points_utils.regularize_pc(crop_pc.points.T, self.point_sample_size)
            timestamp = np.full((points.shape[0], 1), fill_value = (i + 1) * 0.01)
            pc_data.append(np.concatenate([points, timestamp], axis=-1))

            gt_track[i] = points_utils.transform_box(gt_track[i], ref_box)
            
            seg_label_this = geometry_utils.points_in_box(gt_track[i], points.T, 1.25).astype(int)
            seg_label.append(seg_label_this)
            theta_gt = gt_track[i].orientation.radians * gt_track[i].orientation.axis[-1]
            gt_box = np.append(gt_track[i].center, theta_gt).astype('float32')
            track_gt_bbox.append(gt_box)
            track[i] = points_utils.transform_box(track[i], ref_box)
            theta = track[i].orientation.radians * track[i].orientation.axis[-1]
            box = np.append(track[i].center, theta)
            box = np.append(box, (i + 1) * 0.01).astype('float32')
            track_bbox.append(box)
            track_corners.append(track[i].corners())
            track_gt_corners.append(gt_track[i].corners())
        
        stack_seg_label = np.concatenate(seg_label)
        pc_data = np.concatenate(pc_data).astype('float32')
        track_gt_bbox = np.array(track_gt_bbox)
        track_bbox = np.array(track_bbox)
        
        # padding
        pc_data = np.pad(pc_data, ((0, self.max_point_num - len(pc_data)), (0, 0)), 'constant', constant_values = (0, 0))
        track_gt_bbox = np.pad(track_gt_bbox, ((0, self.max_frame_num - frame_num), (0, 0)), 'constant', constant_values = (0, 0))
        track_bbox = np.pad(track_bbox, ((0, self.max_frame_num - frame_num), (0, 0)), 'constant', constant_values = (0, 0))
        stack_label = np.zeros(self.max_point_num)
        stack_label[:frame_num*self.point_sample_size] = stack_seg_label
                
        input_dict = {
            'pc_data': pc_data,
            'gt_track_bbox': track_gt_bbox,
            'track_bbox': track_bbox,
            'bbox_size': begin_sot_box.wlh,
            'range_len': range_len,
            'center_range': center_range,
            'bbox_mask': bbox_mask,
            'frame_num': frame_num,
            'seg_label': stack_label.astype('int'),
            # 'track_corners': np.array(track_corners),
            # 'track_gt_corners': np.array(track_gt_corners),
        }
        return input_dict
    
    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, index):
        input_dict = self.get_data_info(index)
        return input_dict


class TestTraceDataset(Dataset):
    def __init__(self, data_root, ann_file, search_scale = 1.25, search_offset = 2, max_frame_num = 64, max_point_num = 100000):
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.search_scale = search_scale
        self.search_offset = search_offset
        self.max_frame_num = max_frame_num
        self.max_point_num = max_point_num

        # load annotations
        self.data_infos = self.load_annotations(self.ann_file)

    
    def load_annotations(self, ann_file):
        with open(ann_file, 'rb') as f:
            data_infos = pickle.load(f)
        return data_infos

    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, index):
        info = self.data_infos[index]
        return info


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
    args = parser.parse_args()

    return args
    
if __name__ == '__main__':
    cfg = parse_config()
    data_root = '/home/zhangxq/datasets/nuscenes'
    ann_file = os.path.join(data_root, 'nuscenes_track_car_val_extract.pkl')
    data = NuscenceTraceDataset(data_root, ann_file)
    info = data.__getitem__(0)
    data_loader = DataLoader(data, batch_size=1, num_workers=32)
    ranges = []
    # model = MyModel(cfg)
    num = 0
    loss = []
    for batch_data in tqdm(data_loader):
        seg_label = batch_data['seg_label'][0]

    print(len(loss))
    print('finish')

# train range_len = 0  514
# val   range_len = 0  172
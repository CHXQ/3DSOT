import os
import pickle
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F

def extract_data(num, root_path, min_point_num, min_distance):
    indexs = []
    for i in range(num):
        gt_bbox_path = os.path.join(root_path, str(i) + '_gt_bbox.npy')
        pc_path = os.path.join(root_path, str(i) + '_pc.npy')
        gt_center_path = os.path.join(root_path, str(i) + '_gt_center.npy')
        gt_bbox = np.load(gt_bbox_path)
        pc = np.load(pc_path)
        gt_center = np.load(gt_center_path)
        dis = np.linalg.norm(gt_center[-1] - gt_center[0], ord = 1)
        frame = len(gt_bbox)
        if dis >= min_distance and frame >= 10 and pc.shape[0] / frame >= min_point_num:
            indexs.append(i)
    return indexs

def create_nuscenes_track_infos(root_path, track_path, version='v1.0-trainval'):
    """Create tracking info file of nuscene dataset"""

    max_loss = 0
    nusc = NuScenes(version=version, dataroot=root_path)
    nusc_infos = []
    loss = []
    with open(track_path, 'rb') as f:
        data = pickle.load(f)
    for sequence in tqdm(data):
        lidar_path = []
        lidar2ego_translation = []
        lidar2ego_rotation = []
        ego2global_translation = []
        ego2global_rotation = []
        lidar_data = sequence['data_lidar']
        for sample_data_lidar in lidar_data:
            pcl_path = os.path.join(root_path, sample_data_lidar['filename'])
            lidar_path.append(pcl_path)
            cs_record = nusc.get('calibrated_sensor', sample_data_lidar['calibrated_sensor_token'])
            lidar2ego_translation.append(np.array(cs_record['translation']))
            lidar2ego_rotation.append(Quaternion(cs_record['rotation']))
            pose_record = nusc.get('ego_pose', sample_data_lidar['ego_pose_token'])
            ego2global_translation.append(np.array(pose_record['translation']))
            ego2global_rotation.append(Quaternion(pose_record['rotation']))
        gt_track = sequence['gt']
        tracks = sequence['tracks']
        gt_track_center = [bbox.center for bbox in gt_track]
        # gt_track_center = np.array(gt_track_center)
        track_center = [bbox.center for bbox in tracks]
        track_center = np.array(track_center)
        # loss = F.smooth_l1_loss(torch.FloatTensor(gt_track_center), torch.FloatTensor(track_center))
        # if (loss > max_loss):
        #     max_loss = loss
        # if (loss > 1):
        #     continue
        
        info = {
            'lidar_path': lidar_path,
            'lidar2ego_translation': lidar2ego_translation,
            'lidar2ego_rotation': lidar2ego_rotation,
            'ego2global_translation': ego2global_translation,
            'ego2global_rotation': ego2global_rotation,
            'gt_track': gt_track,
            'track': tracks,
            'frame_num': len(gt_track)
        }
        nusc_infos.append(info)
    return nusc_infos, max_loss

if __name__ == '__main__':
    root_path = '/home/zhangxq/datasets/nuscenes'
    train_track_path = 'outputs/nuscenes_car_track_train.dat'
    train_infos, max_loss = create_nuscenes_track_infos(root_path, train_track_path)
    print(len(train_infos))
    num = len(train_infos)
    indexs = extract_data(num, 'train_save_bbox', 10, 5)
    # print(max_loss)
    data = []
    for i in indexs:
        data.append(train_infos[i])
    print(len(data))
    train_info_path = os.path.join(root_path, 'nuscenes_track_car_train_extract.pkl')
    with open(train_info_path, 'wb') as f:
        pickle.dump(data, f, 0)
    # val_track_path = 'outputs/nuscenes_car_track_val.dat'
    # val_infos = create_nuscenes_track_infos(root_path, val_track_path)
    # val_info_path = os.path.join(root_path, 'nuscenes_track_car_val.pkl')
    # with open(val_info_path, 'wb') as f:
    #     pickle.dump(val_infos, f, 0)
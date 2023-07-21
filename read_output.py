import pickle
import os
from tqdm import tqdm

# save_path = 'outputs/nuscenes_car_track_train.dat'

# data_path = "outputs/train_track_Car.dat"


# with open(data_path, 'rb') as f:
#     data = pickle.load(f)

# train
# with open(save_path, 'wb') as f:
#     save_data = []
#     for sequence in tqdm(data):
#         data_lidar = []
#         for s in sequence['meta']:
#             data_lidar.append(s['sample_data_lidar'])
#         for track in sequence['bbox_predict_list']:
#             seq_data = {'data_lidar': data_lidar, 'gt': sequence['bbox_gt'], 'tracks':track}
#             save_data.append(seq_data)
#     pickle.dump(save_data, f, 0)
# print('write finish')

# val
# num = 0
# with open(save_path, 'wb') as f:
#     save_data = []
#     for sequence in tqdm(data):
#         data_lidar = []
#         gt = []
#         if (len(sequence['bbox_gt']) == 1):
#             num += 1
#         for s in sequence['meta']:
#             data_lidar.append(s['sample_data_lidar'])
#         seq_data = {'data_lidar': data_lidar, 'gt': sequence['bbox_gt'], 'tracks':sequence['bbox_predict']}
#         save_data.append(seq_data)
#     pickle.dump(save_data, f, 0)
# print(num)
# print('write finish')

# path = '/home/zhangxq/datasets/nuscenes/nuscenes_track_car_infos_train.pkl'
# with open(path, 'rb') as f:
#     data = pickle.load(f)
#     print(len(data))
# from nuscenes.utils import geometry_utils

# data_path = '/home/zhangxq/datasets/nuscenes/preload_nuscenes_Car_train_track_v1.0-trainval_10_-1.dat'
# with open(data_path, 'rb') as f:
#     training_samples = pickle.load(f)
# num = 0
# for track in tqdm(training_samples):
#     pc = track[0]['pc']
#     bbox = track[0]['3d_bbox']
#     num_points_in_prev_box = geometry_utils.points_in_box(bbox, pc.points).sum()
#     if num_points_in_prev_box <= 10:
#         num += 1

# print(num)
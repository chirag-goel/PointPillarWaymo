import abc
from typing import List

import os
import tensorflow as tf
import math
import itertools
import numpy as np
tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import dataset_pb2

class Label3D:
    def __init__(self, classification: str, centroid: np.ndarray, dimension: np.ndarray, yaw: float):
        self.classification = classification
        self.centroid = centroid
        self.dimension = dimension
        self.yaw = yaw


class DataReader:

    @staticmethod
    @abc.abstractmethod
    def read_lidar(file_path: str) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def read_label(file_path: str) -> List[Label3D]:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def read_calibration(file_path: str) -> np.ndarray:
        raise NotImplementedError


class KittiDataReader(DataReader):

    def __init__(self):
        super(KittiDataReader, self).__init__()

    @staticmethod
    def read_lidar(file_path: str):
        return np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))

    @staticmethod
    def read_label(file_path: str):
        with open(file_path, "r") as f:

            lines = f.readlines()

            elements = []
            for line in lines:

                values = line.split()

                element = Label3D(
                    str(values[0]),
                    np.array(values[11:14], dtype=np.float32),
                    np.array(values[8:11], dtype=np.float32),
                    float(values[14])
                )

                if element.classification == "DontCare":
                    continue
                else:
                    elements.append(element)

        return elements

    @staticmethod
    def read_calibration(file_path: str):
        with open(file_path, "r") as f:
            lines = f.readlines()
            Tr_velo_to_cam = np.array(lines[5].split(": ")[1].split(" "), dtype=np.float32).reshape((3, 4))
            R, t = Tr_velo_to_cam[:, :3], Tr_velo_to_cam[:, 3]
            return R, t




class WaymoDataReader(DataReader):

    def __init__(self):
        super(WaymoDataReader, self).__init__()

    @staticmethod
    def read_lidar(file_path: str, str_ind: int):
        print(file_path)
        def convert_range_image_to_point_cloud_with_intensity(frame,range_images,camera_projections,range_image_top_pose,ri_index=0):
            calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
            points = []
            cp_points = []
            intensity = []
            frame_pose = tf.convert_to_tensor(value=np.reshape(np.array(frame.pose.transform), [4, 4]))
            # [H, W, 6]
            range_image_top_pose_tensor = tf.reshape(tf.convert_to_tensor(value=range_image_top_pose.data),range_image_top_pose.shape.dims)
            # [H, W, 3, 3]
            range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],range_image_top_pose_tensor[..., 2])
            range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
            range_image_top_pose_tensor = transform_utils.get_transform(range_image_top_pose_tensor_rotation,range_image_top_pose_tensor_translation)
            for c in calibrations:
                range_image = range_images[c.name][ri_index]
                if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                    beam_inclinations = range_image_utils.compute_inclination(tf.constant([c.beam_inclination_min, c.beam_inclination_max]),height=range_image.shape.dims[0])
                else:
                    beam_inclinations = tf.constant(c.beam_inclinations)
                beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
                extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])
                range_image_tensor = tf.reshape(tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
                pixel_pose_local = None
                frame_pose_local = None
                if c.name == dataset_pb2.LaserName.TOP:
                    pixel_pose_local = range_image_top_pose_tensor
                    pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                    frame_pose_local = tf.expand_dims(frame_pose, axis=0)
                range_image_mask = range_image_tensor[..., 0] > 0
                range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(tf.expand_dims(range_image_tensor[..., 0], axis=0),tf.expand_dims(extrinsic, axis=0),tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),pixel_pose=pixel_pose_local,frame_pose=frame_pose_local)
                range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
                # print(range_image_cartesian.shape)
                points_tensor = tf.gather_nd(range_image_cartesian,tf.compat.v1.where(range_image_mask))
                # intensity_tensor = points_tensor
                intensity_tensor = tf.gather_nd(range_image_tensor[..., 1],tf.compat.v1.where(range_image_mask))

                cp = camera_projections[c.name][0]
                cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
                cp_points_tensor = tf.gather_nd(cp_tensor,tf.compat.v1.where(range_image_mask))
                points.append(points_tensor.numpy())
                intensity.append(intensity_tensor.numpy())
                cp_points.append(cp_points_tensor.numpy())
            return points, cp_points, intensity
        dataset = tf.data.TFRecordDataset(file_path, compression_type='')
        lidar_points = []
        frameCount = -1
        for data in dataset:
            frameCount+=1
            if((str_ind != frameCount)):
              continue;
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            (range_images, camera_projections,range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
            points, cp_points, intensity = convert_range_image_to_point_cloud_with_intensity(frame,range_images,camera_projections,range_image_top_pose)
            lidar_points.append(np.concatenate((points[0],np.expand_dims(intensity[0],axis=1)),axis=1))
            #print("Got lidar data for frame " + str(flag))
                

        return lidar_points


    @staticmethod
    def read_label(file_path: str, str_ind: int):
        dataset = tf.data.TFRecordDataset(file_path, compression_type='')
        frameCount = -1
        labelsList = []
        for data in dataset:
            frameCount+=1
            if(str_ind != frameCount):
              continue
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            labelConversions = {
                1: "Car",
                2: "Pedestrian",
                3: "Misc",
                4: "Cyclist"
            }

            labels = []
            for frameNum in frame.laser_labels:
                element = Label3D(
                    str(labelConversions[frameNum.type]),
                    np.array([frameNum.box.center_x,frameNum.box.center_y,frameNum.box.center_z], dtype=np.float32),
                    np.array([frameNum.box.length,frameNum.box.width,frameNum.box.height], dtype=np.float32),
                    float(frameNum.box.heading)
                )
                labelsList.append(element)
        return labelsList

    @staticmethod
    def read_calibration(file_path: str):
        with open(file_path, "r") as f:
            lines = f.readlines()
            Tr_velo_to_cam = np.array(lines[5].split(": ")[1].split(" "), dtype=np.float32).reshape((3, 4))
            R, t = Tr_velo_to_cam[:, :3], Tr_velo_to_cam[:, 3]
            return R, t

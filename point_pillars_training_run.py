import os
import tensorflow as tf
from glob import glob

from config import Parameters
from loss import PointPillarNetworkLoss
from network import build_point_pillar_graph
from processors import SimpleDataGenerator,WaymoDataGenerator
from readers import KittiDataReader,WaymoDataReader

import os
import tensorflow as tf
import math
import numpy as np
import itertools

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import dataset_pb2



DATA_ROOT = "/home/chirag1goel_gmail_com/PointPillars/data/training_data/"  # TODO make main arg



if __name__ == "__main__":

    params = Parameters()

    pillar_net = build_point_pillar_graph(params)

    loss = PointPillarNetworkLoss(params)

    optimizer = tf.keras.optimizers.Adam(lr=params.learning_rate, decay=params.decay_rate)

    pillar_net.compile(optimizer, loss=loss.losses(), metrics=['accuracy'])

    log_dir = "./logs"
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(log_dir, "model.h5"), save_best_only=True),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * 0.8 if ((epoch % 15 == 0) and (epoch != 0)) else lr, verbose=True),
        tf.keras.callbacks.EarlyStopping(patience=20),
    ]

    data_reader = KittiDataReader()

    lidar_files = sorted(glob(os.path.join(DATA_ROOT, "velodyne", "*.bin")))
    label_files = sorted(glob(os.path.join(DATA_ROOT, "label", "*.txt")))
    calibration_files = sorted(glob(os.path.join(DATA_ROOT, "calib", "*.txt")))
    
    lidar_files = lidar_files[:500]
    label_files = label_files[:500]
    calibration_files = calibration_files[:500]

    sepVal = (int)(len(lidar_files)/0.8)

    lidar_files_val = lidar_files[sepVal:]
    label_files_val = label_files[sepVal:]
    calibration_files_val = calibration_files[sepVal:]


    lidar_files = lidar_files[:sepVal]
    label_files = label_files[:sepVal]
    calibration_files = calibration_files[:sepVal]


    #training_gen = SimpleDataGenerator(data_reader, params.batch_size, lidar_files, label_files, calibration_files)

    #validation_gen = SimpleDataGenerator(data_reader, params.batch_size, lidar_files_val, label_files_val, calibration_files_val)

    waymo_data_reader = WaymoDataReader()


    waymo_training_files = sorted(glob(os.path.join("/mnt/disks/waymo/", "training", "*.tfrecord")))

    waymo_validation_files = sorted(glob(os.path.join("/mnt/disks/waymo/", "validation", "*.tfrecord")))


    waymo_data_reader = WaymoDataReader()


    training_gen = WaymoDataGenerator(waymo_data_reader, params.batch_size, waymo_training_files)

    validation_gen = WaymoDataGenerator(waymo_data_reader, params.batch_size, waymo_validation_files)


    try:
        pillar_net.fit_generator(training_gen,
                                 len(training_gen),
                                 callbacks=callbacks,
                                 use_multiprocessing=False,
                                 epochs=int(params.total_training_epochs),
                                 workers=6,
                                 validation_data = validation_gen,
                                 validation_steps = len(validation_gen))

    except KeyboardInterrupt:
        pillar_net.save(os.path.join(log_dir, "interrupted.h5"))
        session = tf.keras.backend.get_session()
        session.close()

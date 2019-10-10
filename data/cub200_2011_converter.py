# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 20:11:40 2016

@author: sakurai
"""

import os
import tarfile
import subprocess
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import h5py
import fuel
from fuel.datasets.hdf5 import H5PYDataset
from tqdm import tqdm
import cv2


def extract_class_label(filename):
    """
    arg:
        filename: string, e.g.
        'images/001.Black_footed_Albatross/Black_footed_Albatross_0001_2950163169.jpg'

    return:
        A class label as integer, e.g. 1
    """
    _, class_dir, _ = filename.split("/")
    return int(class_dir.split(".")[0])


def preprocess(hwc_bgr_image, size):
    hwc_rgb_image = cv2.cvtColor(hwc_bgr_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(hwc_rgb_image, (size))
    chw_image = np.transpose(resized, axes=(2, 0, 1))
    return chw_image


if __name__ == '__main__':
    dataset_name = "cub200_2011"
    archive_basename = "CUB_200_2011"

    fuel_data_path = dataset_name
    extracted_dir_path = os.path.join(fuel_data_path, archive_basename)
    archive_filepath = extracted_dir_path + ".tgz"
    images_dir_path = os.path.join(extracted_dir_path, "images")
    label_filepath = os.path.join(extracted_dir_path, "image_class_labels.txt")
    image_list_filepath = os.path.join(extracted_dir_path, "images.txt")

    # Extract CUB_200_2011.tgz if CUB_200_2011 directory does not exist
    if not os.path.exists(os.path.join(fuel_data_path, archive_basename)):
        subprocess.call(["tar", "zxvf", archive_filepath.replace("\\", "/"),
                         "-C", fuel_data_path.replace("\\", "/"),
                         "--force-local"])

    id_name_pairs = np.loadtxt(image_list_filepath, np.str)
    assert np.array_equal(
        [int(i) for i in id_name_pairs[:, 0].tolist()], range(1, 11789))
    id_label_pairs = np.loadtxt(label_filepath, np.str)
    assert np.array_equal(
        [int(i) for i in id_label_pairs[:, 0].tolist()], range(1, 11789))
    jpg_filenames = id_name_pairs[:, 1].tolist()
    class_labels = [int(i) for i in id_label_pairs[:, 1].tolist()]
    num_examples = len(jpg_filenames)
    num_clases = 200
    assert np.array_equal(np.unique(class_labels), range(1, num_clases + 1))

    # open hdf5 file
    hdf5_filename = dataset_name + ".hdf5"
    hdf5_filepath = os.path.join(fuel_data_path, hdf5_filename)
    hdf5 = h5py.File(hdf5_filepath, mode="w")

    # store images
    image_size = (256, 256)
    array_shape = (num_examples, 3) + image_size
    ds_images = hdf5.create_dataset("images", array_shape, dtype=np.uint8)
    ds_images.dims[0].label = "batch"
    ds_images.dims[1].label = "channel"
    ds_images.dims[2].label = "height"
    ds_images.dims[3].label = "width"

    # write images to the disk
    for i, filename in tqdm(enumerate(jpg_filenames), total=num_examples,
                            desc=hdf5_filepath):
        raw_image = cv2.imread(os.path.join(images_dir_path, filename),
                               cv2.IMREAD_COLOR)  # BGR image
        image = preprocess(raw_image, image_size)
        ds_images[i] = image

    # store the targets (class labels)
    targets = np.array(class_labels, np.int32).reshape(num_examples, 1)
    ds_targets = hdf5.create_dataset("targets", data=targets)
    ds_targets.dims[0].label = "batch"
    ds_targets.dims[1].label = "class_labels"

    # specify the splits (labels 1~100 for train, 101~200 for test)
    test_head = class_labels.index(101)
    split_train, split_test = (0, test_head), (test_head, num_examples)
    split_dict = dict(train=dict(images=split_train, targets=split_train),
                      test=dict(images=split_test, targets=split_test))
    hdf5.attrs["split"] = H5PYDataset.create_split_array(split_dict)

    hdf5.flush()
    hdf5.close()

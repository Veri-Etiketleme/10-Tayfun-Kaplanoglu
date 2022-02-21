import argparse
import json
import os
import sys
import glob
import numpy as np


def get_fps(video_dir):

    timestamp_file = os.path.join(video_dir, "timestamp.txt")
    if not os.path.exists(timestamp_file):
        return 1.0

    with open(timestamp_file) as f:
        lines = f.read().splitlines()

    times = [float(line) for line in lines]
    return round(1 / np.mean(np.diff(times)), 1)


def convert_mlgesture_to_dict(video_root, split_path, subset, modality):

    database = {}
    class_names = []
    with open(split_path, 'r') as f:
        data = json.load(f)

    for entry in data:
        # {'video': 'label01/subject1/video0/mlx90640', 'label': ['turn_right']}
        video_name = os.path.join(entry['video'], modality)
        video_dir = os.path.join(video_root, video_name)
        fps = get_fps(video_dir)
        n_frames = len(glob.glob(os.path.join(video_dir, '*.tiff')))
        assert n_frames > 0, "Found video with zero frames: {}".format(video_dir)

        # gather all class names to later construct a list of unique class names
        class_label = entry['label']
        if isinstance(class_label, list):
            class_names.extend(class_label)
        else:
            class_names.append(class_label)

        database[video_name] = {}
        database[video_name]['subset'] = subset
        database[video_name]['annotations'] = {'label': class_label}
        database[video_name]['n_frames'] = n_frames
        database[video_name]['fps'] = fps

    class_labels = list(set(class_names))

    # make sure no_gesture is in the list of unique labels
    if "no_gesture" not in class_labels:
        class_labels = ["no_gesture"] + class_labels

    class_labels = sorted(class_labels)

    return database, class_labels


def convert_mlgesture_to_activitynet_json(video_root, train_path, val_path, dst_json_path, modality):

    train_database, labels_train = convert_mlgesture_to_dict(video_root, train_path, 'training', modality)
    val_database, labels_val = convert_mlgesture_to_dict(video_root, val_path, 'validation', modality)

    assert labels_train == labels_val, "Labels in the training set must be the same as the labels in the validation set"

    dst_data = {}
    dst_data['labels'] = labels_train
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':

    modality = ['cam_front', 'cam_top',
                'mlx90640_front', 'mlx90640_top',
                'mlx90641_front', 'mlx90641_top',
                'lepton_front', 'lepton_top',
                'flow_front', 'flow_top',
                'depth_front', 'depth_top']
    parser = argparse.ArgumentParser("Create Activity net format json annotation file")
    parser.add_argument("root", help="Root folder with videos")
    parser.add_argument("train_json", help="MlGesture annotation file for training")
    parser.add_argument("validation_json", help="MlGesture annotation file for validation")
    parser.add_argument("outfile", help="Json output file to create")
    parser.add_argument("modality", choices=modality, help="Sensor modality to include")
    args = parser.parse_args()

    convert_mlgesture_to_activitynet_json(args.root, args.train_json, args.validation_json, args.outfile, args.modality)

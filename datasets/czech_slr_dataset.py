import ast
import torch
import numpy as np

import pandas as pd
import torch.utils.data as torch_data

from random import randrange
from augmentations import *
from normalization.body_normalization import BODY_IDENTIFIERS
from normalization.hand_normalization import HAND_IDENTIFIERS
from normalization.body_normalization import normalize_single_dict as normalize_single_body_dict
from normalization.hand_normalization import normalize_single_dict as normalize_single_hand_dict

ORI_HAND_IDENTIFIERS = HAND_IDENTIFIERS
HAND_IDENTIFIERS = [id + "_0" for id in HAND_IDENTIFIERS] + [id + "_1" for id in HAND_IDENTIFIERS]


def remove_data(df, num_remove, remove_from):
    length = len(ast.literal_eval(df["indexDIP_left_X"][0]))

    ori_identifiers, extr_identifier = [], []
    if remove_from == 'right_hand':
        ori_identifiers = ORI_HAND_IDENTIFIERS
        extr_identifier = ["_right_X", "_right_Y"]
    elif remove_from == 'left_hand':
        ori_identifiers = ORI_HAND_IDENTIFIERS
        extr_identifier = ["_left_X", "_left_Y"]
    elif remove_from == 'body_face':
        ori_identifiers = BODY_IDENTIFIERS
        extr_identifier = ["_X", "_Y"]

    for i in range(num_remove):
        ori_identifier = ori_identifiers[i]
        identifier_list = [ori_identifier + extr_identifier[0], ori_identifier + extr_identifier[1]]
        for identifier in identifier_list:
            df[identifier] = str([0 for _ in range(length)])

    return df


def load_dataset(file_location: str, num_remove=0, remove_from=None):

    # Load the datset csv file
    df = pd.read_csv(file_location, encoding="utf-8")

    if remove_from is not None:
        df = remove_data(df, num_remove, remove_from)

    # TO BE DELETED
    df.columns = [item.replace("_left_", "_0_").replace("_right_", "_1_") for item in list(df.columns)]
    if "neck_X" not in df.columns:
        df["neck_X"] = [0 for _ in range(df.shape[0])]
        df["neck_Y"] = [0 for _ in range(df.shape[0])]

    # TEMP
    labels = df["labels"].to_list()
    # labels = [label + 1 for label in df["labels"].to_list()]
    data = []

    for row_index, row in df.iterrows():
        current_row = np.empty(shape=(len(ast.literal_eval(row["leftEar_X"])), len(BODY_IDENTIFIERS + HAND_IDENTIFIERS), 2))
        for index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
            current_row[:, index, 0] = ast.literal_eval(row[identifier + "_X"])
            current_row[:, index, 1] = ast.literal_eval(row[identifier + "_Y"])

        data.append(current_row)

    return data, labels


def tensor_to_dictionary(landmarks_tensor: torch.Tensor) -> dict:

    data_array = landmarks_tensor.numpy()
    output = {}

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS + HAND_IDENTIFIERS):
        output[identifier] = data_array[:, landmark_index]

    return output


def dictionary_to_tensor(landmarks_dict: dict, identifier_lst: list) -> torch.Tensor:

    sequence_len = len(next(iter(landmarks_dict.values()), None))
    output = np.empty(shape=(sequence_len, len(identifier_lst), 2))

    for landmark_index, identifier in enumerate(identifier_lst):
        output[:, landmark_index, 0] = [frame[0] for frame in landmarks_dict[identifier]]
        output[:, landmark_index, 1] = [frame[1] for frame in landmarks_dict[identifier]]

    return torch.from_numpy(output)


def compute_position_features(depth_map_dict: dict, reference_points: list = ['nose', 'neck']) -> tuple:
    """
    Compute position features (relative to body reference point) for hands.
    
    Args:
        depth_map_dict: Dictionary containing landmark coordinates
        reference_points: List of body points to use as reference (will average them)
        
    Returns:
        left_hand_pos: (T, 2) - Normalized position of left wrist relative to reference
        right_hand_pos: (T, 2) - Normalized position of right wrist relative to reference
    """
    # Get sequence length
    first_key = list(depth_map_dict.keys())[0]
    sequence_len = len(depth_map_dict[first_key])
    
    # Compute reference point (average of nose and neck)
    reference_coords = np.zeros((sequence_len, 2))
    valid_refs = 0
    
    for ref_point in reference_points:
        if ref_point in depth_map_dict:
            ref_data = np.array(depth_map_dict[ref_point])  # (T, 2)
            reference_coords += ref_data
            valid_refs += 1
    
    if valid_refs > 0:
        reference_coords /= valid_refs
    
    # Get wrist positions
    left_wrist_key = 'wrist_0'  # Left hand wrist
    right_wrist_key = 'wrist_1'  # Right hand wrist
    
    left_wrist = np.array(depth_map_dict.get(left_wrist_key, np.zeros((sequence_len, 2))))
    right_wrist = np.array(depth_map_dict.get(right_wrist_key, np.zeros((sequence_len, 2))))
    
    # Compute relative positions
    left_hand_pos = left_wrist - reference_coords
    right_hand_pos = right_wrist - reference_coords
    
    # Normalize to [-1, 1] range (assuming image size ~640x480)
    # This makes position features scale-invariant
    left_hand_pos = left_hand_pos / np.array([320.0, 240.0])
    right_hand_pos = right_hand_pos / np.array([320.0, 240.0])
    
    # Clip to [-1, 1]
    left_hand_pos = np.clip(left_hand_pos, -1.0, 1.0)
    right_hand_pos = np.clip(right_hand_pos, -1.0, 1.0)
    
    return left_hand_pos, right_hand_pos


def augment_with_position(hand_tensor: torch.Tensor, position_features: np.ndarray) -> torch.Tensor:
    """
    Augment hand tensor with position features.
    
    Args:
        hand_tensor: (T, 21, 2) - Shape-normalized hand landmarks
        position_features: (T, 2) - Position of wrist relative to body
        
    Returns:
        augmented: (T, 21, 4) - [x, y, rel_x, rel_y]
    """
    T, N, C = hand_tensor.shape
    
    # Convert position features to tensor
    pos_tensor = torch.from_numpy(position_features).float()  # (T, 2)
    
    # Expand position to all landmarks (same position for all points in a hand)
    pos_expanded = pos_tensor.unsqueeze(1).expand(T, N, 2)  # (T, 21, 2)
    
    # Concatenate: [shape_coords, position_coords]
    augmented = torch.cat([hand_tensor, pos_expanded], dim=-1)  # (T, 21, 4)
    
    return augmented


def isolate_single_body_dit(row: dict):
    body_identifiers = BODY_IDENTIFIERS  # [0:6] on face [6:] on body

    body_dit = {key: row[key] for key in body_identifiers if key in row}
    # print(f'Keys of the extracted body dit: {list(body_dit.keys())} of length of {len(body_dit.keys())}')
    return body_dit, body_identifiers


def isolate_single_hand(row: dict, hand_index: int):
    hand_identifiers = []
    for identifier in HAND_IDENTIFIERS:
        if identifier[-1] == str(hand_index):
            hand_identifiers.append(identifier)

    hand_dit = {key: row[key] for key in hand_identifiers if key in row}
    # which_hand = "left" if hand_index == 0 else "right"
    # print(f'Keys of the extracted {which_hand} hand dit: {list(hand_dit.keys())} of length of {len(hand_dit.keys())}')
    return hand_dit, hand_identifiers


class CzechSLRDataset(torch_data.Dataset):
    """Advanced object representation of the HPOES dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties"""

    data: [np.ndarray]
    labels: [np.ndarray]

    def __init__(self, dataset_filename: str, num_labels=5, transform=None, augmentations=False,
                 augmentations_prob=0.5, normalize=True, num_remove=0, remove_from=None, use_position=True):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param transform: Any data transformation to be applied (default: None)
        :param use_position: Whether to augment hand data with position features (default: False)
        """

        loaded_data = load_dataset(file_location=dataset_filename, num_remove=num_remove, remove_from=remove_from)
        data, labels = loaded_data[0], loaded_data[1]

        self.data = data
        self.labels = labels
        self.targets = list(labels)
        self.num_labels = num_labels
        self.transform = transform

        self.augmentations = augmentations
        self.augmentations_prob = augmentations_prob
        self.normalize = normalize
        self.use_position = use_position
        
        if self.use_position:
            print("using right")
            print(f"✓ CzechSLRDataset: Position features ENABLED (hands will have 4 channels: x, y, rel_x, rel_y)")
        else:
            print(f"  CzechSLRDataset: Position features DISABLED (hands will have 2 channels: x, y only)")

    def __getitem__(self, idx):
        """
        Allocates, potentially transforms and returns the item at the desired index.

        :param idx: Index of the item
        :return: Tuple containing both the depth map and the label
        """

        depth_map = torch.from_numpy(np.copy(self.data[idx]))
        label = torch.Tensor([self.labels[idx]])

        depth_map = tensor_to_dictionary(depth_map)

        # Apply potential augmentations
        if self.augmentations and random.random() < self.augmentations_prob:

            selected_aug = randrange(5)

            if selected_aug == 0:
                depth_map = augment_rotate(depth_map, (-13, 13))

            if selected_aug == 1:
                depth_map = augment_shear(depth_map, "perspective", (0, 0.1))

            if selected_aug == 2:
                depth_map = augment_shear(depth_map, "squeeze", (0, 0.15))

            if selected_aug == 3:
                depth_map = augment_arm_joint_rotate(depth_map, 0.3, (-4, 4))

            if selected_aug == 4:
                depth_map = depth_map

        if self.normalize:
            depth_map = normalize_single_body_dict(depth_map)
            depth_map = normalize_single_hand_dict(depth_map)

        # Compute position features BEFORE isolating hands (need full depth_map for reference points)
        if self.use_position:
            left_pos, right_pos = compute_position_features(depth_map)
        
        l_hand_depth_map, l_hand_identifiers = isolate_single_hand(depth_map, 0)
        r_hand_depth_map, r_hand_identifiers = isolate_single_hand(depth_map, 1)
        body_depth_map, body_identifiers = isolate_single_body_dit(depth_map)

        l_hand_depth_map = dictionary_to_tensor(l_hand_depth_map, l_hand_identifiers) - 0.5
        r_hand_depth_map = dictionary_to_tensor(r_hand_depth_map, r_hand_identifiers) - 0.5
        body_depth_map = dictionary_to_tensor(body_depth_map, body_identifiers) - 0.5
        
        # Augment with position features if enabled
        if self.use_position:
            l_hand_depth_map = augment_with_position(l_hand_depth_map, left_pos)
            r_hand_depth_map = augment_with_position(r_hand_depth_map, right_pos)
            # Body keeps 2 channels (no position augmentation for body)

        if self.transform:
            l_hand_depth_map = self.transform(l_hand_depth_map)  # (T, 21, 4) or (T, 21, 2)
            r_hand_depth_map = self.transform(r_hand_depth_map)  # (T, 21, 4) or (T, 21, 2)
            body_depth_map = self.transform(body_depth_map)  # (T, 12, 2)

        # print(f"body_depth_map.shape {body_depth_map.shape}")
        # print(f"l_hand_depth_map.shape {l_hand_depth_map.shape}")
        # print(f"r_hand_depth_map.shape {r_hand_depth_map.shape}")

        # print("All ok now")
        # print(error)
        #
        # depth_map = dictionary_to_tensor(depth_map, HAND_IDENTIFIERS+BODY_IDENTIFIERS)
        #
        # # Move the landmark position interval to improve performance
        # depth_map = depth_map - 0.5
        #
        # if self.transform:
        #     depth_map = self.transform(depth_map)

        return l_hand_depth_map, r_hand_depth_map, body_depth_map, label

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    pass

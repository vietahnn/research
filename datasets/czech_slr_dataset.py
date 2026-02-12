import ast
import torch

import pandas as pd
import torch.utils.data as torch_data

from random import randrange
from augmentations import *
from normalization.body_normalization import BODY_IDENTIFIERS
from normalization.hand_normalization import HAND_IDENTIFIERS
from normalization.body_normalization import normalize_single_dict as normalize_single_body_dict
from normalization.hand_normalization import normalize_single_dict as normalize_single_hand_dict


def detect_key_frames(depth_map_dict: dict, target_frames: int = 120, min_motion_threshold: float = 0.001):
    """
    Detect key frames based on motion magnitude.
    
    Args:
        depth_map_dict: Dictionary containing landmark positions over time
        target_frames: Target number of frames to keep (default: 120, reduced from ~204)
        min_motion_threshold: Minimum motion to consider a frame as important
    
    Returns:
        List of selected frame indices
    """
    # Get sequence length from any landmark
    sequence_len = len(next(iter(depth_map_dict.values()), None))
    
    # If sequence is already short, keep all frames
    if sequence_len <= target_frames:
        return list(range(sequence_len))
    
    # Calculate motion magnitude for each frame
    motion_magnitudes = np.zeros(sequence_len - 1)
    
    for identifier, positions in depth_map_dict.items():
        for frame_idx in range(sequence_len - 1):
            # Calculate Euclidean distance between consecutive frames
            diff_x = positions[frame_idx + 1][0] - positions[frame_idx][0]
            diff_y = positions[frame_idx + 1][1] - positions[frame_idx][1]
            motion = np.sqrt(diff_x**2 + diff_y**2)
            motion_magnitudes[frame_idx] += motion
    
    # Normalize motion magnitudes
    motion_magnitudes = motion_magnitudes / len(depth_map_dict)
    
    # Always keep first and last frame
    selected_indices = [0, sequence_len - 1]
    
    # Calculate window size for uniform coverage
    window_size = max(1, (sequence_len - 2) // (target_frames - 2))
    
    # Select key frames from each window based on motion magnitude
    for window_start in range(1, sequence_len - 1, window_size):
        window_end = min(window_start + window_size, sequence_len - 1)
        
        # Find frame with maximum motion in this window
        window_motions = motion_magnitudes[window_start:window_end]
        
        if len(window_motions) > 0:
            max_motion_idx = window_start + np.argmax(window_motions)
            
            # Only add if motion is significant or we need more frames
            if motion_magnitudes[max_motion_idx] > min_motion_threshold or len(selected_indices) < target_frames:
                if max_motion_idx not in selected_indices:
                    selected_indices.append(max_motion_idx)
    
    # Sort indices
    selected_indices = sorted(selected_indices)
    
    # If we have too many, keep the ones with highest motion
    if len(selected_indices) > target_frames:
        # Keep first and last
        first_last = [selected_indices[0], selected_indices[-1]]
        middle_indices = selected_indices[1:-1]
        
        # Sort middle indices by their motion magnitude
        middle_with_motion = [(idx, motion_magnitudes[min(idx, len(motion_magnitudes)-1)]) 
                               for idx in middle_indices]
        middle_with_motion.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top (target_frames - 2) middle frames
        top_middle = sorted([idx for idx, _ in middle_with_motion[:target_frames-2]])
        selected_indices = [first_last[0]] + top_middle + [first_last[1]]
    
    return selected_indices


def apply_frame_sampling(depth_map_dict: dict, selected_indices: list) -> dict:
    """
    Apply frame sampling to the depth map dictionary.
    
    Args:
        depth_map_dict: Dictionary containing landmark positions
        selected_indices: List of frame indices to keep
    
    Returns:
        Dictionary with only selected frames
    """
    sampled_dict = {}
    
    for identifier, positions in depth_map_dict.items():
        sampled_dict[identifier] = [positions[idx] for idx in selected_indices]
    
    return sampled_dict

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
                 augmentations_prob=0.5, normalize=True, num_remove=0, remove_from=None,
                 use_frame_sampling=True, target_frames=120):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param transform: Any data transformation to be applied (default: None)
        :param use_frame_sampling: Whether to apply key frame sampling (default: True)
        :param target_frames: Target number of frames after sampling (default: 120)
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
        self.use_frame_sampling = use_frame_sampling
        self.target_frames = target_frames

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

        # Apply frame sampling to reduce temporal redundancy
        if self.use_frame_sampling:
            key_frame_indices = detect_key_frames(depth_map, target_frames=self.target_frames)
            depth_map = apply_frame_sampling(depth_map, key_frame_indices)

        if self.normalize:
            depth_map = normalize_single_body_dict(depth_map)
            depth_map = normalize_single_hand_dict(depth_map)

        l_hand_depth_map, l_hand_identifiers = isolate_single_hand(depth_map, 0)
        r_hand_depth_map, r_hand_identifiers = isolate_single_hand(depth_map, 1)
        body_depth_map, body_identifiers = isolate_single_body_dit(depth_map)

        l_hand_depth_map = dictionary_to_tensor(l_hand_depth_map, l_hand_identifiers) - 0.5
        r_hand_depth_map = dictionary_to_tensor(r_hand_depth_map, r_hand_identifiers) - 0.5
        body_depth_map = dictionary_to_tensor(body_depth_map, body_identifiers) - 0.5

        if self.transform:
            l_hand_depth_map = self.transform(l_hand_depth_map)  # (B, 204, 21, 2)
            r_hand_depth_map = self.transform(r_hand_depth_map)  # (B, 204, 21, 2)
            body_depth_map = self.transform(body_depth_map)  # (B, 204, 12, 2)

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


import math
import logging
import cv2
import random

import numpy as np

from normalization.body_normalization import BODY_IDENTIFIERS
from normalization.hand_normalization import HAND_IDENTIFIERS


HAND_IDENTIFIERS = [id + "_0" for id in HAND_IDENTIFIERS] + [id + "_1" for id in HAND_IDENTIFIERS]
ARM_IDENTIFIERS_ORDER = ["neck", "$side$Shoulder", "$side$Elbow", "$side$Wrist"]


def __random_pass(prob):
    return random.random() < prob


def __numpy_to_dictionary(data_array: np.ndarray) -> dict:
    """
    Supplementary method converting a NumPy array of body landmark data into dictionaries. The array data must match the
    order of the BODY_IDENTIFIERS list.
    """

    output = {}

    for landmark_index, identifier in enumerate(BODY_IDENTIFIERS):
        output[identifier] = data_array[:, landmark_index].tolist()

    return output


def __dictionary_to_numpy(landmarks_dict: dict) -> np.ndarray:
    """
    Supplementary method converting dictionaries of body landmark data into respective NumPy arrays. The resulting array
    will match the order of the BODY_IDENTIFIERS list.
    """
    sequence_len = len(next(iter(landmarks_dict.values()), None))

    output = np.empty(shape=(sequence_len, len(landmarks_dict), 2))

    for landmark_index, identifier in enumerate(landmarks_dict):
        output[:, landmark_index, 0] = np.array(landmarks_dict[identifier])[:, 0]
        output[:, landmark_index, 1] = np.array(landmarks_dict[identifier])[:, 1]

    return output


def __rotate(origin: tuple, point: tuple, angle: float):
    """
    Rotates a point counterclockwise by a given angle around a given origin.

    :param origin: Landmark in the (X, Y) format of the origin from which to count angle of rotation
    :param point: Landmark in the (X, Y) format to be rotated
    :param angle: Angle under which the point shall be rotated
    :return: New landmarks (coordinates)
    """

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy


def __preprocess_row_sign(sign: dict) -> (dict, dict):
    """
    Supplementary method splitting the single-dictionary skeletal data into two dictionaries of body and hand landmarks
    respectively.
    """

    sign_eval = sign

    if "nose_X" in sign_eval:
        body_landmarks = {identifier: [(x, y) for x, y in zip(sign_eval[identifier + "_X"], sign_eval[identifier + "_Y"])]
                          for identifier in BODY_IDENTIFIERS}
        hand_landmarks = {identifier: [(x, y) for x, y in zip(sign_eval[identifier + "_X"], sign_eval[identifier + "_Y"])]
                          for identifier in HAND_IDENTIFIERS}

    else:
        body_landmarks = {identifier: sign_eval[identifier] for identifier in BODY_IDENTIFIERS}
        hand_landmarks = {identifier: sign_eval[identifier] for identifier in HAND_IDENTIFIERS}

    return body_landmarks, hand_landmarks


def __wrap_sign_into_row(body_identifiers: dict, hand_identifiers: dict) -> dict:
    """
    Supplementary method for merging body and hand data into a single dictionary.
    """

    return {**body_identifiers, **hand_identifiers}


def augment_rotate(sign: dict, angle_range: tuple) -> dict:
    """
    AUGMENTATION TECHNIQUE. All the joint coordinates in each frame are rotated by a random angle up to 13 degrees with
    the center of rotation lying in the center of the frame, which is equal to [0.5; 0.5].

    :param sign: Dictionary with sequential skeletal data of the signing person
    :param angle_range: Tuple containing the angle range (minimal and maximal angle in degrees) to randomly choose the
                        angle by which the landmarks will be rotated from

    :return: Dictionary with augmented (by rotation) sequential skeletal data of the signing person
    """

    body_landmarks, hand_landmarks = __preprocess_row_sign(sign)
    angle = math.radians(random.uniform(*angle_range))

    body_landmarks = {key: [__rotate((0.5, 0.5), frame, angle) for frame in value] for key, value in
                      body_landmarks.items()}
    hand_landmarks = {key: [__rotate((0.5, 0.5), frame, angle) for frame in value] for key, value in
                      hand_landmarks.items()}

    return __wrap_sign_into_row(body_landmarks, hand_landmarks)


def augment_shear(sign: dict, type: str, squeeze_ratio: tuple) -> dict:
    """
    AUGMENTATION TECHNIQUE.

        - Squeeze. All the frames are squeezed from both horizontal sides. Two different random proportions up to 15% of
        the original frame's width for both left and right side are cut.

        - Perspective transformation. The joint coordinates are projected onto a new plane with a spatially defined
        center of projection, which simulates recording the sign video with a slight tilt. Each time, the right or left
        side, as well as the proportion by which both the width and height will be reduced, are chosen randomly. This
        proportion is selected from a uniform distribution on the [0; 1) interval. Subsequently, the new plane is
        delineated by reducing the width at the desired side and the respective vertical edge (height) at both of its
        adjacent corners.

    :param sign: Dictionary with sequential skeletal data of the signing person
    :param type: Type of shear augmentation to perform (either 'squeeze' or 'perspective')
    :param squeeze_ratio: Tuple containing the relative range from what the proportion of the original width will be
                          randomly chosen. These proportions will either be cut from both sides or used to construct the
                          new projection

    :return: Dictionary with augmented (by squeezing or perspective transformation) sequential skeletal data of the
             signing person
    """

    body_landmarks, hand_landmarks = __preprocess_row_sign(sign)

    if type == "squeeze":
        move_left = random.uniform(*squeeze_ratio)
        move_right = random.uniform(*squeeze_ratio)

        src = np.array(((0, 1), (1, 1), (0, 0), (1, 0)), dtype=np.float32)
        dest = np.array(((0 + move_left, 1), (1 - move_right, 1), (0 + move_left, 0), (1 - move_right, 0)),
                        dtype=np.float32)
        mtx = cv2.getPerspectiveTransform(src, dest)

    elif type == "perspective":

        move_ratio = random.uniform(*squeeze_ratio)
        src = np.array(((0, 1), (1, 1), (0, 0), (1, 0)), dtype=np.float32)

        if __random_pass(0.5):
            dest = np.array(((0 + move_ratio, 1 - move_ratio), (1, 1), (0 + move_ratio, 0 + move_ratio), (1, 0)),
                            dtype=np.float32)
        else:
            dest = np.array(((0, 1), (1 - move_ratio, 1 - move_ratio), (0, 0), (1 - move_ratio, 0 + move_ratio)),
                            dtype=np.float32)

        mtx = cv2.getPerspectiveTransform(src, dest)

    else:

        logging.error("Unsupported shear type provided.")
        return {}

    landmarks_array = __dictionary_to_numpy(body_landmarks)
    augmented_landmarks = cv2.perspectiveTransform(np.array(landmarks_array, dtype=np.float32), mtx)

    augmented_zero_landmark = cv2.perspectiveTransform(np.array([[[0, 0]]], dtype=np.float32), mtx)[0][0]
    augmented_landmarks = np.stack([np.where(sub == augmented_zero_landmark, [0, 0], sub) for sub in augmented_landmarks])

    body_landmarks = __numpy_to_dictionary(augmented_landmarks)

    return __wrap_sign_into_row(body_landmarks, hand_landmarks)


def augment_arm_joint_rotate(sign: dict, probability: float, angle_range: tuple) -> dict:
    """
    AUGMENTATION TECHNIQUE. The joint coordinates of both arms are passed successively, and the impending landmark is
    slightly rotated with respect to the current one. The chance of each joint to be rotated is 3:10 and the angle of
    alternation is a uniform random angle up to +-4 degrees. This simulates slight, negligible variances in each
    execution of a sign, which do not change its semantic meaning.

    :param sign: Dictionary with sequential skeletal data of the signing person
    :param probability: Probability of each joint to be rotated (float from the range [0, 1])
    :param angle_range: Tuple containing the angle range (minimal and maximal angle in degrees) to randomly choose the
                        angle by which the landmarks will be rotated from

    :return: Dictionary with augmented (by arm joint rotation) sequential skeletal data of the signing person
    """

    body_landmarks, hand_landmarks = __preprocess_row_sign(sign)

    # Iterate over both directions (both hands)
    for side in ["left", "right"]:
        # Iterate gradually over the landmarks on arm
        for landmark_index, landmark_origin in enumerate(ARM_IDENTIFIERS_ORDER):
            landmark_origin = landmark_origin.replace("$side$", side)

            # End the process on the current hand if the landmark is not present
            if landmark_origin not in body_landmarks:
                break

            # Perform rotation by provided probability
            if __random_pass(probability):
                angle = math.radians(random.uniform(*angle_range))

                for to_be_rotated in ARM_IDENTIFIERS_ORDER[landmark_index + 1:]:
                    to_be_rotated = to_be_rotated.replace("$side$", side)

                    # Skip if the landmark is not present
                    if to_be_rotated not in body_landmarks:
                        continue

                    body_landmarks[to_be_rotated] = [__rotate(body_landmarks[landmark_origin][frame_index], frame,
                        angle) for frame_index, frame in enumerate(body_landmarks[to_be_rotated])]

    return __wrap_sign_into_row(body_landmarks, hand_landmarks)


def augment_temporal_mask(sign: dict, mask_ratio: float = 0.1, mask_consecutive: bool = True) -> dict:
    """
    TEMPORAL AUGMENTATION TECHNIQUE. Randomly masks (sets to zero) a portion of frames in the sequence.
    This simulates temporal occlusion or missing frames.
    
    :param sign: Dictionary with sequential skeletal data of the signing person
    :param mask_ratio: Proportion of frames to mask (default: 0.1 means 10% of frames)
    :param mask_consecutive: If True, mask consecutive frames; if False, mask random scattered frames
    
    :return: Dictionary with temporally masked sequential skeletal data
    """
    body_landmarks, hand_landmarks = __preprocess_row_sign(sign)
    
    # Get sequence length
    seq_len = len(next(iter(body_landmarks.values())))
    num_mask = int(seq_len * mask_ratio)
    
    if num_mask == 0:
        return __wrap_sign_into_row(body_landmarks, hand_landmarks)
    
    if mask_consecutive:
        # Mask consecutive frames
        start_idx = random.randint(0, seq_len - num_mask)
        mask_indices = list(range(start_idx, start_idx + num_mask))
    else:
        # Mask random scattered frames
        mask_indices = random.sample(range(seq_len), num_mask)
    
    # Apply masking to all landmarks
    for key in body_landmarks:
        for idx in mask_indices:
            body_landmarks[key][idx] = (0.0, 0.0)
    
    for key in hand_landmarks:
        for idx in mask_indices:
            hand_landmarks[key][idx] = (0.0, 0.0)
    
    return __wrap_sign_into_row(body_landmarks, hand_landmarks)


def augment_temporal_subsample(sign: dict, subsample_ratio: float = 0.85) -> dict:
    """
    TEMPORAL AUGMENTATION TECHNIQUE. Randomly drops frames and interpolates to maintain sequence length.
    This simulates temporal variations in signing speed.
    
    :param sign: Dictionary with sequential skeletal data of the signing person
    :param subsample_ratio: Ratio of frames to keep (default: 0.85 means keep 85% of frames)
    
    :return: Dictionary with temporally subsampled and interpolated sequential skeletal data
    """
    body_landmarks, hand_landmarks = __preprocess_row_sign(sign)
    
    # Get sequence length
    seq_len = len(next(iter(body_landmarks.values())))
    num_keep = int(seq_len * subsample_ratio)
    
    if num_keep >= seq_len or num_keep < 2:
        return __wrap_sign_into_row(body_landmarks, hand_landmarks)
    
    # Randomly select frames to keep (always keep first and last)
    keep_indices = sorted(random.sample(range(1, seq_len - 1), num_keep - 2))
    keep_indices = [0] + keep_indices + [seq_len - 1]
    
    # Interpolate back to original length
    def interpolate_sequence(values, keep_idx, target_len):
        kept_values = [values[i] for i in keep_idx]
        interpolated = []
        
        for i in range(target_len):
            # Find surrounding kept frames
            ratio = i / (target_len - 1) * (len(kept_values) - 1)
            idx_low = int(ratio)
            idx_high = min(idx_low + 1, len(kept_values) - 1)
            alpha = ratio - idx_low
            
            # Linear interpolation
            x = kept_values[idx_low][0] * (1 - alpha) + kept_values[idx_high][0] * alpha
            y = kept_values[idx_low][1] * (1 - alpha) + kept_values[idx_high][1] * alpha
            interpolated.append((x, y))
        
        return interpolated
    
    # Apply to all landmarks
    body_landmarks = {key: interpolate_sequence(value, keep_indices, seq_len) 
                      for key, value in body_landmarks.items()}
    hand_landmarks = {key: interpolate_sequence(value, keep_indices, seq_len) 
                      for key, value in hand_landmarks.items()}
    
    return __wrap_sign_into_row(body_landmarks, hand_landmarks)


def augment_spatial_mask(sign: dict, mask_ratio: float = 0.15, mask_hands_prob: float = 0.3, 
                          mask_body_prob: float = 0.3) -> dict:
    """
    SPATIAL AUGMENTATION TECHNIQUE. Randomly masks (sets to zero) a subset of keypoints.
    This simulates partial occlusion or detection failures.
    
    :param sign: Dictionary with sequential skeletal data of the signing person
    :param mask_ratio: Proportion of keypoints to mask
    :param mask_hands_prob: Probability of masking hand keypoints
    :param mask_body_prob: Probability of masking body keypoints
    
    :return: Dictionary with spatially masked sequential skeletal data
    """
    body_landmarks, hand_landmarks = __preprocess_row_sign(sign)
    
    # Mask hand keypoints
    if __random_pass(mask_hands_prob):
        num_hand_keypoints = len(hand_landmarks)
        num_mask = int(num_hand_keypoints * mask_ratio)
        
        if num_mask > 0:
            keys_to_mask = random.sample(list(hand_landmarks.keys()), num_mask)
            for key in keys_to_mask:
                hand_landmarks[key] = [(0.0, 0.0) for _ in hand_landmarks[key]]
    
    # Mask body keypoints (but avoid masking critical reference points like neck)
    if __random_pass(mask_body_prob):
        maskable_body_keys = [k for k in body_landmarks.keys() if k not in ['neck']]
        num_body_keypoints = len(maskable_body_keys)
        num_mask = int(num_body_keypoints * mask_ratio)
        
        if num_mask > 0:
            keys_to_mask = random.sample(maskable_body_keys, num_mask)
            for key in keys_to_mask:
                body_landmarks[key] = [(0.0, 0.0) for _ in body_landmarks[key]]
    
    return __wrap_sign_into_row(body_landmarks, hand_landmarks)


def augment_temporal_crop(sign: dict, crop_ratio: float = 0.9) -> dict:
    """
    TEMPORAL AUGMENTATION TECHNIQUE. Randomly crops the sequence and pads or repeats to maintain length.
    
    :param sign: Dictionary with sequential skeletal data of the signing person
    :param crop_ratio: Ratio of sequence to keep (default: 0.9 means keep 90%)
    
    :return: Dictionary with temporally cropped sequential skeletal data
    """
    body_landmarks, hand_landmarks = __preprocess_row_sign(sign)
    
    seq_len = len(next(iter(body_landmarks.values())))
    crop_len = int(seq_len * crop_ratio)
    
    if crop_len >= seq_len or crop_len < 1:
        return __wrap_sign_into_row(body_landmarks, hand_landmarks)
    
    # Random start position
    start_idx = random.randint(0, seq_len - crop_len)
    end_idx = start_idx + crop_len
    
    # Crop and pad/repeat to maintain original length
    def crop_and_pad(values, start, end, target_len):
        # Convert to list to handle both numpy arrays and lists
        cropped = list(values[start:end])
        
        # Pad to original length by repeating last frame
        while len(cropped) < target_len:
            cropped.append(cropped[-1])
        
        return cropped[:target_len]
    
    body_landmarks = {key: crop_and_pad(value, start_idx, end_idx, seq_len) 
                      for key, value in body_landmarks.items()}
    hand_landmarks = {key: crop_and_pad(value, start_idx, end_idx, seq_len) 
                      for key, value in hand_landmarks.items()}
    
    return __wrap_sign_into_row(body_landmarks, hand_landmarks)


if __name__ == "__main__":
    pass

#    The reference for the code is the following
#    Sign Pose-based Transformer for Word-level Sign Language Recognition
#    Author: Matyáš Boháček and Marek Hrúz
#    Availability: https://github.com/matyasbohacek/spoter

# utils


import logging
import torch
import torch.nn.functional as F
import time
from statistics import mean
from siformer.contrastive_loss import CrossModalContrastiveLoss


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None, 
                use_contrastive=True, contrastive_weight=0.5):
    """
    Train for one epoch with optional cross-modal contrastive learning.
    
    Args:
        model: SiFormer model
        dataloader: Training data loader
        criterion: Classification loss (CrossEntropyLoss)
        optimizer: Optimizer
        device: Device to train on
        scheduler: Optional learning rate scheduler
        use_contrastive: Enable cross-modal contrastive learning (default: True)
        contrastive_weight: Weight for contrastive loss (default: 0.5)
    
    Returns:
        running_loss, pred_correct, pred_all, accuracy, avg_train_time
    """
    pred_correct, pred_all = 0, 0
    running_loss = 0.0
    train_time_sec_list = []
    
    # Initialize contrastive loss (enabled by default)
    if use_contrastive:
        contrastive_criterion = CrossModalContrastiveLoss(
            temperature=0.07,
            projection_dim=128,
            d_lhand=42,
            d_rhand=42,
            d_body=24
        ).to(device)
        print(f"Cross-Modal Contrastive Learning ENABLED (weight={contrastive_weight})")
    
    for i, data in enumerate(dataloader):
        l_hands, r_hands, bodies, labels = data
        l_hands = l_hands.to(device)
        r_hands = r_hands.to(device)
        bodies = bodies.to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        start_time = time.time()

        # Forward pass with feature extraction for contrastive learning
        if use_contrastive:
            outputs, lh_feat, rh_feat, body_feat = model(
                l_hands, r_hands, bodies, training=True, return_features=True
            )
        else:
            outputs = model(l_hands, r_hands, bodies, training=True)

        end_time = time.time()
        train_time_sec = end_time - start_time
        train_time_sec_list.append(train_time_sec)

        # Compute classification loss
        ce_loss = criterion(outputs, labels.squeeze(1))
        
        # Compute contrastive loss if enabled
        if use_contrastive:
            contrastive_loss = contrastive_criterion(
                lh_feat, rh_feat, body_feat, labels.squeeze(1)
            )
            # Combined loss
            loss = ce_loss + contrastive_weight * contrastive_loss
            
            # Log both losses occasionally
            if i % 50 == 0 and i > 0:
                print(f"  Batch {i}: CE Loss={ce_loss.item():.4f}, Contrastive Loss={contrastive_loss.item():.4f}")
        else:
            loss = ce_loss
        
        loss.backward()
        optimizer.step()
        running_loss += loss

        # Statistics
        _, preds = torch.max(F.softmax(outputs, dim=1), 1)
        # print(f'preds: {preds}')
        # print(f'label: {labels.view(-1)}')
        pred_correct += torch.sum(preds == labels.view(-1)).item()
        pred_all += labels.size(0)

    if scheduler:
        scheduler.step()

    avg_train_time = mean(train_time_sec_list)

    return running_loss, pred_correct, pred_all, (pred_correct / pred_all), avg_train_time


def evaluate(model, dataloader, device, print_stats=False):
    pred_correct, pred_all = 0, 0
    stats = {i: [0, 0] for i in range(100)}

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            l_hands, r_hands, bodies, labels = data
            l_hands = l_hands.to(device)  # [24, 204, 21, 2]
            r_hands = r_hands.to(device)  # [24, 204, 21, 2]
            bodies = bodies.to(device)  # [24, 204, 12, 2]
            labels = labels.to(device, dtype=torch.long)  # [24, 1]

            for j in range(labels.size(0)):
                l_hand = l_hands[j].unsqueeze(0)  # [1, 204, 21, 2]
                r_hand = r_hands[j].unsqueeze(0)  # [1, 204, 21, 2]
                body = bodies[j].unsqueeze(0)  # [1, 204, 12, 2]
                label = labels[j]

                output = model(l_hand, r_hand, body, training=False)
                output = output.unsqueeze(0).expand(1, -1, -1)

                # Statistics
                if int(torch.argmax(torch.nn.functional.softmax(output, dim=2))) == int(label):
                    stats[int(labels[0][0])][0] += 1
                    pred_correct += 1

                stats[int(labels[0][0])][1] += 1
                pred_all += 1

    if print_stats:
        stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
        print("Label accuracies statistics:")
        print(str(stats) + "\n")
        logging.info("Label accuracies statistics:")
        logging.info(str(stats) + "\n")

    return pred_correct, pred_all, (pred_correct / pred_all)


def evaluate_top_k(model, dataloader, device, k=5):
    pred_correct, pred_all = 0, 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            l_hands, r_hands, bodies, labels = data
            l_hands = l_hands.to(device)
            r_hands = r_hands.to(device)
            bodies = bodies.to(device)
            labels = labels.to(device, dtype=torch.long)

            for j in range(labels.size(0)):
                l_hand = l_hands[j].unsqueeze(0)  # [1, 204, 21, 2]
                r_hand = r_hands[j].unsqueeze(0)  # [1, 204, 21, 2]
                body = bodies[j].unsqueeze(0)  # [1, 204, 12, 2]
                label = labels[j]

                output = model(l_hand, r_hand, body, training=False)
                output = output.unsqueeze(0).expand(1, -1, -1)

                # Statistics
                if int(label[0][0]) in torch.topk(output, k).indices.tolist():
                    pred_correct += 1

                pred_all += 1

    return pred_correct, pred_all, (pred_correct / pred_all)


def get_sequence_list(num):
    if num == 0:
        return [0]

    result, i = [1], 2
    while sum(result) != num:
        if sum(result) + i > num:
            for j in range(i - 1, 0, -1):
                if sum(result) + j <= num:
                    result.append(j)
        else:
            result.append(i)
        i += 1

    return sorted(result, reverse=True)
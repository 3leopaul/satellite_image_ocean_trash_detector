import torch

def compute_confusion_matrix(y_true, y_pred, num_classes):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return cm


def compute_iou(cm, class_id):
    tp = cm[class_id, class_id]
    fp = cm[:, class_id].sum() - tp
    fn = cm[class_id, :].sum() - tp

    denom = tp + fp + fn
    if denom == 0:
        return float('nan')

    return tp / denom


def compute_f1(cm, class_id):
    tp = cm[class_id, class_id]
    fp = cm[:, class_id].sum() - tp
    fn = cm[class_id, :].sum() - tp

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return 2 * precision * recall / (precision + recall + 1e-8)

def evaluate_segmentation(y_true, y_pred, num_classes, debris_class_id):
    """
    Calcule la matrice de confusion + IoU/F1 globaux et pour la classe 'debris'.
    """
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    
    iou_per_class = [compute_iou(cm, c) for c in range(num_classes)]
    f1_per_class = [compute_f1(cm, c) for c in range(num_classes)]

    iou_debris = iou_per_class[debris_class_id]
    f1_debris = f1_per_class[debris_class_id]

    return {
        "confusion_matrix": cm,
        "iou_per_class": iou_per_class,
        "f1_per_class": f1_per_class,
        "iou_debris": iou_debris,
        "f1_debris": f1_debris,
    }



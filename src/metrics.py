class Metrics:
    @staticmethod
    def mean_iou(preds, labels, threshold=0.5, epsilon=1e-6):
        preds = (preds > threshold).float()
        labels = labels.float()
        intersection = (preds * labels).sum((1, 2))
        union = (preds + labels).clamp(0, 1).sum((1, 2))
        iou = (intersection + epsilon) / (union + epsilon)
        return iou.mean().item()

    @staticmethod
    def dice_coefficient(preds, labels, threshold=0.5, epsilon=1e-6):
        preds = (preds > threshold).float()
        labels = labels.float()
        intersection = (preds * labels).sum((1, 2))
        dice = (2 * intersection + epsilon) / (preds.sum((1, 2)) + labels.sum((1, 2)) + epsilon)
        return dice.mean().item()

    @staticmethod
    def accuracy(preds, labels, threshold=0.5):
        preds = (preds > threshold).float()
        labels = labels.float()
        correct = (preds == labels).float().sum((1, 2))
        total = labels.size(1) * labels.size(2)
        return (correct / total).mean().item()


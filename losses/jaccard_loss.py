import torch
import torch.nn as nn
import torch.nn.functional as F

class JaccardLoss(torch.nn.Module):
    def __init__(self, num_classes=2, from_logits: bool=True, smooth=1e-5, reduce: bool=True):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth
        self.from_logits = from_logits
        self.reduce = reduce
        self.num_classes = num_classes

    def forward(self, targets, inputs):

        if self.from_logits:
            inputs = torch.nn.Softmax(dim=1)(inputs)

        B = inputs.shape[0]
        
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).float()
        targets_one_hot = torch.permute(targets_one_hot, (0, 4, 1, 2, 3))

        intersection = targets_one_hot * inputs

        # Reduce the last 3 dimension -> intersection shape = [batch_size, num_classes]
        intersection_flat = intersection.view(B, self.num_classes, -1).sum(dim=2)

        union = inputs + targets_one_hot - intersection

        # Reduce the last 3 dimension -> union shape = [batch_size, num_classes]
        union_flat = union.view(B, self.num_classes, -1).sum(dim=2)

        jaccard = (intersection_flat + self.smooth) / (union_flat + self.smooth)

        if self.reduce:
            jaccard = torch.mean(jaccard)
        
        return 1 - jaccard
        
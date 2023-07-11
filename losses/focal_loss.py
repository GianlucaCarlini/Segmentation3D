import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification

    Parameters
    ----------
    gamma : str, optional
        Focal parameter, by default 2
    num_classes : int, optional
        Number of classes, by default 2
    from_logits : bool, optional
        Wheter if the predicted tensor is a logits tensor (True) or a probability distribution (False), by default True
    reduce : bool, optional
        Wheter if compute the mean of the final tensor or not, by default True
        
    Example
    -------
        >>> import torch
        >>> import torch.nn as nn
        >>> import torch.nn.functional as F
        >>> target = torch.rand((1, 128, 128, 128, 7))
        >>> target = torch.argmax(target, dim=-1)
        >>> input = torch.rand((1, 128, 128, 128, 7))
        >>> input = F.softmax(input, dim=-1)
        >>> loss = FocalLoss(num_clasees=7)
        >>> res = loss(input, target)
    """
    def __init__(self, gamma: int=2, num_classes: int=2, from_logits: bool=True, reduce: bool=True):
        
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.from_logits = from_logits
        self.reduce = reduce
        self.num_clasees = num_classes

    def forward(self, targets, inputs):

        targets = F.one_hot(targets.long(), num_classes=self.num_clasees).float()

        inputs = torch.permute(inputs, (0, 2, 3, 4, 1))

        if self.from_logits:
            probs = F.softmax(inputs, dim=-1)
        else:
            probs = inputs

        cross_entropy = -targets * torch.log(probs + 1e-6)

        focal_loss = torch.pow(1 - probs, self.gamma) * cross_entropy

        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss

if __name__ == "__main__":

    import doctest

    doctest.testmod(verbose=True)



    
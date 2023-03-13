import torch
from torch.autograd import Variable

class MultiCEFocalLoss_New(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, lb_smooth=0,
                 reduction='mean'):
        super(MultiCEFocalLoss_New, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.class_num = class_num

    def forward(self, predict, target):
        pt = torch.softmax(predict, dim=-1).view(-1, self.class_num)
        class_mask = torch.nn.functional.one_hot(
            target, self.class_num).view(-1, self.class_num)
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)].view(-1, 1)
        alpha = alpha.to(predict.device)

        positive_class_mask_indices = torch.nonzero(
            class_mask[:, 2] == 0).squeeze()
        negative_class_mask_indices = torch.nonzero(
            class_mask[:, 2] == 1).squeeze()
        positive_pt = pt[positive_class_mask_indices]
        negative_pt = pt[negative_class_mask_indices]
        positive_class_mask = class_mask[positive_class_mask_indices]
        negative_class_mask = class_mask[negative_class_mask_indices]
        positive_alpha = alpha[positive_class_mask_indices]
        negative_alpha = alpha[negative_class_mask_indices]

        # p_num = torch.sum(class_mask[:, :-1]).item()
        # n_num = torch.sum(class_mask[:, -1]).item()
        # if torch.sum(class_mask[:, -1]) == class_mask.shape[0]:
        #     return 0
        # negative_alpha = 1 / math.log2(n_num / p_num)
        # positive_alpha = 1 - negative_alpha

        positive_probs = (positive_pt * positive_class_mask).sum(-1).view(-1, 1)
        positive_log_p = positive_probs.log()
        positive_loss = -positive_alpha * torch.pow(
            (1 - positive_probs), self.gamma) * positive_log_p

        negative_probs = (negative_pt * negative_class_mask).sum(-1).view(-1, 1)
        negative_log_p = negative_probs.log()
        negative_loss = -negative_alpha * torch.pow(
            torch.clamp(1 - self.lb_smooth - negative_probs, min=0),
            self.gamma) * negative_log_p

        loss = torch.cat((positive_loss, negative_loss))

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


def _focal_loss(output, label, gamma, alpha, lb_smooth):
    output = output.contiguous().view(-1)
    label = label.view(-1)
    mask_class = (label > 0).float()

    # p_num = torch.sum(label > 0).item()
    # n_num = torch.sum(label == 0).item()
    # if p_num == 0:
    #     return 0
    # c_0 = 1 / math.log2(n_num / p_num)
    # c_1 = 1 - c_0

    c_1 = alpha
    c_0 = 1 - c_1
    loss = ((c_1 * torch.abs(label - output)**gamma * mask_class
            * torch.log(output + 0.00001))
            + (c_0 * torch.abs(label + lb_smooth - output)**gamma
            * (1.0 - mask_class)
            * torch.log(1.0 - output + 0.00001)))
    loss = -torch.mean(loss)
    return loss


def _probability_loss(output, score, gamma, alpha, lb_smooth):
    output = torch.sigmoid(output)
    loss = _focal_loss(output, score, gamma, alpha, lb_smooth)
    return loss
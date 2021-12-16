import torch
import torch.nn.functional as F

def entropy(p, prob=True, mean=True):
    if prob:
        p = F.softmax(p)
    en = -torch.sum(p * torch.log(p+1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en

#TODO: look at here!
def ova_loss(out_open, label):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2

    # 36, 2, 1000
    out_open = F.softmax(out_open, 1)
    # 36, 1000
    label_p = torch.zeros((out_open.size(0),
                           out_open.size(2))).long().cuda()
    # label_range = torch.range(0, out_open.size(0) - 1).long()
    # 0, 1, 2, ..., 35
    label_range = torch.arange(0, out_open.size(0)).long()
    label_p[label_range, label] = 1 # 36,1000
    label_n = 1 - label_p # 36,1000

    open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :] + 1e-8) * label_p, 1))
    # hardest negative
    # TODO: add more negatives for ImageNet dataset.
    open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] + 1e-8) * label_n, 1)[0])

    return open_loss_pos, open_loss_neg

def adv_loss(d_source, d_target, openc_target):
    # d_source, d_target 36,2
    # openc_target 36,2,1000 :,0,: -> outer :,1,: ->inter
    assert len(d_source.size())==2 and len(d_target.size())==2 and d_source.size(1)==2 and d_target.size(1)==2
    assert len(openc_target.size())==3
    assert openc_target.size(1) == 2

    d_source = F.softmax(d_source, 1) # 0->source, 1->target
    d_target = F.softmax(d_target, 1)
    openc_target = F.softmax(openc_target, 1)

    classes = openc_target.size(2)
    # w_t = torch.sum(torch.max(openc_target, dim=1)[1],dim=1) / classes # 36,1
    w_t = torch.true_divide(torch.sum(torch.max(openc_target, dim=1)[1],dim=1), classes)

    adv_loss = torch.mean(-torch.log(d_source[:,0] + 1e-8) - w_t*torch.log(d_target[:,1]+1e-8))
    return adv_loss

def open_entropy(out_open):
    assert len(out_open.size()) == 3
    assert out_open.size(1) == 2
    out_open = F.softmax(out_open, 1) # 36, 2, 1000
    ent_open = torch.mean(torch.mean(torch.sum(-out_open * torch.log(out_open + 1e-8), 1), 1))
    return ent_open
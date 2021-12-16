import torch
from torch import nn
import torchvision.transforms as transforms
import torch.optim as optim
from apex import amp, optimizers
from data_loader.get_loader import get_loader, get_loader_label
from .utils import get_model_mme
from models.basenet import ResClassifier_MME
from pytorch_revgrad import RevGrad

def get_dataloaders(kwargs):
    source_data = kwargs["source_data"]
    target_data = kwargs["target_data"]
    evaluation_data = kwargs["evaluation_data"] # target = evaluation
    conf = kwargs["conf"]
    # There is no val in OVANet
    val_data = None
    if "val" in kwargs:
        val = kwargs["val"]
        if val:
            val_data = kwargs["val_data"]
    else:
        val = False

    data_transforms = {
        source_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "eval": transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    return get_loader(source_data, target_data, evaluation_data,
                      data_transforms,
                      batch_size=conf.data.dataloader.batch_size,
                      return_id=True,
                      balanced=conf.data.dataloader.class_balance,
                      val=val, val_data=val_data)



def get_dataloaders_label(source_data, target_data, target_data_label, evaluation_data, conf):

    data_transforms = {
        source_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        evaluation_data: transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return get_loader_label(source_data, target_data, target_data_label,
                            evaluation_data, data_transforms,
                            batch_size=conf.data.dataloader.batch_size,
                            return_id=True,
                            balanced=conf.data.dataloader.class_balance)

def get_models(kwargs):
    net = kwargs["network"]
    num_class = kwargs["num_class"]
    conf = kwargs["conf"]
    # G, dim = get_model_mme(net, num_class=num_class)
    # TAG: 
    G, dim = get_model_mme(net, num_class=num_class, top=True)

    # One vs all Open Set classifier
    C2 = ResClassifier_MME(num_classes=2 * num_class,
                           norm=False, input_size=dim)

    # TODO: Discriminator  dim -> 2  [source, target]
    D_adversarial = nn.Sequential(RevGrad(), nn.Linear(num_class, 2))

    # Close set classifier
    C1 = ResClassifier_MME(num_classes=num_class,
                           norm=False, input_size=dim)
    device = torch.device("cuda")
    G.to(device)
    C1.to(device)
    C2.to(device)
    D_adversarial.to(device)

    params = []
    if net == "vgg16":
        for key, value in dict(G.named_parameters()).items():
            if 'classifier' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]

    else:
        for key, value in dict(G.named_parameters()).items():

            if 'bias' in key:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
            else:
                params += [{'params': [value], 'lr': conf.train.multi,
                            'weight_decay': conf.train.weight_decay}]
    opt_g = optim.SGD(params, momentum=conf.train.sgd_momentum,
                      weight_decay=0.0005, nesterov=True)
    opt_c = optim.SGD(list(C1.parameters()) + list(C2.parameters()), lr=1.0,
                       momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                       nesterov=True)
    # TODO: AdamW
    opt_d = optim.SGD(D_adversarial.parameters(), lr=1e-3,
                       momentum=conf.train.sgd_momentum, weight_decay=0.0005,
                       nesterov=True)

    [G, C1, C2, D_adversarial], [opt_g, opt_c, opt_d] = amp.initialize([G, C1, C2, D_adversarial],
                                                  [opt_g, opt_c, opt_d],
                                                  opt_level="O0")
    G = nn.DataParallel(G)
    C1 = nn.DataParallel(C1)
    C2 = nn.DataParallel(C2)
    D_adversarial = nn.DataParallel(D_adversarial)

    param_lr_g = []
    for param_group in opt_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_c = []
    for param_group in opt_c.param_groups:
        param_lr_c.append(param_group["lr"])
    param_lr_d = []
    for param_group in opt_d.param_groups:
        param_lr_d.append(param_group["lr"])

    return G, C1, C2, D_adversarial, opt_g, opt_c, opt_d, param_lr_g, param_lr_c, param_lr_d
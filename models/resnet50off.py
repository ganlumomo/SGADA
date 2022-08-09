import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torchvision import models


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feat):
        input_nc = feat.shape[1]
        self.mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
        print('p: ', count_parameters(self.mlp))
        if len(self.gpu_ids) > 0:
            self.mlp.cuda()
        init_net(self.mlp, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feat):
        feat = feat.flatten(1, 3)
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feat)
        if self.use_mlp:
            feat = self.mlp(feat)
            feat = self.l2norm(feat)
        return feat


class ResNet50Mod(nn.Module):
    def __init__(self):
        super(ResNet50Mod, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.freezed_rn50 = nn.Sequential(
            model_resnet50.conv1,
            model_resnet50.bn1,
            model_resnet50.relu,
            model_resnet50.maxpool,
            model_resnet50.layer1,
            model_resnet50.layer2,
            model_resnet50.layer3,
        )
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features
    
    def forward(self, x):
        x = self.freezed_rn50(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, h=256, dropout=0.5, srcTrain=False):
        super(Encoder, self).__init__()
        
        rnMod = ResNet50Mod()
        self.feature_extractor = rnMod.freezed_rn50
        self.layer4 = rnMod.layer4
        self.avgpool = rnMod.avgpool
        if srcTrain:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x, encoder_only=False):
        x = x.expand(x.data.shape[0], 3, 224, 224)
        x = self.feature_extractor(x)
        x = self.layer4(x)
        feat = self.avgpool(x)
        x = feat.view(feat.size(0), -1)
        if encoder_only:
            return x, feat
        return x


class Classifier(nn.Module):
    def __init__(self, n_classes, dropout=0.5):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = self.l1(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels=3, n_classes=31, target=False, srcTrain=False):
        super(CNN, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, srcTrain=srcTrain)
        self.classifier = Classifier(n_classes)
        if target:
            for param in self.classifier.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, h=500, args=None):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(2048, h)
        self.l2 = nn.Linear(h, h)
        self.l3 = nn.Linear(h, 2)
        self.l4 = nn.LogSoftmax(dim=1)
        self.slope = args.slope

    def forward(self, x):
        x = F.leaky_relu(self.l1(x), self.slope)
        x = F.leaky_relu(self.l2(x), self.slope)
        x = self.l3(x)
        x = self.l4(x)
        return x

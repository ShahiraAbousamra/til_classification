import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
import collections
from distutils.util import strtobool;
#from torchvision.models.utils import load_state_dict_from_url
#from torch.hub import load_state_dict_from_url
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

#from sa_net_arch_utilities_pytorch import CNNArchUtilsPyTorch;


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class Upsample(nn.Module):
    def __init__(self,  scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

class TILClassifier(nn.Module):
    def __init__(self, load_weights=False, kwargs=None):
        super(TILClassifier,self).__init__()

        # predefined list of arguments
        args = {'input_img_width':-1, 'input_img_height':-1, 'pretrained':'False'
            , 'conv_init': 'he'
            , 'use_softmax':'False', 'use_relu':'False', 'use_tanh':'False'
            ,'n_layers_per_path':4, 'n_conv_blocks_in_start': 64, 'block_size':3, 'pool_size':2
            , 'dropout_keep_prob' : 1.0, 'initial_pad':0, 'interpolate':'False', 'n_classes':1, 'n_channels':3
 
        };

        if(not(kwargs is None)):
            args.update(kwargs);

        # 'conv_init': 'uniform', 'normal', 'xavier_uniform', 'xavier_normal', 'he'

        # read extra argument
        #self.n_layers_per_path = int(args['n_layers_per_path']); # n_layers_per_path in contracting path + n_layers_per_path in expanding path + 1 bottleneck layer
        #self.n_conv_blocks_in_start = int(args['n_conv_blocks_in_start']);
        self.input_img_width = int(args['input_img_width']);
        self.input_img_height = int(args['input_img_height']);
        self.n_channels = int(args['n_channels']);
        self.n_classes = int(args['n_classes']);
        #dropout = args['dropout'];
        self.pretrained = bool(strtobool(args['pretrained']));
        #self.stain_init_name = str(args['stain_init_name']);
        self.conv_init = str(args['conv_init']).lower();
        self.use_softmax = bool(strtobool(args['use_softmax']));
        self.use_relu = bool(strtobool(args['use_relu']));
        self.use_tanh = bool(strtobool(args['use_tanh']));
    
        self.n_layers_per_path = int(args['n_layers_per_path']);
        self.n_conv_blocks_in_start = int(args['n_conv_blocks_in_start']);
        self.block_size = int(args['block_size']);
        self.pool_size = int(args['pool_size']);
        self.dropout_keep_prob = float(args['dropout_keep_prob'])
        self.initial_pad = int(args['initial_pad']);
        self.interpolate = bool(strtobool(args['interpolate']));

        print('self.initial_pad',self.initial_pad)

        n_blocks = self.n_conv_blocks_in_start;
        n_blocks_prev = self.n_channels;


        #self.model = models.resnet34(pretrained = True)
        self.model = resnet34(pretrained = True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
        
        self.sig_layer = torch.nn.Sigmoid()

        self._initialize_weights()

        self.zero_grad() ;


    def forward(self,x):

        x = self.model(x)
        #x = self.sig_layer(x)


        return x;

    def _initialize_weights(self):
        BIAS_INIT = 0.1;
        #for l in self.encoder:
        #    for layer in l:
        #        if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
        #            if(self.conv_init == 'normal'):
        #                torch.nn.init.normal_(layer.weight) ;
        #            elif(self.conv_init == 'xavier_uniform'):
        #                torch.nn.init.xavier_uniform_(layer.weight) ;
        #            elif(self.conv_init == 'xavier_normal'):
        #                torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
        #            elif(self.conv_init == 'he'):
        #                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
        #                #layer.bias.data.fill_(BIAS_INIT);

        #for layer in self.bottleneck:
        #    if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
        #        if(self.conv_init == 'normal'):
        #            torch.nn.init.normal_(layer.weight) ;
        #        elif(self.conv_init == 'xavier_uniform'):
        #            torch.nn.init.xavier_uniform_(layer.weight) ;
        #        elif(self.conv_init == 'xavier_normal'):
        #            torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
        #        elif(self.conv_init == 'he'):
        #            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
        #            #layer.bias.data.fill_(BIAS_INIT);


        #for l in self.decoder:
        #    for layer in l:
        #        if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
        #            if(self.conv_init == 'normal'):
        #                torch.nn.init.normal_(layer.weight) ;
        #            elif(self.conv_init == 'xavier_uniform'):
        #                torch.nn.init.xavier_uniform_(layer.weight) ;
        #            elif(self.conv_init == 'xavier_normal'):
        #                torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
        #            elif(self.conv_init == 'he'):
        #                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
        #                #layer.bias.data.fill_(BIAS_INIT);

        #for layer in self.final_layer:
        #    if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
        #        if(self.conv_init == 'normal'):
        #            torch.nn.init.normal_(layer.weight) ;
        #        elif(self.conv_init == 'xavier_uniform'):
        #            torch.nn.init.xavier_uniform_(layer.weight) ;
        #        elif(self.conv_init == 'xavier_normal'):
        #            torch.nn.init.xavier_normal_(layer.weight, gain=10) ;
        #        elif(self.conv_init == 'he'):
        #            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') ; 
        #            #layer.bias.data.fill_(BIAS_INIT);

        #vgg_model = models.vgg16(pretrained = True)
        #fsd=collections.OrderedDict()
        #i = 0
        ##for l in self.encoder:
        ##    for layer in l:
        ##        if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
        ##            temp_key=list(self.encoder.state_dict().items())[i][0]
        ##            print('temp_key', temp_key)
        ##            fsd[temp_key]=list(vgg_model.state_dict().items())[i][1]
        ##            i += 1
        #for m in self.encoder.state_dict().items():
        #    temp_key=m[0]
        #    print('temp_key', temp_key)
        #    print('vgg_key', list(vgg_model.state_dict().items())[i][0])
        #    fsd[temp_key]=list(vgg_model.state_dict().items())[i][1]
        #    i += 1
        #self.encoder.load_state_dict(fsd)

        #fsd=collections.OrderedDict()
        ##for layer in self.bottleneck:
        ##    if(isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Conv2d)):
        ##        temp_key=list(self.bottleneck.state_dict().items())[i][0]
        ##        fsd[temp_key]=list(vgg_model.state_dict().items())[i][1]
        ##        i += 1
        #for m in self.bottleneck.state_dict().items():
        #    temp_key=m[0]
        #    print('temp_key', temp_key)
        #    print('vgg_key', list(vgg_model.state_dict().items())[i][0])
        #    fsd[temp_key]=list(vgg_model.state_dict().items())[i][1]
        #    i += 1
        #self.bottleneck.load_state_dict(fsd)

        ##del vgg_model

def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False, deconv=None, pad_list=None):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    #for v in cfg:
    for i in range(len(cfg)):
        v=cfg[i]
        print('in_channels=',in_channels)
        print('v=',v)
        if(not (deconv is None)):
            print('deconv[i]=',deconv[i])
        if(pad_list is None):
            padding = d_rate;
        else:
            padding = pad_list[i];
        print('padding =', padding);
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if(deconv is None or deconv[i] == False):
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=padding, dilation = d_rate)
            else:
                conv2d = nn.ConvTranspose2d(in_channels, v, stride=2, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



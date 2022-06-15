import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


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
class BasicBlock(nn.Module):
        
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsaple=None):
        super(BasicBlock, self).__init__()
       
        norm_layer = nn.BatchNorm2d
        

        # 下面定义BasicBlock中的各个层
        #conv 3x3
        
        self.conv1 = nn.Conv2d(in_channels= inplanes,out_channels=planes,kernel_size=3 ,stride= stride, padding=1)
        self.bn1 = norm_layer(planes)
        # inplace为True表示进行原地操作，一般默认为False，表示新建一个变量存储操作
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels= planes,out_channels=planes,kernel_size=3 ,stride= 1, padding=1)
        self.bn2 = norm_layer(planes)
        self.dowansample = downsaple


    # 定义前向传播函数将前面定义的各层连接起来
    def forward(self, x):
        identity = x  # 残差块需要保留原始输入
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.dowansample is not None: # 保证原始输入与卷积后的输出层叠加时维度相同
            identity = self.dowansample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.

    expansion = 4  

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        
        self.conv1 = nn.Conv2d(in_channels= inplanes, out_channels=planes, kernel_size=1 ,stride= 1)  #conv 1x1
        self.bn1 = norm_layer(planes)  # BN层
        
        self.conv2 = nn.Conv2d(in_channels= planes, out_channels=planes, kernel_size=3 ,stride= stride, padding=1)  #conv 3x3
        self.bn2 = norm_layer(planes)  # BN层
        self.conv3 = nn.Conv2d(in_channels= planes, out_channels=planes * self.expansion, kernel_size=1 ,stride= 1)  #conv 1x1
        self.bn3 = norm_layer(planes * self.expansion)  # BN层
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 下采样层
 
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

    def __init__(self, block, layers, num_classes=2, zero_init_residual= True):
        """ResNet

        Args:
            block : 基本模块类BasicBlock(18,34)或者Bottleneck(50,101,152)
            layers (list): e.g. [2, 2, 2, 2]是每个layer的重复次数 
            num_classes (int, optional): num_classes类别数
            zero_init_residual (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_
        """
        super(ResNet, self).__init__()
        
        norm_layer = nn.BatchNorm2d  
        self._norm_layer = norm_layer

        self.inplanes = 64  
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)   
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)    
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  
        self.layer1 = self._make_layer(block, 64, layers[0])  
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # 自适应平均池化层，输出大小为（1,1）
        self.fc_new = nn.Linear(512 * block.expansion, num_classes)  

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

    def _make_layer(self, block, planes, blocks, stride=1, ):
        norm_layer = self._norm_layer  
        downsample = None  
        # stride != 1，or self.inplances != planes * block.expansion ，则给下采样层赋值
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                #conv1x1 for downsample
                nn.Conv2d(in_channels= self.inplanes, out_channels= planes * block.expansion,kernel_size=1,stride= stride),
                norm_layer(planes * block.expansion),
            ) 
            

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))  
        self.inplanes = planes * block.expansion  #update self.inplanes
        for _ in range(1, blocks):  
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # torch.Size([1, 3, 224, 224])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # torch.Size([1, 64, 112, 112])
        x = self.maxpool(x)  # torch.Size([1, 64, 56, 56])

        x = self.layer1(x)  # torch.Size([1, 64, 56, 56])
        x = self.layer2(x)  # torch.Size([1, 128, 28, 28])
        x = self.layer3(x)  # torch.Size([1, 128, 14, 14])
        x = self.layer4(x)  # torch.Size([1, 512, 7, 7])

        x = self.avgpool(x)  # torch.Size([1, 512, 1, 1])
        x = torch.flatten(x, 1)  # torch.Size([1, 512])
        x = self.fc_new(x)  # torch.Size([1, num_classes])

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)  
    if pretrained:  
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model_dict = model.state_dict()    
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

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





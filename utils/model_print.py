import sys
sys.path.append('')

from model.ResNet import resnet101
model = resnet101()
print(f"Model structure: {model}\n\n")

from thop import profile,clever_format
import torch
input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input,))

macs, params = clever_format([macs, params], "%.3f")
print('MACs:',macs,'Params:',params)

from torchinfo import summary


summary(model, input_size=(1, 3, 224, 224))
print(model)
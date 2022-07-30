import argparse
from utils.param_FLOPs_counter import model_info
from utils.Logger import Logger


from model.ResNet import resnet101
model = resnet101()
from torchvision.models import resnet50
model = resnet50()
parser = argparse.ArgumentParser(description='RefineNet Training With PyTorch')
parser.add_argument('--model_name', default='ResNet101', type=str)
args = parser.parse_args()
log = Logger('log/'+ args.model_name+'_INFO.log',level='debug')
model_info(model,log,img_size=[224,224])


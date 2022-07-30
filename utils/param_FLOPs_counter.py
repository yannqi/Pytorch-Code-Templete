import sys
import thop
from copy import deepcopy
import torch

def model_info(model, log, verbose=False, img_size=[224,224]):
    """Model information. 

    Args:
        model (_type_): _description_
        verbose (bool, optional): _description_. Defaults to False.
        img_size ( optional): img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    """
    #logger
   
    
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    total = sum([param.nelement() for param in model.parameters()]) # Total number
    log.logger.info(' Number of params: %.2fM' % (total / 1e6))
    if verbose:
        log.logger.info(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}") # Leave space for more info。
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            log.logger.info('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    
    from thop import profile,clever_format
    input = torch.randn(1, 3, img_size[0],img_size[1])
    macs, params = profile(model, inputs=(input, ))   # stride GFLOPs
    log.logger.info(' Number of params: %.2fM' % (total / 1e6))
    log.logger.info(' FLOPs:%.2f, Params: %.2f' ,macs,params )
    macs, params = clever_format([macs, params], "%.3f")
    print(f'FLOPs:',macs,'Params:',params)
    
    
    # name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(model, 'yaml_file') else 'Model'
    log.logger.info(f"Model summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients")

if __name__ == '__main__':
    sys.path.append('')
    from model.ResNet import resnet101
    model = resnet101()
    print(f"Model structure: {model}\n\n")
    

    
    from thop import profile,clever_format
    import torch
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input,))
    print('MACs:',macs,'Params:',params)
    macs, params = clever_format([macs, params], "%.3f")
    print('MACs:',macs,'Params:',params)
    from torchinfo import summary
    summary(model, input_size=(1, 3, 224, 224))
    
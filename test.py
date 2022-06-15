import os
import argparse
import yaml
import tqdm
import pandas as pd

import torch 
import torch.nn as nn

from torch.utils import data

from model.ResNet import resnet101

from utils.logging import Logger

from data.CatVsDog import CatVsDogDataset
from data.utils.data_aug import train_transform,valid_transform
#https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview/evaluation 
def test(args,cfg):
    #logger
    log = Logger(cfg['LOG_DIR']+'/'+args.model_name+'.log',level='debug')

    
    
    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    
    annotations_file = cfg['DATASET']['ANNOTATIONS_FILE']
    img_dir = cfg['DATASET']['IMG_DIR']
    
    
    # Pre dataset
    test_dataset = CatVsDogDataset(annotations_file,img_dir,is_train=False,transform=valid_transform)                 
    
    test_dataloader = data.DataLoader(test_dataset,batch_size=cfg['TEST']['BATCH_SIZE'],
                                       num_workers=cfg['TRAIN']['NUM_WORKERS'], shuffle=False, collate_fn=None, pin_memory=True)
    
    
    
    # Load model
    net = resnet101(pretrained=False).to(args.device)
    # Load checkpoint
    if args.use_ckpt:
        net.load_state_dict(torch.load(cfg['CHECKPOINT_DIR']+'/' + args.model_name+'.pth'))    

    csv_ids = []
    csv_predict_labels = []
    net.eval()
    
    for i, targets in enumerate(tqdm.tqdm(test_dataloader,desc=f"Test")):
        imgs = targets[0].to(args.device)
        img_ids = targets[2]
        predicts = net(imgs)
        #predict_labels = torch.argmax(predicts,dim=1)
        predict_labels = torch.max(predicts,dim=1).values
        csv_ids += img_ids[0]
        csv_predict_labels+=(predict_labels.cpu().tolist())




    #字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'id':csv_ids,'label':csv_predict_labels})

    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(cfg['OUTPUT_DIR']+"/test.csv",index=False,sep=',')

    
            
            
def main():
    parser = argparse.ArgumentParser(description='RefineNet Training With PyTorch')
    parser.add_argument(
        "--config_file",
        default="config/ResNet101.yaml",
        metavar="FILE",
        help="path to cfg file",
        type=str,)
    parser.add_argument('--model_name', default='ResNet101', type=str)
    parser.add_argument('--use_tensorboard', default=False, type=bool)
    parser.add_argument('--num_gpu', default=3, type=int,
                    help='Use which gpu to train model, Default is 0.')
    parser.add_argument("--use_ckpt", default=True, help="Whether to use pretrain model")
    args = parser.parse_args()
    

    cfg_path = open(args.config_file)
    cfg = yaml.full_load(cfg_path)
    if cfg['OUTPUT_DIR'] and not os.path.exists(cfg['OUTPUT_DIR']) :
        os.makedirs(cfg['OUTPUT_DIR'])
    if cfg['CHECKPOINT_DIR'] and not os.path.exists(cfg['CHECKPOINT_DIR']) :
        os.makedirs(cfg['CHECKPOINT_DIR'])
    if cfg['LOG_DIR'] and not os.path.exists(cfg['LOG_DIR']) :
        os.makedirs(cfg['LOG_DIR'])
    
    
    # Use Gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.num_gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    args.device = device
        
    test(args,cfg)    
if __name__ == "__main__":
    main()
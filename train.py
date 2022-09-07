import os
import argparse
import yaml
import tqdm
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils import data

from model.ResNet import resnet101
from module.utils.accuracy_compute import compute_accuracy
from utils.Logger import Logger

from data.CatVsDog import CatVsDogDataset
from data.utils.data_aug import train_transform,valid_transform

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
    parser.add_argument("--use_ckpt", default=False, help="Whether to use pretrain model")
    parser.add_argument("--save_data", default=True, help="Whether to save train data for plot")
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
   
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    args = parser.parse_args()
    

    cfg_path = open(args.config_file)
    # 引入EasyDict 可以让你像访问属性一样访问dict里的变量。
    from easydict import EasyDict as edict
    cfg = yaml.full_load(cfg_path)
    cfg = edict(cfg) # 将普通的字典传入到edict()
    #     os.makedirs(cfg.PLOT_DIR)
    if cfg.OUTPUT_DIR and not os.path.exists(cfg.OUTPUT_DIR) :
        os.makedirs(cfg.OUTPUT_DIR)
    if cfg.CHECKPOINT_DIR and not os.path.exists(cfg.CHECKPOINT_DIR) :
        os.makedirs(cfg.CHECKPOINT_DIR)
    if cfg.LOG_DIR and not os.path.exists(cfg.LOG_DIR) :
        os.makedirs(cfg.LOG_DIR)
    if cfg.PLOT_DIR and not os.path.exists(cfg.PLOT_DIR) :
        os.makedirs(cfg.PLOT_DIR)

    
    #Random seed
    np.random.seed(cfg.seed)
    cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(cfg.seed)
    
    
    # Use Gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.num_gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    args.device = device
         
    #logger
    log = Logger(cfg.LOG_DIR+'/'+args.model_name+'.log',level='debug')

    #Initial Logging
    log.logger.info('gpu device = %s' % args.gpu_id)
    log.logger.info("args = %s", args)
    log.logger.info("cfgs = %s", cfg)
    
    #Save data for plot
    if args.save_data == True:
        save_loss = []
        save_acc = []
    
    
    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    
    annotations_file = cfg.DATASET.ANNOTATIONS_FILE
    img_dir = cfg.DATASET.IMG_DIR
    
    
    # Pre dataset
    train_dataset = CatVsDogDataset(annotations_file,img_dir,is_train=True,transform=train_transform)                 
    test_dataset = CatVsDogDataset(annotations_file,img_dir,is_train=False,transform=valid_transform)  
    
    train_dataloader = data.DataLoader(train_dataset,batch_size=cfg.TRAIN.BATCH_SIZE,
                                       num_workers=cfg.TRAIN.NUM_WORKERS, shuffle=True, collate_fn=None, pin_memory=True)
    
    test_dataloader = data.DataLoader(test_dataset,batch_size=cfg.TEST.BATCH_SIZE,
                                        num_workers=cfg.TEST.NUM_WORKERS, shuffle=False, collate_fn=None, pin_memory=True)
    
    
    
    # Load model
    #net = resnet101(pretrained=True).to(args.device)
    net = resnet101().to(args.device)
    # Load checkpoint
    if args.use_ckpt:
        net.load_state_dict(torch.load(cfg.CHECKPOINT_DIR+'/' + args.model_name+'.pth'))    
    
    
    # Load loss function
    min_loss = 1
    criterion = nn.CrossEntropyLoss()

    # Load Optimizer
    optimizer =  torch.optim.Adam(net.parameters(), lr=cfg.SOLVER.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)



    best_test_accuracy = compute_accuracy(args,test_dataloader,net)
    for epoch in range(cfg.TRAIN.EPOCHS):
        net.train()
        running_loss = 0
        for i, targets in enumerate(tqdm.tqdm(train_dataloader,desc=f"Training Epoch {epoch}")):
            imgs = targets[0].to(args.device)
            labels = targets[1].to(args.device)
            optimizer.zero_grad()
            predicts = net(imgs)
            
            loss = criterion(predicts, labels)
            running_loss += loss
            loss.backward()
            optimizer.step()
            
            
        mean_loss = running_loss/len(train_dataloader)
        log.logger.info('Epoch:%s Meanloss: %s',epoch,mean_loss.item())
        
    #Save model   
        
        test_accuracy = compute_accuracy(args,test_dataloader,net)
        log.logger.info('test_accuracy is: %s',test_accuracy)
        scheduler.step(mean_loss)
        
        
        if  test_accuracy > best_test_accuracy :
            best_epoch = epoch
            best_test_accuracy = test_accuracy
            print('Best acc is:',best_test_accuracy)
            save_path = cfg.CHECKPOINT_DIR+'/'+args.model_name+'.pth'
            torch.save(net.state_dict(), save_path)
        if args.save_data == True:
            save_loss.append(mean_loss.item())
            save_acc.append(test_accuracy)
    log.logger.info('Best acc is: %s \n,Best epoch is: %s',best_test_accuracy,best_epoch)
    if args.save_data == True:
        #字典中的key值即为csv中列名
        dataframe = pd.DataFrame({'Epoch_loss':save_loss,'val_acc':save_acc})
        #将DataFrame存储为csv,index表示是否显示行名，default=True
        dataframe.to_csv("output/plot_data/"+args.model_name+".csv",index=False,sep=',')


    cfg_path = open(args.config_file)
    # 引入EasyDict 可以让你像访问属性一样访问dict里的变量。
    from easydict import EasyDict as edict
    cfg = yaml.full_load(cfg_path)
    cfg = edict(cfg) # 将普通的字典传入到edict()
    # if cfg['OUTPUT_DIR'] and not os.path.exists(cfg['OUTPUT_DIR']) :
    #     os.makedirs(cfg['OUTPUT_DIR'])
    # if cfg['CHECKPOINT_DIR'] and not os.path.exists(cfg['CHECKPOINT_DIR']) :
    #     os.makedirs(cfg['CHECKPOINT_DIR'])
    # if cfg['LOG_DIR'] and not os.path.exists(cfg['LOG_DIR']) :
    #     os.makedirs(cfg['LOG_DIR'])
    # if cfg['PLOT_DIR'] and not os.path.exists(cfg['PLOT_DIR']) :
    #     os.makedirs(cfg['PLOT_DIR'])
    if cfg.OUTPUT_DIR and not os.path.exists(cfg.OUTPUT_DIR) :
        os.makedirs(cfg.OUTPUT_DIR)
    if cfg.CHECKPOINT_DIR and not os.path.exists(cfg.CHECKPOINT_DIR) :
        os.makedirs(cfg.CHECKPOINT_DIR)
    if cfg.LOG_DIR and not os.path.exists(cfg.LOG_DIR) :
        os.makedirs(cfg.LOG_DIR)
    if cfg.PLOT_DIR and not os.path.exists(cfg.PLOT_DIR) :
        os.makedirs(cfg.PLOT_DIR)

def train():
    #需要train的函数另外定义
    return 0


if __name__ == "__main__":
    main()
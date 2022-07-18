import os
import argparse
import yaml
import tqdm
import pandas as pd

import torch 
import torch.nn as nn
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

<<<<<<< HEAD
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
=======
def train():
    需要train的函数另外定义
    return 0
>>>>>>> 34449d9a1e7b5bfa27c1e285dcec7d31131906ca

if __name__ == "__main__":
    main()
import config as cfg
import torch
import cv2
from  networks.MobileNetV2 import *
from  networks.ShuffleNetV2 import *
from dataloader.mpii.dataset_factory import DatasetFactory
import torch.optim as optim
import torch.utils.data
import dsntnn
import  os
import numpy as np
def main():
    minloss = np.float("inf")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_log = open(cfg.train_log, 'a+')
    valid_log = open(cfg.valid_log, 'a+')
    # train_dset = COCOBBox(cfg)
    # train_loader = torch.utils.data.DataLoader(train_dset, batch_size=cfg.BATCH_SIZE,
    #                                            shuffle=True, num_workers=cfg.NUM_WORKS)
    train_dataset = DatasetFactory.get_train_dataset(cfg.model, cfg.INPUT_SIZE)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                                  shuffle=True, num_workers=cfg.NUM_WORKS)

    test_dataset = DatasetFactory.get_test_dataset(cfg.model, cfg.INPUT_SIZE)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE,
                            shuffle=False, num_workers = cfg.NUM_WORKS)
    net = MobileNet(cfg.n_locations)
    if os.path.exists(cfg.CHECKPOINT_PATH):
        checkpoint = torch.load(cfg.CHECKPOINT_PATH)
        step = checkpoint['step']
        start = checkpoint['epoch']
        minloss = checkpoint['minloss']
        net.load_state_dict(checkpoint['net_state_dict'])
        #net = which_model(cfg.IS_SHALLOW, net_state_dict=checkpoint['net_state_dict'])
        # net = ShuffleNetV2()
    else:
        start = 0
        step = -1
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.LR)
    if os.path.exists(cfg.CHECKPOINT_PATH):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # criterion = torch.nn.MSELoss(reduction='sum').to(device)
    #reduction='sum'

    # training
    net.train()
    for epoch in range(start , cfg.MAX_EPOCH):
        train_loss = []
        train_loss_hm = []
        train_loss_coords = []
        valid_loss = []
        valid_loss_hm = []
        valid_loss_coords = []
        for i , data in enumerate(train_loader):
            #img , heat_map , limb_sigma ,ignore_mask = img.float().to(device) , heat_map.float().to(device) , limb_sigma.float().to(device), ignore_mask.float().to(device)
            images, poses = data['image'], data['pose']
            images, poses = images.to(device), poses.to(device)
            output = net(images)
            hm_loss = 0
            euc = 0
            reg = 0
            for i in range(len(output)):
                tmp =dsntnn.flat_softmax( output[i])
                coords = dsntnn.dsnt(tmp)
                euc_losses = dsntnn.euclidean_losses(coords, poses)
                reg_losses = dsntnn.js_reg_losses(tmp, poses, sigma_t=1.0)
                euc += euc_losses
                reg += reg_losses
                # Combine losses into an overall loss
                hm_loss += dsntnn.average_loss(reg_losses+euc_losses)
            loss = hm_loss
            step += 1
            lr = cfg.LR * (1 - step / (cfg.MAX_EPOCH * len(train_loader)))
            optimizer.param_groups[0]['lr'] = lr
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx = step%len(train_loader)
            train_loss.append(loss.item())
            train_loss_hm.append(torch.mean(euc).item())
            train_loss_coords.append(torch.mean(reg).item())
            # print('epoch:{}\tstep:{}/{}\thm_loss:{}\tcoords_loss:{}\tloss:{}\n'.format(epoch, idx, len(train_loader),
            #                                                                            train_loss_hm[idx],
            #                                                                            train_loss_coords[idx],
            #                                                                            train_loss[idx]))

        torch.save({'epoch': epoch+1,
                            'step': step,
                            'net_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'minloss':minloss},
                           cfg.CHECKPOINT_PATH_NEW)
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(test_dataloader):
                images = sample_batched['image'].to(device)
                poses = sample_batched['pose'].to(device)
                output = net(images)
                hm_loss = 0
                euc = 0
                reg = 0
                for i in range(len(output)):
                    tmp = dsntnn.flat_softmax(output[i])
                    coords = dsntnn.dsnt(tmp)
                    euc_losses = dsntnn.euclidean_losses(coords, poses)
                    reg_losses = dsntnn.js_reg_losses(tmp, poses, sigma_t=1.0)
                    euc += euc_losses
                    reg += reg_losses
                    # Combine losses into an overall loss
                    hm_loss += dsntnn.average_loss(reg_losses + euc_losses)
                loss = hm_loss
                valid_loss.append(loss.item())
                valid_loss_hm.append(torch.mean(euc).item())
                valid_loss_coords.append(torch.mean(reg).item())
            print(np.mean(np.array(valid_loss)))
            if np.mean(np.array(valid_loss)) < minloss:
                minloss = np.mean(np.array(valid_loss))
                torch.save({'epoch': epoch+1,
                            'step': step,
                            'net_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'minloss':minloss},
                           cfg.CHECKPOINT_PATH)

        # log
        for idx in range(len(train_loss)):
            train_log.write(
                'epoch:{}\tstep:{}/{}\thm_loss:{}\tcoords_loss:{}\tloss:{}\n'.format(epoch, idx, len(train_loader),
                                                                                     train_loss_hm[idx],
                                                                                     train_loss_coords[idx],
                                                                                     train_loss[idx]))
            train_log.flush()

        for idx in range(len(valid_loss)):
            valid_log.write(
                'epoch:{}\tstep:{}/{}\thm_loss:{}\tcoords_loss:{}\tloss:{}\n'.format(epoch, idx, len(test_dataloader),
                                                                                     valid_loss_hm[idx],
                                                                                     valid_loss_coords[idx],
                                                                                     valid_loss[idx]))
            valid_log.flush()



if __name__ == '__main__':
    main()

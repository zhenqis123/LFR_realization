
import torch
import torch.nn as nn
import torch.optim as optim
from model.model import UNet
from dataset_manager.latent import Latent
import argparse
import torch.utils.data.dataloader
import tqdm
from PIL import Image
from utils.saveimg import saveimg
import os
from torchvision import datasets, transforms
from utils.toLatentTransform import toLatentTransform
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.profiler
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_epochs', type=int, default=15)
argparser.add_argument('--batch_size', type=int, default=10)
argparser.add_argument('--learning_rate', type=float, default=0.001)
argparser.add_argument('--latent_path', type=str, default='../latent_synthetic/')
argparser.add_argument('--rolled_path', type=str, default='/disk1/panzhiyu/fingerprint/NIST14/image')
argparser.add_argument('--ridge_path', type=str, default='../NIST14_veri/enh/')
argparser.add_argument('--mask_path', type=str, default='../NIST14_veri/seg/')
argparser.add_argument('--model_path', type=str, default='../latentFinger/log/')
argparser.add_argument('--info_path', type=str, default='../latentFinger/info.txt')
argparser.add_argument('--save_path', type=str, default='../latentFinger/enhancedimg/')
argparser.add_argument('--eval',action='store_true')
arg = argparser.parse_args()

def main():
    model = UNet(1, 2)
    model = model.float()
    
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    transfrom = transforms.Compose([
        toLatentTransform(0.5),
        transforms.ToTensor(),
    ])
    dataset = Latent(arg.rolled_path, arg.ridge_path, arg.mask_path, arg.info_path, 10, transform=transfrom)
    train_size = int(0.99 * len(dataset))
    test_size = len(dataset) - train_size
    test_dataset, train_dataset = torch.utils.data.random_split(dataset, [test_size, train_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=arg.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=arg.batch_size, shuffle=False)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # for i, (latent_img, target, ridge_img, mask)in enumerate((test_loader)):
    #     for j in range(arg.batch_size):
    #         saveimg(latent_img[j], arg.save_path + 'latent_{}.png'.format(i*arg.batch_size+j))
    # writer = SummaryWriter('../latentFinger/log')
    if arg.eval:
        model.load_state_dict(torch.load(arg.model_path + 'model_{}.pth'.format(1)))
        test(model, test_loader, criterion)
    else:
        model.load_state_dict(torch.load(arg.model_path + 'model_{}.pth'.format(1)))
        for epoch in range(arg.num_epochs):
            
            model.train()
            dataset_pr = tqdm.tqdm(train_loader)
            dataset_pr.set_description('epoch: {}'.format(epoch))
            losses = 0
            for i, (latent_img, target, ridge_img, mask)in enumerate((dataset_pr)):
                optimizer.zero_grad()
                latent_img = latent_img.cuda()
                target = target.cuda()
                ridge_img = ridge_img.cuda()
                mask = mask.cuda()
                output = model(latent_img)

                output1 = output[:, 0].unsqueeze(1)
                output2 = output[:, 1].unsqueeze(1)
                loss1 = F.mse_loss(output1, target, reduction='none') * mask.unsqueeze(0)
                loss1 = loss1.sum() / mask.sum()
                loss2 =  criterion(output2, ridge_img)
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()
                losses += loss.item()
                dataset_pr.set_description('epoch: {}, loss_target: {}, loss_ridge: {}'.format(epoch, loss1.item()/arg.batch_size, loss2.item()/arg.batch_size))
                dataset_pr.update(1)
                # writer.add_scalar('loss_target', loss1.item()/arg.batch_size, epoch*len(dataset_pr)+i)
                # writer.add_scalar('loss_ridge', loss2.item()/arg.batch_size, epoch*len(dataset_pr)+i)
                # writer.add_scalar('loss', loss.item()/arg.batch_size, epoch*len(dataset_pr)+i)
                # writer.add_image('output1', output1[0], epoch*len(dataset_pr)+i)
                # writer.add_image('output1_target', target[0], epoch*len(dataset_pr)+i)
                # writer.add_image('output2', output2[0], epoch*len(dataset_pr)+i)
                # writer.add_image('output2_target', ridge_img[0], epoch*len(dataset_pr)+i)

                # for j in range(arg.batch_size):
                #     saveimg(latent[j], arg.save_path + 'latent_{}.png'.format(i*arg.batch_size+j))
                #     saveimg(rolled[j], arg.save_path + 'rolled_{}.png'.format(i*arg.batch_size+j))
                #     saveimg(ridge[j], arg.save_path + 'ridge_{}.png'.format(i*arg.batch_size+j))
                # import pdb; pdb.set_trace()
            print('epoch: {}, loss: {}'.format(epoch, losses/arg.batch_size))
            dataset_pr.close()
            # test(model, test_loader, criterion)
            torch.save(model.state_dict(), arg.model_path + 'model_{}.pth'.format(epoch))
    # writer.close()


def train(model, dataset, criterion, optimizer, epoch, writer):
    model.train()
    dataset_pr = tqdm.tqdm(dataset)
    dataset_pr.set_description('epoch: {}'.format(epoch))
    losses = 0
    for i, (latent_img, target, ridge_img, mask)in enumerate((dataset_pr)):
        optimizer.zero_grad()
        latent_img = latent_img.cuda()
        target = target.cuda()
        ridge_img = ridge_img.cuda()
        mask = mask.cuda()
        output = model(latent_img)

        mask_new = mask.unsqueeze(0)
        mask_new = (1-mask_new)*0.01 + mask_new
        loss1 = F.mse_loss(output[:, 0].unsqueeze(1), target, reduction='none') * mask_new
        loss1 = loss1.sum() / mask.sum()
        loss2 =  criterion(output[:, 1].unsqueeze(1), ridge_img)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        losses += loss.item()
        dataset_pr.set_description('epoch: {}, loss_target: {}, loss_ridge: {}'.format(epoch, loss1.item()/arg.batch_size, loss2.item()/arg.batch_size))
        dataset_pr.update(1)
        # for j in range(arg.batch_size):
        #     saveimg(latent[j], arg.save_path + 'latent_{}.png'.format(i*arg.batch_size+j))
        #     saveimg(rolled[j], arg.save_path + 'rolled_{}.png'.format(i*arg.batch_size+j))
        #     saveimg(ridge[j], arg.save_path + 'ridge_{}.png'.format(i*arg.batch_size+j))
        # import pdb; pdb.set_trace()
    print('epoch: {}, loss: {}'.format(epoch, losses/arg.batch_size))
    dataset_pr.close()
def test(model, dataset, criterion):
    model.eval()
    with torch.no_grad():
        for i, (latent_img, target, ridge_img, mask)in enumerate((dataset)):
            latent_img = latent_img.cuda()
            target = target.cuda()
            ridge_img = ridge_img.cuda()
            mask = mask.cuda()
            output = model(latent_img)
            loss = F.mse_loss(output[:, 0].unsqueeze(1), target, reduction='none') * mask.unsqueeze(0)
            loss = loss.sum() / mask.sum()
            loss =  loss + criterion(output[:, 1].unsqueeze(1), ridge_img)
            rolled = output[:, 0].unsqueeze(1)
            ridge = output[:, 1].unsqueeze(1)
            print('loss: {}'.format(loss.item()/arg.batch_size))
            for j in range(arg.batch_size):
                saveimg(latent_img[j], arg.save_path + 'latent_{}.png'.format(i*arg.batch_size+j))
                saveimg(rolled[j], arg.save_path + 'rolled_{}.png'.format(i*arg.batch_size+j))
                saveimg(ridge[j], arg.save_path + 'ridge_{}.png'.format(i*arg.batch_size+j))


if __name__ == '__main__':
    path = os.listdir(arg.save_path)
    path = os.listdir(arg.model_path)
    for p in path:
        os.remove(arg.model_path + p)
    # main()
    
    # path = os.listdir(arg.latent_path)
    # for p in path:
    #     os.remove(arg.latent_path + p)
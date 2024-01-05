import os
import argparse

import time

import torch
from tensorboardX import SummaryWriter

from models.densenet import DenseNet
from models.dataset import NeiziData

parser = argparse.ArgumentParser(description="Pytorch Non-perfusion Classification")
parser.add_argument("--exp", required=True, help="experiment name")
parser.add_argument("--resume", "-r", default=None, action="store_true", help="resume from checkpoint")
args = parser.parse_args()

writer = SummaryWriter()


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (input_var, target_var) in enumerate(loader):
        # Create vaiables
        if torch.cuda.is_available():
            input_var = input_var.cuda()
            target_var = target_var.cuda()
        '''
        else:
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
        '''
        # compute output
        output = model(input_var)

        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        batch_size = target_var.size(0)
        _, pred = torch.topk(output, k=1, dim=1)
        pred = pred.cpu().data
        error.update(torch.ne(pred.squeeze(), target_var.cpu().squeeze()).float().sum() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def test_epoch(model, loader, print_freq=1, is_test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on eval mode
    model.eval()

    end = time.time()
    for batch_idx, (input_var, target_var) in enumerate(loader):
        # Create vaiables
        if torch.cuda.is_available():
            input_var = input_var.cuda()
            target_var = target_var.cuda()
        '''
        else:
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)
        '''
        # compute output
        output = model(input_var)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        batch_size = target_var.size(0)
        _, pred = torch.topk(output, k=1, dim=1)
        pred = pred.cpu().data
        error.update(torch.ne(pred.squeeze(), target_var.cpu().squeeze()).float().sum() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Test' if is_test else 'Valid',
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def train(model, train_label_file, valid_label_file, save, n_epochs=800, batch_size=16, momentum=0.9,
          lr=0.1, wd=1e-5, seed=None, best_error=1):
    if seed is not None:
        torch.manual_seed(seed)
    trainset = NeiziData(train_label_file)
    validset = NeiziData(valid_label_file)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=int(batch_size / 2), shuffle=True, num_workers=0)

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        print('Using %d GPUs' % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.1)
    # Train model
    for epoch in range(n_epochs):
        _, train_loss, train_error = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs
        )
        _, valid_loss, valid_error = test_epoch(
            model=model,
            loader=valid_loader,
            is_test=False
        )

        writer.add_scalars(args.exp + "/loss", {"train": train_loss}, epoch)
        writer.add_scalars(args.exp + "/loss", {"valid": valid_loss}, epoch)
        writer.add_scalars(args.exp + "/error", {"train": train_error}, epoch)
        writer.add_scalars(args.exp + "/error", {"valid": valid_error}, epoch)
        # Determine if model is the best
        state = {
            "net": model.state_dict(),
            "lr": optimizer.param_groups[0]['lr'],
            "epoch": epoch,
            "best_error": best_error,
            "seed": seed
        }
        if valid_loader and valid_error < best_error:
            best_error = valid_error
            print('New best error: %.4f' % best_error)
            torch.save(state, os.path.join(save, args.exp + '_best_model.pth'))
        else:
            torch.save(state, os.path.join(save, args.exp + '_model.pth'))

        # Log results
        with open(os.path.join(save, args.exp + '_results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
                (epoch + 1),
                train_loss,
                train_error,
                valid_loss,
                valid_error,
            ))
        scheduler.step()


if __name__ == "__main__":
    train_label_file = "./data/train.csv"
    valid_label_file = "./data/valid.csv"
    save_pth = "/data/feituai/eyelid_screen/ckpts/neizi_class/"
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    net = DenseNet(num_classes=3, drop_rate=0.3, small_inputs=True)
    lr = 0.01
    best_error = 1
    seed = None
    if args.resume:
        print('Loading pretrained model...')
        state_dict = torch.load(os.path.join("/data/feituai/ckpts/lung_nodule_classification/cube32_20201201",
                                             "cube32_20201201_best_model.pth"))
        net.load_state_dict(state_dict['net'])
        lr = state_dict['lr']
        seed = state_dict['seed']
        best_error = state_dict['best_error']

    train(net, train_label_file, valid_label_file, save_pth, lr=0.001, batch_size=8, seed=seed, best_error=best_error)

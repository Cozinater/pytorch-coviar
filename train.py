"""Run training."""

import shutil
import time
import numpy as np
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torchvision

from dataset import CoviarDataSet
from model import Model
from train_options import parser
from transforms import GroupCenterCrop
from transforms import GroupScale

SAVE_FREQ = 1
PRINT_FREQ = 20
best_prec1 = 0


def main():
    global args
    global best_prec1
    start_epoch = 0
    args = parser.parse_args()

    print('Training arguments:')
    for k, v in vars(args).items():
        print('\t{}: {}'.format(k, v))

    if args.data_name == 'ucf101':
        num_class = 101
    elif args.data_name == 'hmdb51':
        num_class = 51
    else:
        raise ValueError('Unknown dataset '+ args.data_name)

    model = Model(num_class, args.num_segments, args.representation, args.no_TopKAtt, args.topk,
                  base_model=args.arch)
    print(model)

    train_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            args.data_name,
            video_list=args.train_list,
            num_segments=args.num_segments,
            representation=args.representation,
            transform=model.get_augmentation(),
            is_train=True,
            accumulate=(not args.no_accumulation),
            ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            args.data_name,
            video_list=args.test_list,
            num_segments=args.num_segments,
            representation=args.representation,
            transform=torchvision.transforms.Compose([
                GroupScale(int(model.scale_size)),
                GroupCenterCrop(model.crop_size),
                ]),
            is_train=False,
            accumulate=(not args.no_accumulation),
            ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    cudnn.benchmark = True

    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        decay_mult = 0.0 if 'bias' in key else 1.0

        if ('module.base_model.conv1' in key
                or 'module.base_model.bn1' in key
                or 'data_bn' in key) and args.representation in ['mv', 'residual']:
            lr_mult = 0.1
        elif '.fc.' in key:
            lr_mult = 1.0
        else:
            lr_mult = 0.01

        params += [{'params': value, 'lr': args.lr, 'lr_mult': lr_mult, 'decay_mult': decay_mult}]


    optimizer = torch.optim.Adam(
        params,
        weight_decay=args.weight_decay,
        eps=0.001)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # resume training from previous checkpoint
    if args.weights is not None:
        if os.path.isfile(args.weights):
            print("=> loading checkpoint '{}'".format(args.weights))
            checkpoint = torch.load(args.weights)
            print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer'])

            def load_opt_update_cuda(optimizer, cuda_id):
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda(cuda_id)

            load_opt_update_cuda(optimizer, args.gpus[0])

    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #
    # # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])

    for epoch in range(start_epoch, args.epochs):
        cur_lr = adjust_learning_rate(optimizer, epoch, args.lr_steps, args.lr_decay)

        loss, prec1, prec5 = train(train_loader, model, criterion, optimizer, epoch, cur_lr)

        #if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion)

        is_best = val_prec1 > best_prec1
        best_prec1 = max(val_prec1, best_prec1)
        if is_best or epoch % SAVE_FREQ == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                },
                is_best,
                filename='checkpoint.pth.tar')

        log(epoch, prec1, prec5, loss, val_prec1, val_prec5, val_loss, cur_lr)

def log(epoch, prec1, prec5, loss, val_prec1, val_prec5, val_loss, cur_lr):
    f = open("output/pytorch_coviar_{0}_topk_{1}_output.log".format(args.representation, args.topk),"a")
    f.write('Epoch:{0} prec@1:{accuracy1:.3f} prec@5:{accuracy5:.3f} test_prec@1:{val_accuracy1:.3f} '
            'test_prec@5:{val_accuracy5:.3f} loss:{loss:.5f} val_loss:{val_loss:.5f} '
            'cur_lr:{cur_lr:.5f}\n'
            .format(epoch, accuracy1=prec1, accuracy5=prec5, val_accuracy1=val_prec1, val_accuracy5=val_prec5,
                    loss=loss, val_loss=val_loss, cur_lr=cur_lr))

def train(train_loader, model, criterion, optimizer, epoch, cur_lr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        #input_var = torch.autograd.Variable(input)
        #target_var = torch.autograd.Variable(target)

        output = model(input)
        output = output.view((-1, args.num_segments) + output.size()[1:])
        output = torch.mean(output, dim=1)

        loss = criterion(output, target)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader),
                       batch_time=batch_time,
                       data_time=data_time,
                       loss=losses,
                       top1=top1,
                       top5=top5,
                       lr=cur_lr)))

    return losses.avg, top1.avg, top5.avg

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        #input_var = torch.autograd.Variable(input, volatile=True)
        #target_var = torch.autograd.Variable(target, volatile=True)
        with torch.no_grad():
            output = model(input)
            output = output.view((-1, args.num_segments) + output.size()[1:])
            output = torch.mean(output, dim=1)
            loss = criterion(output, target)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % PRINT_FREQ == 0:
            print(('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader),
                       batch_time=batch_time,
                       loss=losses,
                       top1=top1,
                       top5=top5)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1, top5=top5, loss=losses)))

    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, filename):
    filename = '_'.join((args.model_prefix,"topk",str(args.topk), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.model_prefix,"topk",str(args.topk), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def adjust_learning_rate(optimizer, epoch, lr_steps, lr_decay):
    decay = lr_decay ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    wd = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = wd * param_group['decay_mult']
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

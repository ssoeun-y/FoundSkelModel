import argparse
import torch.nn.functional as F
import os
import random
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tools import AverageMeter, remove_prefix, sum_para_cnt

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
from dataset import get_finetune_training_set, get_finetune_validation_set

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--epochs', default=150, type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=30., type=float, metavar='LR', dest='lr')
parser.add_argument('--schedule', default=[120, 140], nargs='*', type=int)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=0., type=float, metavar='W', dest='weight_decay')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--pretrained', default='', type=str)
parser.add_argument('--evaluate', default='false', type=str)
parser.add_argument('--finetune-dataset', default='ntu60', type=str)
parser.add_argument('--protocol', default='cross_view', type=str)
parser.add_argument('--moda', default='joint', type=str)
parser.add_argument('--backbone', default='DSTE', type=str,
                    help='DSTE, STTR, GAT, TSM, BoundaryReg, DSTEAux, CausalDSTEAux, CausalDSTEError, CausalDSTEAFCM, BSDSTE, BSv2, STFM, or CDED')
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--lam', default=1.0, type=float)
parser.add_argument('--lam-future', default=0.1, type=float,
                    help='CausalDSTEAFCM: future prediction loss weight')
parser.add_argument('--lam-gate', default=0.5, type=float, help='CausalDSTEGate: background gate loss weight')
parser.add_argument('--snap-k', default=3, type=int)
parser.add_argument('--sigma', default=2.0, type=float)
parser.add_argument('--field-k', default=20.0, type=float)

args = parser.parse_args()
best_acc1 = 0


def load_pretrained(pretrained, model):
    if os.path.isfile(pretrained):
        checkpoint = torch.load(pretrained, map_location="cpu")
        state_dict = checkpoint['state_dict']
        state_dict = remove_prefix(state_dict)
        model_state = model.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_state and v.shape == model_state[k].shape}
        skipped = [k for k, v in state_dict.items()
                   if k in model_state and v.shape != model_state[k].shape]
        if skipped:
            print(f"[WARN] shape mismatch — skipped {len(skipped)} keys: {skipped}")
        msg = model.load_state_dict(filtered, strict=False)
        print("message", msg)
        print(set(msg.missing_keys))
        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        raise FileNotFoundError(
            "pretrained checkpoint not found: '{}'. "
            "Refusing to train from random initialization.".format(pretrained)
        )


def load_detector(detector, model):
    if os.path.isfile(detector):
        checkpoint = torch.load(detector, map_location="cpu")
        state_dict = checkpoint['state_dict']
        state_dict = remove_prefix(state_dict)
        msg = model.load_state_dict(state_dict, strict=True)
        print("message", msg)
        assert len(msg.missing_keys) + len(msg.unexpected_keys) == 0
        print("=> loaded detector '{}'".format(detector))
    else:
        print("=> no checkpoint found at '{}'".format(detector))


def main():
    args = parser.parse_args()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if (
        args.evaluate in ['false', 'False', 'none', 'None']
        and args.pretrained
        and not os.path.exists(args.pretrained)
    ):
        raise FileNotFoundError(
            "pretrained checkpoint not found: '{}'. "
            "Check the path relative to your current working directory.".format(args.pretrained)
        )
    print(type(args.evaluate), args.evaluate)
    if args.evaluate not in ['false', 'False', 'none', 'None']:
        pass
    main_worker(args)


def main_worker(args):
    global best_acc1

    from options import options_downstream as options
    if args.finetune_dataset == 'pku_v1':
        opts = options.opts_pku_v1_xsub()
    elif args.finetune_dataset == 'pku_v2' and args.protocol == 'cross_subject':
        opts = options.opts_pku_v2_xsub()

    if args.backbone == 'DSTE':
        from model.DSTE import Downstream
        model = Downstream(**opts.encoder_args)
    elif args.backbone == 'CausalDSTE':
        from model.DSTE_causal import DownstreamCausal
        model = DownstreamCausal(**opts.encoder_args)
    elif args.backbone == 'STTR':
        from model.STTR import Downstream
        model = Downstream(**opts.encoder_args)
    elif args.backbone == 'GAT':
        from model.GAT_detection import DownstreamGAT
        model = DownstreamGAT(**opts.encoder_args)
    elif args.backbone == 'TSM':
        from model.TSM_detection import DownstreamTSM
        model = DownstreamTSM(**opts.encoder_args)
    elif args.backbone == 'BoundaryReg':
        from model.BoundaryReg_detection import DownstreamBoundary
        model = DownstreamBoundary(**opts.encoder_args)
    elif args.backbone == 'DSTEAux':
        from model.DSTE_aux import DownstreamAux
        model = DownstreamAux(**opts.encoder_args)
    elif args.backbone == 'CausalDSTEAux':
        from model.DSTE_causal_aux import DownstreamCausalAux
        model = DownstreamCausalAux(**opts.encoder_args)
    elif args.backbone == 'CausalDSTEError':
        from model.DSTE_causal_error import DownstreamCausalError
        model = DownstreamCausalError(**opts.encoder_args)
    elif args.backbone == 'CausalDSTEGate':
        from model.DSTE_causal_gate import DownstreamCausalGate
        model = DownstreamCausalGate(**opts.encoder_args)
    elif args.backbone == 'CausalDSTEAFCM':
        from model.DSTE_causal_afcm import DownstreamCausalAFCM
        model = DownstreamCausalAFCM(**opts.encoder_args, lam_future=args.lam_future)
    elif args.backbone == 'BSDSTE':
        from model.BS_DSTE import DownstreamBS
        model = DownstreamBS(**opts.encoder_args)
    elif args.backbone == 'BSv2':
        from model.BS_DSTE_v2 import DownstreamBSv2
        model = DownstreamBSv2(**opts.encoder_args)
    elif args.backbone == 'STFM':
        from model.STFM_detection import DownstreamSTFM
        model = DownstreamSTFM(**opts.encoder_args, field_k=args.field_k)
    elif args.backbone == 'CDED':
        from model.CDED_detection import DownstreamCDED
        model = DownstreamCDED(**opts.encoder_args)

    tag = args.tag if args.tag else f'{args.backbone.lower()}_{args.moda}'
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')
    save_dir = os.path.join('results', f'{tag}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    print(f'[save_dir] {save_dir}')
    print(sum_para_cnt(model) / 1e6)
    print("options", opts.encoder_args, opts.train_feeder_args, opts.test_feeder_args)
    print('\n', args)

    if args.backbone == 'BoundaryReg':
        nn.init.normal_(model.cls_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(model.cls_head.bias)
        nn.init.xavier_uniform_(model.start_head.weight)
        nn.init.zeros_(model.start_head.bias)
        nn.init.xavier_uniform_(model.end_head.weight)
        nn.init.zeros_(model.end_head.bias)
        nn.init.dirac_(model.cls_conv.weight)
        nn.init.dirac_(model.reg_conv.weight)
    elif args.backbone in ('DSTEAux', 'CausalDSTEAux', 'CausalDSTEError', 'CausalDSTEAFCM', 'CausalDSTEGate'):
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
        nn.init.xavier_uniform_(model.start_head.weight)
        nn.init.zeros_(model.start_head.bias)
        nn.init.xavier_uniform_(model.end_head.weight)
        nn.init.zeros_(model.end_head.bias)
        if hasattr(model, 'fc_proj'):
            nn.init.xavier_uniform_(model.fc_proj.weight)
            nn.init.zeros_(model.fc_proj.bias)
    elif args.backbone == 'BSDSTE':
        nn.init.normal_(model.cls_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(model.cls_head.bias)
        nn.init.xavier_uniform_(model.start_head.weight)
        nn.init.zeros_(model.start_head.bias)
        nn.init.xavier_uniform_(model.end_head.weight)
        nn.init.zeros_(model.end_head.bias)
        nn.init.xavier_uniform_(model.start_head_mid.weight)
        nn.init.zeros_(model.start_head_mid.bias)
        nn.init.xavier_uniform_(model.end_head_mid.weight)
        nn.init.zeros_(model.end_head_mid.bias)
    elif args.backbone == 'BSv2':
        nn.init.normal_(model.cls_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(model.cls_head.bias)
        for head in ['start_head', 'end_head', 'start_head_mid', 'end_head_mid',
                     'start_head2', 'end_head2']:
            nn.init.xavier_uniform_(getattr(model, head).weight)
            nn.init.zeros_(getattr(model, head).bias)
    elif args.backbone == 'STFM':
        nn.init.normal_(model.cls_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(model.cls_head.bias)
        for head in ['start_head', 'end_head', 'field_head']:
            nn.init.xavier_uniform_(getattr(model, head).weight)
            nn.init.zeros_(getattr(model, head).bias)
    elif args.backbone == 'CDED':
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
        for head in ['start_head', 'end_head', 'loc_head']:
            nn.init.xavier_uniform_(getattr(model, head).weight)
            nn.init.zeros_(getattr(model, head).bias)
    else:
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()

    if args.pretrained and 'pth' not in args.evaluate:
        load_pretrained(args.pretrained, model)
        model = nn.DataParallel(model)

    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    fc_parameters = []
    other_parameters = []
    for name, param in model.named_parameters():
        is_head = name.startswith('module.fc') or (
            args.backbone == 'BoundaryReg' and
            any(name.startswith(f'module.{h}') for h in ['cls_head', 'start_head', 'end_head'])
        ) or (
            args.backbone in ('DSTEAux', 'CausalDSTEAux', 'CausalDSTEError') and
            any(name.startswith(f'module.{h}') for h in ['start_head', 'end_head', 'e_gate_start', 'e_gate_end'])
        ) or (
            args.backbone == 'CausalDSTEGate' and
            any(name.startswith(f'module.{h}') for h in ['start_head', 'end_head', 'gate_head'])
        ) or (
            args.backbone == 'CausalDSTEAFCM' and
            any(name.startswith(f'module.{h}') for h in ['start_head', 'end_head', 'afcm'])
        ) or (
            args.backbone == 'BSDSTE' and
            any(name.startswith(f'module.{h}') for h in [
                'cls_head', 'start_head', 'end_head',
                'start_head_mid', 'end_head_mid', 'guided_t_tr1.alpha'])
        ) or (
            args.backbone == 'BSv2' and
            any(name.startswith(f'module.{h}') for h in [
                'cls_head', 'start_head', 'end_head',
                'start_head_mid', 'end_head_mid',
                'start_head2', 'end_head2', 'guided_t_tr1.alpha'])
        ) or (
            args.backbone == 'STFM' and
            any(name.startswith(f'module.{h}') for h in [
                'cls_head', 'start_head', 'end_head', 'field_head'])
        ) or (
            args.backbone == 'CDED' and
            any(name.startswith(f'module.{h}') for h in [
                'fc', 'start_head', 'end_head', 'loc_head'])
        )
        param.requires_grad = True
        if is_head:
            fc_parameters.append(param)
        else:
            other_parameters.append(param)

    params = [{'params': fc_parameters, 'lr': args.lr},
              {'params': other_parameters, 'lr': args.lr}]
    optimizer = torch.optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay)
    for parm in optimizer.param_groups:
        print("optimize parameters lr ", parm['lr'])

    train_dataset = get_finetune_training_set(opts)
    val_dataset = get_finetune_validation_set(opts)
    trainloader_params = {
        'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 8,
        'pin_memory': True, 'prefetch_factor': 4, 'persistent_workers': True
    }
    valloader_params = {
        'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 8,
        'pin_memory': True, 'prefetch_factor': 4, 'persistent_workers': True
    }
    train_loader = torch.utils.data.DataLoader(train_dataset, **trainloader_params)
    val_loader = torch.utils.data.DataLoader(val_dataset, **valloader_params)

    for epoch in range(0, args.epochs):
        if args.evaluate not in ['false', 'False', 'none', 'None']:
            detector_path = args.evaluate
            load_detector(detector_path, model)
            with torch.no_grad():
                generate_bbox(val_loader, model, args, save_dir=save_dir)
            break
        adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, criterion, optimizer, epoch, args)
        if (epoch + 1) % 5 == 0:
            state = {'state_dict': model.state_dict()}
            torch.save(state, os.path.join(save_dir, f'epoch{epoch}_detection.pth.tar'))
            acc1 = validate(val_loader, model, criterion, args)
        else:
            acc1 = 0
    print("class head Final best accuracy", best_acc1)


def _build_boundary_gt(target, sigma):
    B, T = target.shape
    frames = torch.arange(T, device=target.device).float()
    prev_t = target[:, :-1]
    curr_t = target[:, 1:]
    start_gt = torch.zeros(B, T, device=target.device)
    end_gt = torch.zeros(B, T, device=target.device)
    for b in range(B):
        s_pos = (1 + ((curr_t[b] != 0) & (prev_t[b] == 0)).nonzero(as_tuple=True)[0]).tolist()
        e_pos = ((curr_t[b] == 0) & (prev_t[b] != 0)).nonzero(as_tuple=True)[0].tolist()
        for t0 in s_pos:
            start_gt[b] = torch.maximum(start_gt[b],
                                        torch.exp(-((frames - t0) ** 2) / (2 * sigma ** 2)))
        for t0 in e_pos:
            end_gt[b] = torch.maximum(end_gt[b],
                                      torch.exp(-((frames - t0) ** 2) / (2 * sigma ** 2)))
    return start_gt, end_gt


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':1.4f')
    top1 = AverageMeter('Acc@1', ':3.2f')
    bgtop1 = AverageMeter('bgAcc@1', ':3.2f')
    actop1 = AverageMeter('acAcc@1', ':3.2f')
    top5 = AverageMeter('Acc@5', ':3.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, losses, top1, bgtop1, actop1, top5],
                             prefix="Epoch: [{}]".format(epoch))
    model.eval()
    end = time.time()

    for i, (jt, js, bt, bs, mt, ms, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        jt = jt.float().cuda(non_blocking=True)
        js = js.float().cuda(non_blocking=True)
        bt = bt.float().cuda(non_blocking=True)
        bs = bs.float().cuda(non_blocking=True)
        mt = mt.float().cuda(non_blocking=True)
        ms = ms.float().cuda(non_blocking=True)
        target = target.long().cuda(non_blocking=True)

        # ── forward ───────────────────────────────────────────────────
        if args.backbone == 'CausalDSTEAFCM':
            output = model(jt, js, bt, bs, mt, ms, knn_eval=False, detect=True,
                           compute_future_loss=True)
        else:
            output = model(jt, js, bt, bs, mt, ms, knn_eval=False, detect=True)

        # ── loss ──────────────────────────────────────────────────────
        if args.backbone == 'BoundaryReg':
            cls_logits, start_logits, end_logits = output
            prev_t = target[:, :-1]
            curr_t = target[:, 1:]
            start_gt = torch.zeros_like(target, dtype=torch.float32)
            start_gt[:, 1:] = ((curr_t != 0) & (prev_t == 0)).float()
            end_gt = torch.zeros_like(target, dtype=torch.float32)
            end_gt[:, :-1] = ((curr_t == 0) & (prev_t != 0)).float()
            loss_cls = criterion(cls_logits.reshape(-1, cls_logits.size(-1)), target.reshape(-1))
            num_pos = start_gt.sum(dim=1) + 1e-6
            w = min(((start_gt.size(1) - num_pos) / num_pos).mean().item(), 50.0)
            pos_weight = torch.tensor([w], device=target.device)
            loss_start = F.binary_cross_entropy_with_logits(
                start_logits.squeeze(-1), start_gt, pos_weight=pos_weight)
            loss_end = F.binary_cross_entropy_with_logits(
                end_logits.squeeze(-1), end_gt, pos_weight=pos_weight)
            loss = loss_cls + args.lam * (loss_start + loss_end)
            output = cls_logits.reshape(-1, cls_logits.size(-1))
            target = target.reshape(-1)

        elif args.backbone in ('DSTEAux', 'CausalDSTEAux', 'CausalDSTEError'):
            cls_logits, start_logits, end_logits = output
            start_gt, end_gt = _build_boundary_gt(target, args.sigma)
            loss_cls = criterion(cls_logits.reshape(-1, cls_logits.size(-1)), target.reshape(-1))
            loss_start = F.binary_cross_entropy_with_logits(start_logits.squeeze(-1), start_gt)
            loss_end = F.binary_cross_entropy_with_logits(end_logits.squeeze(-1), end_gt)
            loss = loss_cls + args.lam * (loss_start + loss_end)
            output = cls_logits.reshape(-1, cls_logits.size(-1))
            target = target.reshape(-1)

        elif args.backbone == 'CausalDSTEGate':
            cls_logits, start_logits, end_logits, gate = output
            start_gt, end_gt = _build_boundary_gt(target, args.sigma)
            action_mask = (target > 0).float()
            loss_cls   = criterion(cls_logits.reshape(-1, cls_logits.size(-1)), target.reshape(-1))
            loss_start = F.binary_cross_entropy_with_logits(start_logits.squeeze(-1), start_gt)
            loss_end   = F.binary_cross_entropy_with_logits(end_logits.squeeze(-1), end_gt)
            loss_gate  = F.binary_cross_entropy(gate.squeeze(-1), action_mask)
            loss = (loss_cls + args.lam * (loss_start + loss_end) + args.lam_gate * loss_gate)
            output = cls_logits.reshape(-1, cls_logits.size(-1))
            target = target.reshape(-1)
        elif args.backbone == 'CausalDSTEAFCM':
            cls_logits, start_logits, end_logits, L_future = output
            start_gt, end_gt = _build_boundary_gt(target, args.sigma)
            loss_cls = criterion(cls_logits.reshape(-1, cls_logits.size(-1)), target.reshape(-1))
            loss_start = F.binary_cross_entropy_with_logits(start_logits.squeeze(-1), start_gt)
            loss_end = F.binary_cross_entropy_with_logits(end_logits.squeeze(-1), end_gt)
            loss = (loss_cls + args.lam * (loss_start + loss_end)
                    + args.lam_future * L_future.mean())
            output = cls_logits.reshape(-1, cls_logits.size(-1))
            target = target.reshape(-1)

        elif args.backbone == 'BSDSTE':
            cls_logits, start_logits, end_logits, A_start, A_end = output
            start_gt, end_gt = _build_boundary_gt(target, args.sigma)
            loss_cls = criterion(cls_logits.reshape(-1, cls_logits.size(-1)), target.reshape(-1))
            loss_start = F.binary_cross_entropy_with_logits(start_logits.squeeze(-1), start_gt)
            loss_end = F.binary_cross_entropy_with_logits(end_logits.squeeze(-1), end_gt)
            loss_start_mid = F.binary_cross_entropy(A_start, start_gt)
            loss_end_mid = F.binary_cross_entropy(A_end, end_gt)
            loss = (loss_cls + args.lam * (loss_start + loss_end)
                    + args.lam * (loss_start_mid + loss_end_mid))
            output = cls_logits.reshape(-1, cls_logits.size(-1))
            target = target.reshape(-1)

        elif args.backbone == 'BSv2':
            cls_logits, start_logits, end_logits, \
                A_start_logit, A_end_logit, A_start_mid, A_end_mid, \
                start_logits2, end_logits2 = output
            start_gt, end_gt = _build_boundary_gt(target, args.sigma)
            num_pos = start_gt.sum(dim=1) + 1e-6
            w = min(((start_gt.size(1) - num_pos) / num_pos).mean().item(), 50.0)
            pos_weight = torch.tensor([w], device=target.device)
            loss_cls = criterion(cls_logits.reshape(-1, cls_logits.size(-1)), target.reshape(-1))
            loss_start = F.binary_cross_entropy_with_logits(start_logits.squeeze(-1), start_gt)
            loss_end = F.binary_cross_entropy_with_logits(end_logits.squeeze(-1), end_gt)
            loss_start_mid = F.binary_cross_entropy_with_logits(A_start_logit, start_gt,
                                                                 pos_weight=pos_weight)
            loss_end_mid = F.binary_cross_entropy_with_logits(A_end_logit, end_gt,
                                                               pos_weight=pos_weight)
            loss_start2 = F.binary_cross_entropy_with_logits(start_logits2.squeeze(-1), start_gt)
            loss_end2 = F.binary_cross_entropy_with_logits(end_logits2.squeeze(-1), end_gt)
            loss = (loss_cls + args.lam * (loss_start + loss_end)
                    + args.lam * (loss_start_mid + loss_end_mid)
                    + args.lam * (loss_start2 + loss_end2))
            output = cls_logits.reshape(-1, cls_logits.size(-1))
            target = target.reshape(-1)

        elif args.backbone == 'STFM':
            cls_logits, start_logits, end_logits, F_pred, start_from_field, end_from_field = output
            start_gt, end_gt = _build_boundary_gt(target, args.sigma)
            B2, T2 = target.shape
            F_gt = torch.zeros(B2, T2, device=target.device)
            for b in range(B2):
                f_step = np.where(target[b].cpu().numpy() > 0, 1.0, -1.0).astype(np.float64)
                f_smooth = gaussian_filter1d(f_step, sigma=0.5)
                F_gt[b] = torch.tensor(f_smooth, dtype=torch.float32, device=target.device)
            loss_cls = criterion(cls_logits.reshape(-1, cls_logits.size(-1)), target.reshape(-1))
            loss_start = F.binary_cross_entropy_with_logits(start_logits.squeeze(-1), start_gt)
            loss_end = F.binary_cross_entropy_with_logits(end_logits.squeeze(-1), end_gt)
            loss_field = F.mse_loss(F_pred.squeeze(-1), F_gt)
            loss_grad_start = F.binary_cross_entropy(start_from_field, start_gt[:, 1:])
            loss_grad_end = F.binary_cross_entropy(end_from_field, end_gt[:, :-1])
            loss = (loss_cls + args.lam * (loss_start + loss_end)
                    + args.lam * (loss_field + loss_grad_start + loss_grad_end))
            output = cls_logits.reshape(-1, cls_logits.size(-1))
            target = target.reshape(-1)

        elif args.backbone == 'CDED':
            cls_logits, start_logits, end_logits, loc_pred = output
            start_gt, end_gt = _build_boundary_gt(target, args.sigma)
            B2, T2 = target.shape
            gt_loc = torch.zeros(B2, T2, 2, device=target.device)
            for b in range(B2):
                label_np = target[b].cpu().numpy()
                t = 0
                while t < T2:
                    if label_np[t] == 0:
                        t += 1
                        continue
                    s, cls_val = t, label_np[t]
                    while t < T2 and label_np[t] == cls_val:
                        t += 1
                    e = t - 1
                    seg_len = e - s + 1
                    gt_loc[b, s:e + 1, 0] = torch.arange(seg_len, dtype=torch.float32)
                    gt_loc[b, s:e + 1, 1] = torch.arange(seg_len - 1, -1, -1, dtype=torch.float32)
            fg_mask = (target > 0)
            loss_cls = criterion(cls_logits.reshape(-1, cls_logits.size(-1)), target.reshape(-1))
            loss_start = F.binary_cross_entropy_with_logits(start_logits.squeeze(-1), start_gt)
            loss_end = F.binary_cross_entropy_with_logits(end_logits.squeeze(-1), end_gt)
            loss_loc = (F.smooth_l1_loss(loc_pred[fg_mask], gt_loc[fg_mask])
                        if fg_mask.any() else loc_pred.sum() * 0.0)
            loss = loss_cls + args.lam * (loss_start + loss_end) + args.lam * loss_loc
            output = cls_logits.reshape(-1, cls_logits.size(-1))
            target = target.reshape(-1)

        else:
            output = output.reshape(-1, 52)
            target = target.reshape(-1)
            loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 5), ignore=-1)
        bgacc1, _ = accuracy(output, target, topk=(1, 5), ignore=1)
        acacc1, _ = accuracy(output, target, topk=(1, 5), ignore=0)
        losses.update(loss.item(), output.shape[0])
        top1.update(acc1[0], output.shape[0])
        top5.update(acc5[0], output.shape[0])
        bgtop1.update(bgacc1[0], output.shape[0])
        actop1.update(acacc1[0], output.shape[0])
        batch_time.update(time.time() - end)
        end = time.time()
        if i + 1 == len(train_loader):
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':1.4f')
    top1 = AverageMeter('Acc@1', ':3.2f')
    bgtop1 = AverageMeter('bgAcc@1', ':3.2f')
    actop1 = AverageMeter('acAcc@1', ':3.2f')
    top5 = AverageMeter('Acc@5', ':3.2f')
    progress = ProgressMeter(len(val_loader),
                             [batch_time, losses, top1, bgtop1, actop1, top5],
                             prefix='Test: ')
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (jt, js, bt, bs, mt, ms, target, sample_name) in enumerate(val_loader):
            jt = jt.float().cuda(non_blocking=True)
            js = js.float().cuda(non_blocking=True)
            bt = bt.float().cuda(non_blocking=True)
            bs = bs.float().cuda(non_blocking=True)
            mt = mt.float().cuda(non_blocking=True)
            ms = ms.float().cuda(non_blocking=True)
            target = target.long().cuda(non_blocking=True)
            output = model(jt, js, bt, bs, mt, ms, knn_eval=False, detect=True)
            if args.backbone in ('BoundaryReg', 'DSTEAux', 'CausalDSTEAux', 'CausalDSTEError',
                                  'CausalDSTEAFCM', 'CausalDSTEGate', 'BSDSTE', 'BSv2', 'STFM', 'CDED'):
                cls_logits = output[0]
                output = cls_logits.reshape(-1, cls_logits.size(-1))
            else:
                output = output.reshape(-1, 52)
            target = target.reshape(-1)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5), ignore=-1)
            bgacc1, _ = accuracy(output, target, topk=(1, 5), ignore=1)
            acacc1, _ = accuracy(output, target, topk=(1, 5), ignore=0)
            losses.update(loss.item(), output.shape[0])
            top1.update(acc1[0], output.shape[0])
            top5.update(acc5[0], output.shape[0])
            bgtop1.update(bgacc1[0], output.shape[0])
            actop1.update(acacc1[0], output.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if i + 1 == len(val_loader):
                progress.display(i)
        print(' * Acc@1 {top1.avg:.3f} * bgAcc@1 {bgtop1.avg:.3f} * acAcc@1 {actop1.avg:.3f}\t\tAcc@5 {top5.avg:.3f}'
              .format(top1=top1, bgtop1=bgtop1, actop1=actop1, top5=top5))
    return top1.avg


def generate_bbox(val_loader, model, args, thereshold=0.02, save_dir=None):
    from tqdm import tqdm
    model.eval()
    proposal = {}
    base = save_dir if save_dir else os.path.dirname(args.evaluate)
    sp = os.path.join(base, 'detect_each_frame') + '/'
    os.makedirs(sp, exist_ok=True)

    for i, (jt, js, bt, bs, mt, ms, target, sample_name) in tqdm(enumerate(val_loader)):
        jt = jt.float().cuda(non_blocking=True)
        js = js.float().cuda(non_blocking=True)
        bt = bt.float().cuda(non_blocking=True)
        bs = bs.float().cuda(non_blocking=True)
        mt = mt.float().cuda(non_blocking=True)
        ms = ms.float().cuda(non_blocking=True)
        target = target.long().cuda()
        raw = model(jt, js, bt, bs, mt, ms, detect=True)

        if args.backbone == 'BoundaryReg':
            cls_logits, start_logits, end_logits = raw
            start_score = torch.sigmoid(start_logits.squeeze(-1))
            end_score = torch.sigmoid(end_logits.squeeze(-1))
            output = F.softmax(cls_logits, dim=-1)
        elif args.backbone in ('DSTEAux', 'CausalDSTEAux', 'CausalDSTEError', 'CausalDSTEAFCM', 'CausalDSTEGate', 'BSDSTE'):
            cls_logits, start_logits, end_logits = raw[0], raw[1], raw[2]
            start_score = torch.sigmoid(start_logits.squeeze(-1))
            end_score = torch.sigmoid(end_logits.squeeze(-1))
            output = F.softmax(cls_logits, dim=-1)
        elif args.backbone == 'BSv2':
            cls_logits = raw[0]
            start_logits2 = raw[7]
            end_logits2 = raw[8]
            start_score = torch.sigmoid(start_logits2.squeeze(-1))
            end_score = torch.sigmoid(end_logits2.squeeze(-1))
            output = F.softmax(cls_logits, dim=-1)
        elif args.backbone == 'STFM':
            cls_logits = raw[0]
            start_from_field = raw[4]
            end_from_field = raw[5]
            start_score = F.pad(start_from_field, (1, 0))
            end_score = F.pad(end_from_field, (0, 1))
            output = F.softmax(cls_logits, dim=-1)
        elif args.backbone == 'CDED':
            cls_logits, start_logits, end_logits, loc_pred = raw
            start_score = torch.sigmoid(start_logits.squeeze(-1))
            end_score = torch.sigmoid(end_logits.squeeze(-1))
            output = F.softmax(cls_logits, dim=-1)
        else:
            output = F.softmax(raw, dim=-1)

        for idx, file in enumerate(sample_name):
            proposal[file] = []
            with open(os.path.join(sp, file), 'a') as f:
                pred_bs, gt_bs = output[idx], target[idx]
                pred_fs = torch.argmax(pred_bs, dim=1)
                if args.backbone == 'CDED':
                    ss = start_score[idx].unsqueeze(1)
                    es = end_score[idx].unsqueeze(1)
                    ds = loc_pred[idx, :, 0].unsqueeze(1)
                    de = loc_pred[idx, :, 1].unsqueeze(1)
                    results = torch.cat(
                        (pred_fs.unsqueeze(1), gt_bs.unsqueeze(1), pred_bs, ss, es, ds, de), dim=1)
                elif args.backbone in ('BoundaryReg', 'DSTEAux', 'CausalDSTEAux', 'CausalDSTEError',
                                       'CausalDSTEAFCM', 'CausalDSTEGate', 'BSDSTE', 'BSv2', 'STFM'):
                    ss = start_score[idx].unsqueeze(1)
                    es = end_score[idx].unsqueeze(1)
                    results = torch.cat(
                        (pred_fs.unsqueeze(1), gt_bs.unsqueeze(1), pred_bs, ss, es), dim=1)
                else:
                    results = torch.cat(
                        (pred_fs.unsqueeze(1), gt_bs.unsqueeze(1), pred_bs), dim=1)
                for result in results:
                    s = ','.join(map(str, result.tolist())) + '\n'
                    f.write(s)

    print('========== mask matrix thereshold =', thereshold)
    for file in tqdm(os.listdir(sp)):
        with open(sp + file, 'r') as f:
            data = [u.lstrip().rstrip().split(',') for u in f.readlines()]
            data = np.array(data, dtype=np.float32)
        pb_matrix = data[:, 2:54].T
        mask_matrix = (pb_matrix > thereshold).astype(int)
        if args.backbone == 'BoundaryReg':
            start_scores = data[:, 54]
            end_scores = data[:, 55]
        elif args.backbone == 'CDED':
            d_start_pred = data[:, 56]
            d_end_pred = data[:, 57]
        for i in range(1, mask_matrix.shape[0]):
            pro_ = get_proposal(mask_matrix[i])
            if args.backbone == 'BoundaryReg' and len(pro_) > 0:
                k = args.snap_k
                T_len = data.shape[0]
                refined = []
                for u, v in pro_:
                    s_lo = max(0, u - k)
                    s_hi = min(T_len, u + k + 1)
                    new_u = s_lo + int(np.argmax(start_scores[s_lo:s_hi]))
                    e_lo = max(0, v - k)
                    e_hi = min(T_len, v + k + 1)
                    new_v = e_lo + int(np.argmax(end_scores[e_lo:e_hi]))
                    if new_u >= new_v:
                        new_u, new_v = u, v
                    refined.append([new_u, new_v])
                pro_ = refined
            elif args.backbone == 'CDED' and len(pro_) > 0:
                T_len = data.shape[0]
                refined = []
                for u, v in pro_:
                    new_u = int(round(u - d_start_pred[u]))
                    v_idx = min(v, T_len - 1)
                    new_v = int(round(v + d_end_pred[v_idx]))
                    new_u = max(0, min(new_u, T_len - 1))
                    new_v = max(new_u + 1, min(new_v, T_len))
                    refined.append([new_u, new_v])
                pro_ = refined
            proposal[file] += [[i, u, v, np.mean(pb_matrix[i][u:v])] for u, v in pro_]

    detect_result_dir = os.path.join(base, 'detect_result')
    os.makedirs(detect_result_dir, exist_ok=True)
    for k, v in proposal.items():
        with open(os.path.join(detect_result_dir, k), 'a') as ff:
            s = ''
            for lb, st, ed, score in v:
                s += str(int(lb)) + ',' + str(int(st)) + ',' + str(int(ed)) + ',' + str(score) + '\n'
            ff.write(s)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if epoch in args.schedule:
        print(f'[LR] epoch {epoch}: lr → {lr}')


def accuracy(output, target, topk=(1,), ignore=-1):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        if ignore != -1:
            if ignore == 0:
                mask = target != 0
            elif ignore == 1:
                mask = target == 0
            output = output[mask, :]
            target = target[mask]
            batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_proposal(brr):
    arr = np.append(brr, 0)
    proposal, start = [], None
    for i in range(arr.shape[0]):
        if arr[i] == 1:
            if start is None:
                start = i
        elif start is not None:
            proposal.append([start, i])
            start = None
    return proposal


def temporal_nms(actions, iou_threshold):
    if len(actions) == 0:
        return []
    actions = np.array(actions, dtype=np.float32)
    starts = actions[:, 1].astype(np.int32)
    ends = actions[:, 2].astype(np.int32)
    scores = actions[:, 3].astype(float)
    area = ends - starts
    indices = np.argsort(scores)[::-1]
    keep = []
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        tt1 = np.maximum(starts[i], starts[indices[1:]])
        tt2 = np.minimum(ends[i], ends[indices[1:]])
        intersection = np.maximum(0.0, tt2 - tt1)
        iou = intersection / (area[i] + area[indices[1:]] - intersection)
        remaining = np.where(iou <= iou_threshold)[0]
        indices = indices[remaining + 1]
    return actions[keep].tolist()


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    main()

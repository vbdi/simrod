import argparse
import os
import statistics
os.environ['QT_QPA_PLATFORM']='offscreen'
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf, open_dict
import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from models.yolov4 import Model as Yolov4
from utils.datasets import *
from utils.utils import *
from pathlib import Path
import yaml
import re
import glob


mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

USERNAME = os.environ['USER']

RUNS_DIR = "{}/runs".format(os.environ['GUILD_HOME']) if os.environ['GUILD_HOME'] else "/home/{}/tmp_guild/runs".format(USERNAME)

# Hyperparameters
hyp = {'optimizer': 'SGD',  # ['adam', 'SGD', None] if none, default is SGD
       'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
       'momentum': 0.937,  # SGD momentum/Adam beta1
       'weight_decay': 5e-4,  # optimizer weight decay
       'giou': 0.05,  # giou loss gain
       'cls': 0.58,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'anchor_t': 4.0,  # anchor-multiple threshold
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.014,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.68,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 0.0,  # image rotation (+/- deg)
       'translate': 0.0,  # image translation (+/- fraction)
       'scale': 0.5,  # image scale (+/- gain)
       'shear': 0.0}  # image shear (+/- deg)



def train(hyp, data_dict):
    print(f'Hyperparameters {hyp}')
    log_dir = tb_writer.log_dir if tb_writer else 'runs/evolution'  # run directory
    print('log_dir is {}'.format(log_dir))
    wdir = str(Path(log_dir) / 'weights') + os.sep  # weights directory

    os.makedirs(wdir, exist_ok=True)
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'
    results_file = log_dir + os.sep + 'results.txt'

    # Save run settings
    with open(Path(log_dir) / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f)
    with open(Path(log_dir) / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f)

    epochs = opt.epochs  # 300
    batch_size = opt.batch_size  # 64
    weights = opt.weights  # initial training weights
    learning_rate = (opt.lr * batch_size / 16) if opt.lr else hyp['lr0']
    print('==> learning rate: {}'.format(learning_rate))

    # Configure
    init_seeds(1)

    train_path = data_dict['train']
    test_path = data_dict['val']
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)

    # Create
    if "v4" in opt.cfg:
        model = Yolov4(opt.cfg, ch=3, nc=nc).to(device)
    else:
        model = Model(opt.cfg, nc=nc).to(device)


    # Image sizes
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay


    # Select trainable variables
    if opt.train_layers == 'bn_only':
        pg = []  # optimizer parameter groups
        n, rest = [], []
        for k, v in model.named_parameters():
            if '.bn' in k:
                pg.append(v)  # biases
                n.append(k)
                v.requires_grad = True
            else:
                rest.append(k)
                v.requires_grad = False

        optimizer = optim.Adam(pg, lr=learning_rate) if opt.adam else \
            optim.SGD(pg, lr=learning_rate, momentum=hyp['momentum'], nesterov=True)

        print('Optimizer groups: %g .bn' % (len(pg)))
        del pg
    else:
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        n0, n1, n2 = [], [], []
        for k, v in model.named_parameters():
            if v.requires_grad:
                if '.bias' in k:
                    pg2.append(v)  # biases
                    n2.append(k)
                elif '.weight' in k and '.bn' not in k:
                    pg1.append(v)  # apply weight decay
                    n1.append(k)
                else:
                    pg0.append(v)  # all else
                    n0.append(k)

        optimizer = optim.Adam(pg0, lr=learning_rate) if opt.adam else \
            optim.SGD(pg0, lr=learning_rate, momentum=hyp['momentum'], nesterov=True)

        optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs, save_dir=log_dir)

    # Load Model
    start_epoch, best_fitness = 0, 0.0
    if weights.endswith('.pt'):  # pytorch format
        ckpt = torch.load(weights, map_location=device)  # load checkpoint

        # load model
        try:
            ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                             if model.state_dict()[k].shape == v.shape}  # to FP32, filter
            model.load_state_dict(ckpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. This may be due to model differences or %s may be out of date. " \
                "Please delete or update %s and try again, or use --weights '' to train from scratch." \
                % (opt.weights, opt.cfg, opt.weights, opt.weights)
            raise KeyError(s) from e

        # TODO: add flag to load optimizer or NOT
        # if ckpt['optimizer'] is not None:
        #     optimizer.load_state_dict(ckpt['optimizer'])
        #     best_fitness = ckpt['best_fitness']

        # load results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # epochs
        start_epoch = ckpt['epoch'] + 1
        if epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (opt.weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    if "v4" in opt.cfg:
        rank = -1
        opt.local_rank = -1
        opt.sync_bn = False
        cuda = device.type != 'cpu'
        if cuda and rank == -1 and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        # SyncBatchNorm
        if opt.sync_bn and cuda and rank != -1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            print('Using SyncBatchNorm()')

        # Exponential moving average
        ema = torch_utils.ModelEMA(model) if rank in [-1, 0] else None

        # DDP mode
        if cuda and rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=(opt.local_rank))
    else:
        # Distributed training
        if device.type != 'cpu' and torch.cuda.device_count() > 1 and dist.is_available():
            dist.init_process_group(backend='nccl',  # distributed backend
                                    init_method='tcp://127.0.0.1:{}'.format(opt.ddp_port),  # init method
                                    world_size=1,  # number of nodes
                                    rank=0)  # node rank
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters = True)

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp,
                                            augment=opt.augment, nomosaic=opt.nomosaic,
                                            augmix=opt.augmix,
                                            cache=opt.cache_images, rect=opt.rect, shuffle=opt.shuffle)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Correct your labels or your model.' % (mlc, nc, opt.cfg)

    # Testloader
    testloader = create_dataloader(test_path, imgsz_test, batch_size, gs, opt,
                                   hyp=hyp, augment=False, cache=opt.cache_images, rect=True)[0]

    # Model parameters
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model.names = names

    # Class frequency
    labels = np.concatenate(dataset.labels, 0)
    c = torch.tensor(labels[:, 0])  # classes
    # cf = torch.bincount(c.long(), minlength=nc) + 1.
    # model._initialize_biases(cf.to(device))
    # plot_labels(labels, save_dir=log_dir)
    if tb_writer:
        tb_writer.add_histogram('classes', c, 0)

    # Check anchors
    if not opt.noautoanchor:
        check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # Exponential moving average
    if not "v4" in opt.cfg:
        ema = torch_utils.ModelEMA(model)

    # Start training
    t0 = time.time()
    nw = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  # do not move
    print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
    print('Using %g dataloader workers' % dataloader.num_workers)
    print('Starting training for %g epochs...' % epochs)
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            # Fast dev run, only run for 10 steps
            if opt.fast and i == 10:
                break
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            pred = model(imgs)

            # Loss
            loss, loss_items = compute_loss(pred, targets.to(device), model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Backward
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)

            # Plot
            if ni < 3:
                f = str(Path(log_dir) / ('train_batch%g.jpg' % ni))  # filename
                result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                if tb_writer and result is not None:
                    tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()

        # mAP
        ema.update_attr(model, include=['md', 'nc', 'hyp', 'gr', 'names', 'stride'])
        final_epoch = epoch + 1 == epochs
        if not opt.notest or final_epoch:  # Calculate mAP
            results, maps, times = test.test(opt.data,
                                             opt,
                                             batch_size=batch_size,
                                             imgsz=imgsz_test,
                                             save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                                             model=ema.ema,
                                             single_cls=opt.single_cls,
                                             dataloader=testloader,
                                             save_dir=log_dir,
                                             fast=opt.fast)

        # Write
        with open(results_file, 'a') as f:
            f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(opt.name) and opt.bucket:
            os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

        # Tensorboard
        if tb_writer:
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        save = (not opt.nosave) or (final_epoch and not opt.evolve)
        if save:
            with open(results_file, 'r') as f:  # create checkpoint
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': f.read(),
                        'model': ema.ema,
                        'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last, best and delete
            torch.save(ckpt, last)
            if (best_fitness == fi) and not final_epoch:
                torch.save(ckpt, best)
            del ckpt


        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    # Strip optimizers
    n = ('_' if len(opt.name) and not opt.name.isnumeric() else '') + opt.name
    fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
    for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
        if os.path.exists(f1):
            os.rename(f1, f2)  # rename
            ispt = f2.endswith('.pt')  # is *.pt
            strip_optimizer(f2) if ispt else None  # strip optimizer
            os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload

    # print('%g epochs completed in %.3f hours.\n' % (opt.epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if device.type != 'cpu' and torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    # check_git_status()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/city5m.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/cityscapes_source.yml', help='data.yaml path')
    parser.add_argument('--runs-dir', type=str, default='', help='defines the RUNS_DIR')
    parser.add_argument('--ckpt-run-id', type=str, default='', help='*.checkpoint run id')
    parser.add_argument('--pseudo-run-id', type=str, default='', help='run id for pseudo-labels')
    parser.add_argument('--resume-run-id', type=str, default='', help='resume run id')
    parser.add_argument('--split', type=str, default='', help='defines the split of the dataset')
    parser.add_argument('--labeled-ratio', type=float, default=1.0,
                        help='Ratio of ground-truth labels for corrupted image set')
    parser.add_argument('--hyp', type=str, default='', help='hyp.yaml path (optional)')
    parser.add_argument('--corruption', type=str, default='contrast', help='Corruption type')
    parser.add_argument('--severity', type=str, default='3', help='Corrupt severity level for test')
    parser.add_argument('--test-severity', type=int, default=0, help='Corrupt severity level for test')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--train-layers', type=str, default='', help='Trainable layers')
    parser.add_argument('--ddp-port', type=int, default=9999)
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--augment', action='store_true', help='Augment images and labels')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--nomosaic', action='store_true', help='No not apply mosaic augmentation')
    parser.add_argument('--augmix', action='store_true', help='Apply Augmix augmentation')
    parser.add_argument('--resume', nargs='?', const='get_last', default=False,
                        help='resume from given path/to/last.pt, or most recent run if blank.')
    parser.add_argument('--shuffle', action='store_true', help='rectangular training')
    parser.add_argument('--fast', action='store_true', help='trains only for few steps per epoch for debugging')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='2,3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50 percent')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--result-dir', default='', help='Desired output folder, will override the one specified in the yml file')

    opt = parser.parse_args()

    if opt.runs_dir is not None and opt.runs_dir != '' and opt.runs_dir != 'none':
        RUNS_DIR = opt.runs_dir


    last = get_latest_run(opt, search_dir=os.path.join(RUNS_DIR, opt.resume_run_id)) if opt.resume == 'get_last' else opt.resume
    # resume from most recent run
    if last and not opt.weights:
        print(f'Resuming training from {last}')
    opt.weights = last if opt.resume and not opt.weights else opt.weights

    # Use weights from ckpt_run_id if available and if it exists
    if opt.ckpt_run_id and opt.ckpt_run_id != '' and opt.ckpt_run_id != 'none':
        weights_path = os.path.join(RUNS_DIR, opt.ckpt_run_id, "outputs/best.pt")
        weights_alt_path = os.path.join(RUNS_DIR, opt.ckpt_run_id, "outputs/weights/best.pt")
        weights_alt_path_2 = os.path.join(RUNS_DIR, opt.ckpt_run_id, "weights/best.pt")
        if os.path.exists(weights_path):
            opt.weights = weights_path
        elif os.path.exists(weights_alt_path):
            opt.weights = weights_alt_path
        elif os.path.exists(weights_alt_path_2):
            opt.weights = weights_alt_path_2
        else:
            raise ValueError("No ckpt found in {} nor in {} nor in {}".format(weights_path, weights_alt_path, weights_alt_path_2))
    print("opt.weights: {}".format(opt.weights))

    opt.cfg = check_file(opt.cfg)  # check file
    print("checked opt.cfg")
    opt.data = check_file(opt.data)  # check file
    print("checked opt.data")
    if opt.hyp:  # update hyps
        opt.hyp = check_file(opt.hyp)  # check file
        with open(opt.hyp) as f:
            hyp.update(yaml.load(f))  # update hyps
    print(opt)
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False

    # with open(opt.data) as f:
    #     data_cfg = yaml.load(f, Loader=yaml.FullLoader)
    data_cfg = OmegaConf.load(opt.data)
    with open_dict(data_cfg) as d:
        d.split = opt.split
        d.corruption = opt.corruption
        d.severity = opt.severity
        d.test.severity = opt.test_severity
        d.pseudo_run_id = opt.pseudo_run_id
        d.ckpt_run_id = opt.ckpt_run_id
        if opt.result_dir is not None and opt.result_dir != '':
            d.result_dir = opt.result_dir


        # semi-supervised case: set labeled_ratio config in voc_pseudo_ssl.yaml
        if "pseudo_ssl" in opt.data:
            d.labeled_ratio = opt.labeled_ratio

    result_dir = data_cfg['result_dir'] # os.getcwd()
    print("result_dir: {}".format(result_dir))

    # Train
    tb_writer = SummaryWriter(log_dir=result_dir, comment=opt.name)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    train(hyp, data_cfg)

    # Get best weights
    weights = os.path.join(result_dir, "weights/best.pt")


    # Test on clean domain
    clean_path = data_cfg.clean_test
    nc, names = int(data_cfg['nc']), data_cfg['names']  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)
    data = {"nc": nc, "test": clean_path}
    print("===> Testing on clean domain: {}".format(clean_path))
    opt.task = 'test'
    clean_results = test.test(data, opt,
                         weights,
                         opt.batch_size,
                         opt.img_size,
                         opt.conf_thres,
                         opt.iou_thres)
    print("Clean - {}: {}".format("clean_P", 100.0 * clean_results[0][0]))
    print("Clean - {}: {}".format("clean_R", 100.0 * clean_results[0][1]))
    print("Clean - {}: {}".format("clean_AP50", 100.0 * clean_results[0][2]))
    print("Clean - {}: {}".format("clean_AP", 100.0 * clean_results[0][3]))


    # Test on corrupted domain
    corrupt_paths = glob.glob(data_cfg.corrupt_test)
    if corrupt_paths:
        p_list = []
        r_list = []
        ap50_list = []
        ap_list = []
        rp_list = []
        rr_list = []
        rpc_50_list = []
        rpc_list = []
        for p in corrupt_paths:
            print("===> Testing on data from {}".format(p))
            data = {"nc": nc, "test": p}
            corrupt_results = test.test(data, opt,
                                        weights,
                                        opt.batch_size,
                                        opt.img_size,
                                        opt.conf_thres,
                                        opt.iou_thres)

            corrupt_P = 100.0 * corrupt_results[0][0]
            p_list.append(corrupt_P)
            corrupt_R = 100.0 * corrupt_results[0][1]
            r_list.append(corrupt_R)
            corrupt_AP50 = 100.0 * corrupt_results[0][2]
            ap50_list.append(corrupt_AP50)
            corrupt_AP = 100.0 * corrupt_results[0][3]
            ap_list.append(corrupt_AP)
            rP = corrupt_results[0][0] * 100.0 / clean_results[0][0]
            rp_list.append(rP)
            rR = corrupt_results[0][1] * 100.0 / clean_results[0][1]
            rr_list.append(rR)
            rPC_50 = corrupt_results[0][2] * 100.0 / clean_results[0][2]
            rpc_50_list.append(rPC_50)
            rPC = corrupt_results[0][3] * 100.0 / clean_results[0][3]
            rpc_list.append(rPC)

            corrupt_name = os.path.basename(p)
            print("Corrupt {} - {}: {}".format(corrupt_name, "corrupt_P", corrupt_P))
            print("Corrupt {} - {}: {}".format(corrupt_name, "corrupt_R", corrupt_R))
            print("Corrupt {} - {}: {}".format(corrupt_name, "corrupt_AP50", corrupt_AP50))
            print("Corrupt {} - {}: {}".format(corrupt_name, "corrupt_AP", corrupt_AP))

            print("Corrupt {} - {}: {}".format(corrupt_name, "rP", rP))
            print("Corrupt {} - {}: {}".format(corrupt_name, "rR", rR))
            print("Corrupt {} - {}: {}".format(corrupt_name, "rPC_50", rPC_50))
            print("Corrupt {} - {}: {}".format(corrupt_name, "rPC", rPC))
        # calculate the average
        avg_P = statistics.mean(p_list)
        avg_R = statistics.mean(r_list)
        avg_AP50 = statistics.mean(ap50_list)
        avg_AP = statistics.mean(ap_list)
        avg_rP = statistics.mean(rp_list)
        avg_rR = statistics.mean(rr_list)
        avg_rPC_50 = statistics.mean(rpc_50_list)
        avg_rPC = statistics.mean(rpc_list)

        print('Corrupt - {}: {}'.format("corrupt_P", avg_P))
        print('Corrupt - {}: {}'.format("corrupt_R", avg_R))
        print('Corrupt - {}: {}'.format("corrupt_AP50", avg_AP50))
        print('Corrupt - {}: {}'.format("corrupt_AP", avg_AP))

        print('Corrupt - {}: {}'.format("rP", avg_rP))
        print('Corrupt - {}: {}'.format("rR", avg_rR))
        print('Corrupt - {}: {}'.format("rPC_50", avg_rPC_50))
        print('Corrupt - {}: {}'.format("rPC", avg_rPC))
    else:
        print("No data_cfg.corrupt_test found. Not running test on corruption")





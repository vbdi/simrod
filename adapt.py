# step 1: generate pseudo-labels from the baseline model
# step 2: train a new model with the pseudolabels from step 1 combined with ground-truths from the source domain data best.pt
# step 3: generate refined pseudo-labels (i.e. use the model from step 2 to label target domain data again and get the pseudo-labels)
# step 4: use the model from step 2 as initial checkpoint, finetune it with the refined pseudolabel from step 3

import argparse
import re

from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf, open_dict
from utils.datasets import *
from utils.utils import *
import label as labeler
import trainer as trainer
import evaluate as tester


mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed


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


def adapt(data_cfg, opt):
    # step 1: generate pseudo-labels from the baseline model
    print("opt.teacher_adapt: {}".format(opt.teacher_adapt))
    if opt.teacher_adapt:
        # ckpt_run_id should be different with pseudo_run_id in teacher distillation
        assert opt.ckpt_run_id != opt.pseudo_run_id
    else:
        # ckpt_run_id should be the same as pseudo_run_id in normal proposed
        assert opt.ckpt_run_id == opt.pseudo_run_id
    opt.conf_thres = 0.4
    opt.iou_thres = 0.65
    opt.augment = False
    step_1_source = data_cfg['step_1']['source']
    step_1_weights = data_cfg['step_1']['weights']
    step_1_has_extra = data_cfg['step_1']['extra_source'] and data_cfg['step_1']['extra_labels_dir'] and data_cfg['step_1']['extra_predictions_dir']
    print("\n============== START STEP: Generating pseudolabels ==============")
    with torch.no_grad():
        labeler.generate_labels(step_1_source, step_1_weights, opt, step_1_labels_dir, step_1_predictions_dir, opt.label_img_size)
        if step_1_has_extra:
            labeler.generate_labels(data_cfg['step_1']['extra_source'], step_1_weights, opt, step_1_extra_labels_dir, step_1_extra_predictions_dir,
                                    opt.label_img_size)

    # step 2: train a new model with the pseudolabels from step 1 combined with ground-truths from the source domain data best.pt
    # This is the bn_only train
    step2_tb_writer = SummaryWriter(log_dir=step_2_result_dir, comment=opt.name)
    print("\n============== START STEP: Gradual adaptation of BN layers ==============")
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    opt.train_layers = 'bn_only'
    opt.augment = True
    opt.nomosaic = False
    opt.conf_thres = 0.001
    opt.iou_thres = 0.65
    # calculate epochs to be trained
    ckpt = torch.load(data_cfg['step_2']['weights'], map_location=device)  # load checkpoint
    opt.epochs = ckpt['epoch'] + opt.epochs + 1  # add 1 because the start ckpt is ckpt['epochs'] + 1
    print('Target data for step 2 is {}'.format(data_cfg['step_2']['train'][1]))
    trainer.train(data_cfg['step_2']['train'], data_cfg['step_2']['weights'], hyp, data_cfg, opt, step2_tb_writer, data_cfg['step_2']['result_dir'], device)

    # Get best weights from step 2, use it as step 3's initial weights
    step_3_weights = data_dict['step_3']['weights']

    # step 3: generate refined pseudo-labels (i.e. use the model from step 2 to label target domain data again and get the pseudo-labels)
    if not opt.teacher_adapt:
        assert step_3_weights == os.path.join(step_2_result_dir, "weights/best.pt")
        opt.conf_thres = 0.4
        opt.iou_thres = 0.65
        opt.augment = False
        step_3_source = data_cfg['step_3']['source']
        step_3_has_extra = data_cfg['step_3']['extra_source'] and data_cfg['step_3']['extra_labels_dir'] and \
                           data_cfg['step_3']['extra_predictions_dir']

        print("\n============== START STEP: Refining pseudolabels ==============")
        with torch.no_grad():
            labeler.generate_labels(step_3_source, step_3_weights, opt, step_3_labels_dir, step_3_predictions_dir,
                                    opt.label_img_size)
            if step_3_has_extra:
                labeler.generate_labels(data_cfg['step_3']['extra_source'], step_3_weights, opt,
                                        step_3_extra_labels_dir, step_3_extra_predictions_dir,
                                        opt.label_img_size)


    # step 4: use the model from step 2 as initial checkpoint, finetune it with the refined pseudolabel
    final_tb_writer = SummaryWriter(log_dir=step_4_result_dir, comment=opt.name)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')

    opt.train_layers = ''
    if opt.step_4_lr:
        opt.lr = opt.step_4_lr
    opt.augment = True
    opt.nomosaic = False
    opt.conf_thres = 0.001
    opt.iou_thres = 0.65
    # calculate epochs to be finetuned
    ckpt = torch.load(data_cfg['step_4']['weights'], map_location=device)  # load checkpoint
    opt.epochs = ckpt['epoch'] + opt.finetune_epochs + 1
    if opt.teacher_adapt:
        # if distillate from teacher model, change the target data to be step-1's pseudolabels
        step_4_train_size = len(data_cfg['step_4']['train'])
        if (step_4_train_size >= 2):
            for i in range(step_4_train_size):
                data_cfg['step_4']['train'][i] = data_cfg['step_2']['train'][i]
    print("\n============== START STEP: Finetuning all layers ==============")
    trainer.train(data_cfg['step_4']['train'], data_cfg['step_4']['weights'], hyp, data_cfg, opt, final_tb_writer, data_cfg['step_4']['result_dir'], device)

    # Get best weights
    final_weights = os.path.join(step_4_result_dir, "weights/best.pt")
    # Test
    tester.test(data_cfg, opt, final_weights, output_dir=step_4_result_dir, copyback_dir=data_cfg['step_4']['result_dir'])

    print("\n==========================Adaptation Completed==============================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Train parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/city5x.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/cityscapes_adapt.yml', help='data.yaml path')
    parser.add_argument('--ckpt-run-id', type=str, default='', help='*.checkpoint run id; this defines initial checkpoint')
    parser.add_argument('--pseudo-run-id', type=str, default='', help='run id for pseudo-labels')
    parser.add_argument('--output-run-id', type=str, default='', help='output id in RUNS_DIR for the current job; '
                                                                      'mutually exclusive with result_dir')
    parser.add_argument('--teacher-adapt', action='store_true', help='set to true when proposed with teacher distillation')
    parser.add_argument('--labeled-ratio', type=float, default=1.0,
                        help='Ratio of ground-truth labels for corrupted image set')
    parser.add_argument('--hyp', type=str, default='', help='hyp.yaml path (optional)')
    parser.add_argument('--corruption', type=str, default='contrast', help='Corruption type')
    parser.add_argument('--severity', type=str, default='3', help='Corrupt severity level for test')
    parser.add_argument('--test-severity', type=int, default=0, help='Corrupt severity level for test')
    parser.add_argument('--pseudo-dataset', type=str, default='', help='the dataset that defines pseudo-run-id')
    parser.add_argument('--ckpt-dataset', type=str, default='', help='the dataset that defines ckpt-run-id')
    parser.add_argument('--epochs', type=int, default=300, help='Step 2 training epochs')
    parser.add_argument('--finetune-epochs', type=int, default=100, help='Step 4 finetune epochs')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--step-4-lr', type=float, default=None)
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
    parser.add_argument('--shuffle', action='store_true', help='rectangular training')
    parser.add_argument('--nouniformloss', action='store_true', help='disable uniform weights for loss functions')
    parser.add_argument('--fast', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50 percent')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--result-dir', default='', help='desired output folder; '
                                                         'mutually exclusive with output_run_id')
    parser.add_argument('--year', type=int, default=2012, help='Corrupt severity level for test')
    parser.add_argument('--split', type=str, default='', help='defines the split of the dataset')

    # Label parameters
    parser.add_argument('--label-img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--confidence', action='store_true', help='save confidence score')
    parser.add_argument('--save', action='store_true', help='save images and predictions')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--display', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--models', type=str, default='',
                        help='model used for training, should be one of yolov5s, yolov5m, yolov5x')
    opt = parser.parse_args()
    print(opt)

    opt.cfg = check_file(opt.cfg)  # check file
    print("checked opt.cfg")
    opt.data = check_file(opt.data)  # check file
    print("checked opt.data")
    if opt.hyp:  # update hyps
        opt.hyp = check_file(opt.hyp)  # check file
        with open(opt.hyp) as f:
            hyp.update(yaml.load(f))  # update hyps
    # print(opt)
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False

    data_dict = OmegaConf.load(opt.data)
    with open_dict(data_dict) as d:
        d.split = opt.split
        d.corruption = opt.corruption
        d.severity = opt.severity
        d.test.severity = opt.test_severity
        if opt.pseudo_dataset is not None and opt.pseudo_dataset != '' and opt.pseudo_dataset != 'none':
            d.pseudo_dataset = opt.pseudo_dataset
        if opt.ckpt_dataset is not None and opt.ckpt_dataset != '' and opt.ckpt_dataset != 'none':
            d.ckpt_dataset = opt.ckpt_dataset
        if opt.ckpt_run_id is not None and opt.ckpt_run_id != '' and opt.ckpt_run_id != 'none':
            d.ckpt_run_id = opt.ckpt_run_id
        if opt.pseudo_run_id is not None and opt.pseudo_run_id != '' and opt.pseudo_run_id != 'none':
            d.pseudo_run_id = opt.pseudo_run_id
        if opt.output_run_id is not None and opt.output_run_id != '' and opt.output_run_id != 'none':
            d.output_run_id = opt.output_run_id
        if opt.result_dir is not None and opt.result_dir != '' and opt.result_dir != 'none':
            d.result_dir = opt.result_dir
        d.year = opt.year

        # semi-supervised case: set labeled_ratio config in voc_pseudo_ssl.yaml
        if "pseudo_ssl" in opt.data:
            d.labeled_ratio = opt.labeled_ratio

    step_2_result_dir = data_dict['step_2']['result_dir']
    step_4_result_dir = data_dict['step_4']['result_dir']
    step_1_labels_dir = data_dict['step_1']['labels_dir']
    step_1_predictions_dir = data_dict['step_1']['predictions_dir']
    if data_dict['step_1']['extra_labels_dir']:
        step_1_extra_labels_dir = data_dict['step_1']['extra_labels_dir']
    if data_dict['step_1']['extra_predictions_dir']:
        step_1_extra_predictions_dir = data_dict['step_1']['extra_predictions_dir']
    if not opt.teacher_adapt:
        step_3_labels_dir = data_dict['step_3']['labels_dir']
        step_3_predictions_dir = data_dict['step_3']['predictions_dir']
        if data_dict['step_3']['extra_labels_dir']:
            step_3_extra_labels_dir = data_dict['step_3']['extra_labels_dir']
        if data_dict['step_3']['extra_predictions_dir']:
            step_3_extra_predictions_dir = data_dict['step_3']['extra_predictions_dir']
    print("step_2_result_dir: {}".format(step_2_result_dir))
    print("step_4_result_dir: {}".format(step_4_result_dir))
    print("step_1_labels_dir: {}".format(step_1_labels_dir))
    print("step_1_predictions_dir: {}".format(step_1_predictions_dir))
    if not opt.teacher_adapt:
        print("step_3_labels_dir: {}".format(step_3_labels_dir))
        print("step_3_predictions_dir: {}".format(step_3_predictions_dir))

    adapt(data_dict, opt)
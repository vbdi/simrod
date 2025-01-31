import argparse
import json

from models.experimental import *
from utils.datasets import *
from omegaconf import OmegaConf, open_dict
from utils import torch_utils
from models.experimental import attempt_load

import glob
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm


USERNAME = os.environ['USER']

RUNS_DIR = "{}/runs".format(os.environ['GUILD_HOME']) if os.environ['GUILD_HOME'] else "/home/{}/tmp_guild/runs".format(USERNAME)

def test(data,
         opt,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir='',
         merge=False,
         fast=False):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by trainer.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = torch_utils.select_device(opt.device, batch_size=batch_size)
        merge = opt.merge  # use Merge NMS

        # Remove previous
        for f in glob.glob(str(Path(save_dir) / 'test_batch*.jpg')):
            os.remove(f)

        model = attempt_load(weights, map_location=device)  # load FP32 model
        if isinstance(imgsz, list):
            imgsz = imgsz[0]
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str) and (data.endswith('yml') or data.endswith('yaml')):
        with open(data) as f:
            data = yaml.safe_load(f) #, Loader=yaml.FullLoader)  # model dict
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt,
                                       hyp=None, augment=False, cache=False, pad=0.5, rect=True)[0]

    seen = 0
    names = model.names if hasattr(model, 'names') else model.module.names
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        if fast and batch_i == 10:
            break
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = torch_utils.time_synchronized()
            inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += torch_utils.time_synchronized() - t

            # Compute loss
            if training:  # if model has loss hyperparameters
                loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # GIoU, obj, cls

            # Run NMS
            t = torch_utils.time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge)
            t1 += torch_utils.time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if batch_i < 1:
            f = Path(save_dir) / ('test_batch%g_gt.jpg' % batch_i)  # filename
            plot_images(img, targets, paths, str(f), names)  # ground truth
            f = Path(save_dir) / ('test_batch%g_pred.jpg' % batch_i)
            plot_images(img, output_to_target(output, width, height), paths, str(f), names)  # predictions

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        print('ap shape is: {}'.format(ap.shape))
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if type(names).__name__ == 'ListConfig':
        names = OmegaConf.to_container(names, resolve=True)
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and map50 and len(jdict):
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
        f = 'detections_val2017_%s_results.json' % \
            (weights.split(os.sep)[-1].replace('.pt', '') if isinstance(weights, str) else '')  # filename
        print('\nCOCO mAP with pycocotools... saving %s...' % f)
        with open(f, 'w') as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes(f)  # initialize COCO pred api

            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # image IDs to evaluate
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except:
            print('WARNING: pycocotools must be installed with numpy==1.17 to run correctly. '
                  'See https://github.com/cocodataset/cocoapi/issues/356')

    # Return results
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    # parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--ckpt-run-id', type=str, required=True, help='*.checkpoint run id')
    parser.add_argument('--data', type=str, default='data/pascal-test.yml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--runs-dir', type=str, default='', help='defines the RUNS_DIR')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='corrupt', help="'val', 'test', 'corrupt'")
    parser.add_argument('--corruption', type=str, default='contrast', help='Corruption type')
    parser.add_argument('--severity', type=int, default=0, help='Corrupt severity level for test')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    opt = parser.parse_args()
    # opt.save_json = opt.save_json or opt.data.endswith('coco.yaml')

    if opt.runs_dir is not None and opt.runs_dir != '' and opt.runs_dir != 'none':
        RUNS_DIR = opt.runs_dir

    # data_cfg = OmegaConf.load(opt.data)
    # with open_dict(data_cfg) as d:
    #     d.corruption = opt.corruption
    #     d.severity = opt.severity

    # Use weights from ckpt_run_id if available and if it exists
    weights_path = os.path.join(RUNS_DIR, opt.ckpt_run_id, "outputs/best.pt")
    weights_alt_path = os.path.join(RUNS_DIR, opt.ckpt_run_id, "outputs/weights/best.pt")
    weights_alt_path_2 = os.path.join(RUNS_DIR, opt.ckpt_run_id, "weights/best.pt")
    if os.path.exists(weights_path):
        weights = weights_path
    elif os.path.exists(weights_alt_path):
        weights = weights_alt_path
    elif os.path.exists(weights_alt_path_2):
        weights = weights_alt_path_2
    else:
        raise ValueError(
            "No ckpt found in {} nor in {} nor in {}".format(weights_path, weights_alt_path, weights_alt_path_2))

    # task = 'val', 'test', 'study'
    if opt.task in ['val', 'test']:  # (default) run normally
        opt.data = check_file(opt.data)  # check file
        print(opt)
        test(opt.data, opt,
             weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose)
    elif opt.task == 'corrupt':
        # Test on clean domain
        print("===> Testing on clean domain")
        data_dict = OmegaConf.load(opt.data)
        clean_path = data_dict['test']
        nc, names = int(data_dict['nc']), data_dict['names']  # number classes, names
        assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)
        data = {"nc": nc, "test": clean_path}
        opt.task = 'test'
        clean_results = test(data, opt, weights, opt.batch_size, opt.img_size, opt.conf_thres, opt.iou_thres,
                             opt.save_json, opt.single_cls, opt.augment, opt.verbose)
        print("Clean - {}: {}".format("clean_P", 100.0 * clean_results[0][0]))
        print("Clean - {}: {}".format("clean_R", 100.0 * clean_results[0][1]))
        print("Clean - {}: {}".format("clean_AP50", 100.0 * clean_results[0][2]))
        print("Clean - {}: {}".format("clean_AP", 100.0 * clean_results[0][3]))

        # Test on corrupted domain
        print("===> Testing on corrupted domain averaged over severity level")
        if opt.corruption:
            corrupt_path = '{}-{}-{}'.format(clean_path, opt.corruption, opt.severity)
            data = {"nc": nc, "test": corrupt_path}
            corrupt_results = test(data,
                                   opt,
                                   weights,
                                   opt.batch_size,
                                   opt.img_size,
                                   opt.conf_thres,
                                   opt.iou_thres,
                                   opt.save_json,
                                   opt.single_cls,
                                   opt.augment,
                                   opt.verbose)

            print("Corrupt - {}: {}".format("corrupt_P", 100.0 * corrupt_results[0][0]))
            print("Corrupt - {}: {}".format("corrupt_R", 100.0 * corrupt_results[0][1]))
            print("Corrupt - {}: {}".format("corrupt_AP50", 100.0 * corrupt_results[0][2]))
            print("Corrupt - {}: {}".format("corrupt_AP", 100.0 * corrupt_results[0][3]))

            print("Corrupt - {}: {}".format("rP", corrupt_results[0][0] * 100.0 / clean_results[0][0]))
            print("Corrupt - {}: {}".format("rR", corrupt_results[0][1] * 100.0 / clean_results[0][1]))
            print("Corrupt - {}: {}".format("rPC_50", corrupt_results[0][2] * 100.0 / clean_results[0][2]))
            print("Corrupt - {}: {}".format("rPC", corrupt_results[0][3] * 100.0 / clean_results[0][3]))


    else:
        raise ValueError("Unsupported task")

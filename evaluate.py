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
import test as tester  # import test.py to get mAP after each epoch
from utils.datasets import *
from utils.utils import *


mixed_precision = True
try:
    from apex import amp
except:
    mixed_precision = False  # not installed

USERNAME = os.environ['USER']

RUNS_DIR = "{}/runs".format(os.environ['GUILD_HOME']) if os.environ['GUILD_HOME'] else "/home/{}/tmp_guild/runs".format(USERNAME)


def test(data_cfg, opt, weights, output_dir=None, copyback_dir=None, verbose=True):

    # Test on clean domain
    clean_path = data_cfg.clean_test
    nc, names = int(data_cfg['nc']), data_cfg['names']  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)
    data = {"nc": nc, "test": clean_path}
    print("===> Testing on clean domain: {}".format(clean_path))
    opt.task = 'test'
    clean_results = tester.test(data, opt,
                              weights,
                              opt.batch_size,
                              opt.img_size,
                              opt.conf_thres,
                              opt.iou_thres,
                              verbose)
    clean_P = 100.0 * clean_results[0][0]
    clean_R = 100.0 * clean_results[0][1]
    clean_AP50 = 100.0 * clean_results[0][2]
    clean_AP = 100.0 * clean_results[0][3]
    print("Clean - {}: {}".format("clean_P", clean_P))
    print("Clean - {}: {}".format("clean_R", clean_R))
    print("Clean - {}: {}".format("clean_AP50", clean_AP50))
    print("Clean - {}: {}".format("clean_AP", clean_AP))

    if not data_cfg.corrupt_test or data_cfg.corrupt_test=='None' or data_cfg.corrupt_test=='none':
        corrupt_paths = None
    else:
        corrupt_paths = data_cfg.corrupt_test

    print('corrupt_paths: {}', corrupt_paths)
    if corrupt_paths and len(corrupt_paths) > 0:
        p_list = []
        r_list = []
        ap50_list = []
        ap_list = []
        rp_list = []
        rr_list = []
        rpc_50_list = []
        rpc_list = []

        if type(corrupt_paths).__name__ == 'ListConfig':
            corrupt_paths = OmegaConf.to_container(corrupt_paths, resolve=True)
        elif isinstance(corrupt_paths, list):
            corrupt_paths = corrupt_paths
        else:
            corrupt_paths = [corrupt_paths]

        for p in corrupt_paths:
            print("===> Testing on data from {}".format(p))
            data = {"nc": nc, "test": p}
            corrupt_results = tester.test(data, opt,
                                        weights,
                                        opt.batch_size,
                                        opt.img_size,
                                        opt.conf_thres,
                                        opt.iou_thres,
                                        verbose=verbose)

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/cross-test-20-nc.yml', help='data.yaml path')
    parser.add_argument('--corruption', type=str, default='comic', help='Corruption type')
    parser.add_argument('--severity', type=str, default='3', help='Corrupt severity level for test')
    parser.add_argument('--test-severity', type=int, default=0, help='Corrupt severity level for test')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--ckpt-run-id', type=str, help='*.checkpoint run id')
    parser.add_argument('--runs-dir', type=str, default='', help='defines the RUNS_DIR')
    parser.add_argument('--ddp-port', type=int, default=9999)
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    opt = parser.parse_args()

    if opt.runs_dir is not None and opt.runs_dir != '' and opt.runs_dir != 'none':
        RUNS_DIR = opt.runs_dir

    # Use weights from ckpt_run_id if available and if it exists
    if opt.ckpt_run_id and opt.ckpt_run_id != '' and opt.ckpt_run_id != 'none':
        weights_path = os.path.join(RUNS_DIR, opt.ckpt_run_id, "outputs/best.pt")
        weights_alt_path = os.path.join(RUNS_DIR, opt.ckpt_run_id, "outputs/weights/best.pt")
        weights_alt_path_2 = os.path.join(RUNS_DIR, opt.ckpt_run_id, "weights/best.pt")
        weights_alt_path_3 = os.path.join(RUNS_DIR, opt.ckpt_run_id, "best.pt")
        if os.file.exists(weights_path):
            opt.weights = weights_path
        elif os.file.exists(weights_alt_path) :
            opt.weights = weights_alt_path
        elif os.file.exists(weights_alt_path_2):
            opt.weights = weights_alt_path_2
        elif os.file.exists(weights_alt_path_3):
            opt.weights = weights_alt_path_3
        else:
            raise ValueError("No ckpt found in {} nor in {} nor in {} nor in {}".format(weights_path, weights_alt_path, weights_alt_path_2, weights_alt_path_3))
    print("opt.weights: {}".format(opt.weights))


    opt.data = check_file(opt.data)  # check file
    print(opt)
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False


    data_cfg = OmegaConf.load(opt.data)
    with open_dict(data_cfg) as d:
        d.corruption = opt.corruption
        d.severity = opt.severity
        d.test.severity = opt.test_severity

    test(data_cfg, opt, opt.weights, verbose=False)






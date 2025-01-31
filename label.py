import argparse
import os
import re
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from models.experimental import *

from utils.datasets import *
from utils.utils import *
from omegaconf import OmegaConf, open_dict

USERNAME = os.environ['USER']

RUNS_DIR = "{}/runs".format(os.environ['GUILD_HOME']) if os.environ['GUILD_HOME'] else "/home/{}/tmp_guild/runs".format(USERNAME)

def generate_labels(source, weights, opt, labels_dir, predictions_dir, img_size, save_img=False):

    view_img, imgsz = opt.display, img_size

    # source = data_dict['source']
    # labels_dir = data_dict['labels_dir']
    # predictions_dir = data_dict['predictions_dir']
    if opt.confidence:
        labels_dir += "-conf"
        predictions_dir += "-conf"

    webcam = False # source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(labels_dir):
        shutil.rmtree(labels_dir)  # delete output folder
    os.makedirs(labels_dir)  # make new output folder
    if os.path.exists(predictions_dir):
        shutil.rmtree(predictions_dir)  # delete output folder
    os.makedirs(predictions_dir)  # make new output folder

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = opt.save
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    # colors = [[242, 182, 177], [67, 155, 169], [77, 170, 137], [207, 252, 135], [9, 225, 83], [27, 88, 76],
    #           [63, 139, 93], [154, 237, 157], [64, 35, 58], [226, 33, 234], [189, 28, 147], [146, 11, 251],
    #           [83, 202, 55], [29, 31, 233], [11, 12, 208], [67, 155, 169], [78, 81, 120], [233, 168, 212], [114, 81, 160],
    #           [188, 73, 249]]
    # chair: [11, 12, 208] red
    # person: [64, 35, 58] dark
    # bicycle: [67, 155, 169] yellow
    # potted plant: [17, 3, 220] red

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in tqdm(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(predictions_dir) / Path(p).name)
            txt_path = str(Path(labels_dir) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    with open(txt_path + '.txt', 'a') as f:
                        if opt.confidence:
                            f.write(('%g ' * 6 + '\n') % (cls, *xywh, conf))
                        else:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)


    print('Results saved to %s' % labels_dir)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-run-id', type=str, required=True, help='*.checkpoint run id')
    parser.add_argument('--output-run-id', type=str, default='', help='output id in RUNS_DIR for the current job')
    parser.add_argument('--corruption', type=str, default='contrast', help='Corruption type')
    parser.add_argument('--severity', type=str, default='3', help='Corrupt severity level for test')
    parser.add_argument('--runs-dir', type=str, default='', help='defines the RUNS_DIR')
    parser.add_argument('--year', type=int, default=2012, help='year of the dataset')
    parser.add_argument('--split', type=str, default='target', help='split of the dataset')
    parser.add_argument('--part', type=str, default='images', help='part of the dataset')
    parser.add_argument('--data', type=str, default='data/city_label.yml', help='*.data path')
    parser.add_argument('--confidence', action='store_true', help='save confidence score')
    parser.add_argument('--save', action='store_true', help='save images and predictions')
    # parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--display', action='store_true', help='display results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    if opt.runs_dir is not None and opt.runs_dir != '' and opt.runs_dir != 'none':
        RUNS_DIR = opt.runs_dir



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
        raise ValueError("No ckpt found in {} nor in {}".format(weights_path, weights_alt_path))
    print("weights are: {}".format(weights))

    opt.data = check_file(opt.data)
    data_dict = OmegaConf.load(opt.data)
    with open_dict(data_dict) as d:
        d.corruption = opt.corruption
        d.severity = opt.severity
        d.ckpt_run_id = opt.ckpt_run_id
        if opt.output_run_id is not None and opt.output_run_id != '' and opt.output_run_id != 'none':
            d.output_run_id = opt.output_run_id
        d.year = opt.year
        d.split = opt.split
        d.part = opt.part
        labels_dir = d.labels_dir
        predictions_dir = d.predictions_dir


    with torch.no_grad():
        generate_labels(data_dict['source'], weights, opt, labels_dir, predictions_dir, opt.img_size)

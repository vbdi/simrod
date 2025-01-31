import streamlit as st
import streamlit.components.v1 as components
from analysis.analyze_robustness import display_chart, analyze
from omegaconf import OmegaConf, open_dict

from models.experimental import *
from utils.datasets import *
from utils.utils import *
import os, urllib, cv2
import ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context




cfg = OmegaConf.load('./cfg/app_config.yml')

RUNS_DIR = cfg['RUNS_DIR']
MODEL_PATH = cfg['MODEL_PATH']

ALL_CORRUPTIONS = cfg.corruptions.all
SELECTED_CORRUPTIONS = cfg.corruptions.selected


class LoadImagePairs(Dataset):

    def __init__(self, one, two, transform=None):
        self.one = one
        self.two = two

    def __len__(self):
        return len(self.one)

    def __getitem__(self, idx):
        # get images and labels here
        # returned images must be tensor
        # labels should be int
        _, img_1, im0s_1, _ = self.one[idx]
        _, img_2, im0s_2, _ = self.two[idx]
        return img_1, im0s_1, im0s_2, im0s_2

def get_model_path(ckpt_run_id):
    weights_path = os.path.join(RUNS_DIR, ckpt_run_id, "outputs/best.pt")
    weights_alt_path = os.path.join(RUNS_DIR, ckpt_run_id, "outputs/weights/best.pt")

    if os.path.exists(weights_path):
        weights = weights_path
    elif os.path.exists(weights_alt_path):
        weights = weights_alt_path
    else:
        raise ValueError("No ckpt found in {} nor in {}".format(weights_path, weights_alt_path))
    return weights

# @st.cache
def get_model_and_device(weights, imgsz):
    device = torch_utils.select_device("1", batch_size=1)
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()

    # Run once
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    return model, device


def get_clean_voc_dataset():
    # Load the dataset
    dataset = LoadImages(cfg.data.pascal.SOURCE_DIR, img_size=416)
    return dataset

def get_corrupted_voc_dataset(corruption, severity):
    # Load the dataset
    assert severity > 0, "severity must be between 1 and 5"
    dataset = LoadImages('{}-{}-{}'.format(cfg.data.pascal.SOURCE_DIR, corruption, severity), img_size=416)
    return dataset

# @st.cache
def get_dataset(dataset_path):
    return LoadImages(dataset_path, img_size=416)

# This sidebar UI lets the user select corruption type and severity
ALL_CORRUPTIONS_DICT = {s: ' '.join(s.title().split('_')) for s in ALL_CORRUPTIONS}
SELECTED_CORRUPTIONS_DICT = {s: ' '.join(s.title().split('_')) for s in SELECTED_CORRUPTIONS}
ALL_CONDITIONS_DICT = {'all': 'All weather scene (target)',
                       'clear': 'Clear street scene (source)',
                       'day': 'Rainy day scene',
                       'dusk': 'Rainy dusk scene',
                       'night': 'Rainy night scene'}

STREET_CONDITIONS_DICT = {'clear': 'Clear street scene (source)',
                       'foggy': 'Foggy street scene (target)'}

def corruption_ui():
    st.sidebar.markdown("# Corruption")
    corruption = st.sidebar.selectbox("Select the image corruption type", SELECTED_CORRUPTIONS, 0, format_func=SELECTED_CORRUPTIONS_DICT.get)
    severity = st.sidebar.slider("Severity level", 1, 5, 3, step=1)
    return corruption, severity

def data_city_ui():
    st.sidebar.markdown("# Conditions")
    ALL_CONDITIONS = cfg.models.city.conditions
    condition = st.sidebar.selectbox("Select the visibility conditions", ALL_CONDITIONS, 0, format_func=ALL_CONDITIONS_DICT.get)
    return condition

def data_street_ui():
    st.sidebar.markdown("# Conditions")
    ALL_CONDITIONS = cfg.models.street.conditions
    condition = st.sidebar.selectbox("Select the visibility conditions", ALL_CONDITIONS, 0, format_func=STREET_CONDITIONS_DICT.get)
    return condition

def model_select_ui(selections):
    st.sidebar.markdown("# Model")
    model_name = st.sidebar.selectbox("Select an object detection model", selections, 0)
    return model_name

def main():
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Robust Domain Adaptation")
    app_mode = st.sidebar.selectbox("First, select what to demo",
                                    ["Visualize the domain shift on synthetic image corruptions",
                                     "Compare models on synthetic image corruptions",
                                     "Apply to all-weather driving",
                                     "Apply to all-weather street view"])
    # st._main.empty()
    chart_holder = st.empty()
    baseline_viz_holder = st.empty()
    proposed_viz_holder = st.empty()
    demo_header_holder = st.empty()
    image_caption_holder = st.empty()

    if app_mode == "Visualize the domain shift on synthetic image corruptions":
        st.sidebar.success('Comparing baseline model performance on clean and corrupted images.')
        selections = cfg['models']['pascal']['selections']
        selected_model = model_select_ui(selections)
        if selected_model == "Pascal S":
            model_size = 'small'
        elif selected_model == 'Pascal M':
            model_size = 'medium'
        elif selected_model == 'Pascal L':
            model_size = 'large'
        else:
            model_not_found_warning_holder = st.sidebar.warning("Model {} not found. Select another model".format())
            time.sleep(2.5)
            model_not_found_warning_holder.empty()

        corruption, severity = corruption_ui()

        model_cfg = cfg.models.pascal.baseline[model_size]
        ckpt_run_id = model_cfg.id
        src_ap, tgt_ap = model_cfg.ap.src, model_cfg.ap[corruption]

        description = """This demonstrates the effect of domain shift due to image corruptions on the performance of the baseline (no adapation) model.
                      <br>You can experiment with different corruption types and levels of severity by using the controls in the side bar.
                      <br>The {} model achieves an <strong>AP50 of {}% on clean images</strong> and an <strong>AP50 of {}% on corrupted images</strong>.
                      <br>The AP performance of the model is averaged over 5 levels of corruption severity.""".format(selected_model, src_ap, tgt_ap)
        header = "Visualizing the effect of domain shift due to common image corruptions"
        demo_header_holder = demo_header_ui(header, description)
        image_caption_holder = components.html("""
            <h3>
                <b style="color:green">Source Domain</b> (clean images)    &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <b style="color:red">Target Domain</b> (corrupted images) 
            </h3>
            """, height=40)


        baseline_viz_holder = detect_images(ckpt_run_id, corruption, severity)
        chart_holder.empty()
        # proviz_holder.empty()
    elif app_mode == "Compare models on synthetic image corruptions":
        st.sidebar.success('Testing the self-adaptation framework on synthetically corrupted images.')
        selections = cfg.models.pascal['limited_selections']
        model_name = model_select_ui(selections)
        if model_name == "Pascal S":
            model_size = 'small'
        elif model_name == 'Pascal M':
            model_size = 'medium'
        elif model_name == 'Pascal L':
            model_size = 'large'
        else:
            model_not_found_warning_holder = st.sidebar.warning("Model {} not found. Select another model".format())
            time.sleep(2.5)
            model_not_found_warning_holder.empty()

        corruption, severity = corruption_ui()

        baseline_cfg = cfg.models.pascal.baseline[model_size]
        proposed_cfg = cfg.models.pascal[corruption][model_size]
        baseline_ckpt_run_id, proposed_ckpt_run_id = baseline_cfg.id, proposed_cfg.id
        baseline_ap, proposed_ap = baseline_cfg.ap[corruption], proposed_cfg.ap

        description = """Here we can compare the performance of an adapted model against that of a baseline model trained on source data.
                         <br>You can experiment with different models sizes, corruption types and levels of severity by using the controls in the side bar.
                         <br>The AP performance of a given model is averaged over 5 levels of corruption severity."""
        header = "Applying the unsupervised domain adaptation algorithm to the baseline model"
        demo_header_holder = demo_header_ui(header, description)

        image_caption_holder = components.html("""
                    <h3>
                        <b style="color:blue">Baseline {} </b> (AP50 = {}% on target)  &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  <b style="color:green">Adapted {} </b> (AP50 = {}%  on target) 
                    </h3>
                    """.format(model_name, baseline_ap, model_name, proposed_ap), height=40)


        chart_holder.empty()
        baseline_viz_holder.empty()
        proposed_viz_holder = compare_models(baseline_ckpt_run_id, proposed_ckpt_run_id, corruption, severity)
    elif app_mode == 'Apply to all-weather driving':
        st.sidebar.success('Testing the self-adaptation framework to driving scenes under different weather and natural corruptions.')
        selections = cfg.models.city.selections
        model_name = model_select_ui(selections)
        if model_name == "City S":
            model_size = 'small'
        elif model_name == 'City M':
            model_size = 'medium'
        elif model_name == 'City L':
            model_size = 'large'
        else:
            model_not_found_warning_holder = st.sidebar.warning("Model {} not found. Select another model".format())
            time.sleep(2.5)
            model_not_found_warning_holder.empty()

        baseline_cfg = cfg.models.city.baseline[model_size]
        proposed_cfg = cfg.models.city.adapted[model_size]
        baseline_ckpt_run_id, proposed_ckpt_run_id = baseline_cfg.id, proposed_cfg.id

        baseline_ap = baseline_cfg.ap.tgt if (baseline_cfg and baseline_cfg.ap and baseline_cfg.ap.tgt) else ''
        proposed_ap = proposed_cfg.ap.tgt if (baseline_cfg and baseline_cfg.ap and baseline_cfg.ap.tgt) else ''

        description = """We apply the adaptation framework to driving footages under poor visibility conditions.
                                 <br>The Rainy Driving footage showcases rainy and dark scenes as well as other corruptions such as pixelation and motion blur. The baseline model is trained on cityscapes dataset as source data. The target domain includes rainy scenes shot from day to night. The indicated validation AP50 metric is measured on the target domain."""
        header = "Applying the domain adaptation algorithm to the all-weather driving scenes."
        demo_header_holder = demo_header_ui(header, description)

        image_caption_holder = components.html("""
                            <h3>
                                <b style="color:blue">Baseline {} </b> (AP50 = {}%  on target)  &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  <b style="color:green">Adapted {} </b> (AP50 = {}%  on target) 
                            </h3>
                            """.format(model_name, baseline_ap, model_name, proposed_ap), height=40)

        condition = data_city_ui()

        chart_holder.empty()
        baseline_viz_holder.empty()
        proposed_viz_holder.empty()
        path = cfg.data.city[condition].path
        dataset = LoadImages(path)

        proposed_viz_holder = display_video(baseline_ckpt_run_id, proposed_ckpt_run_id, dataset, city_model=('City' in model_name) or ('Street' in model_name))

    elif app_mode == 'Apply to all-weather street view':
        st.sidebar.success('Testing the self-adaptation framework to outdoor camera scenes under different weather and time of the day.')
        selections = cfg.models.street.selections
        model_name = model_select_ui(selections)
        if model_name == "Street S":
            model_size = 'small'
        elif model_name == 'Street M':
            model_size = 'medium'
        elif model_name == 'Street L':
            model_size = 'large'
        else:
            model_not_found_warning_holder = st.sidebar.warning("Model {} not found. Select another model".format())
            time.sleep(2.5)
            model_not_found_warning_holder.empty()

        baseline_cfg = cfg.models.street.baseline[model_size]
        proposed_cfg = cfg.models.street.adapted[model_size]
        baseline_ckpt_run_id, proposed_ckpt_run_id = baseline_cfg.id, proposed_cfg.id

        baseline_ap = baseline_cfg.ap.tgt if (baseline_cfg and baseline_cfg.ap and baseline_cfg.ap.tgt) else ''
        proposed_ap = proposed_cfg.ap.tgt if (baseline_cfg and baseline_cfg.ap and baseline_cfg.ap.tgt) else ''

        description = """We apply the adaptation framework to street view footages under poor visibility conditions.
                                 <br>The baseline model is trained with clear street scenes as source data. The target domain is foggy street scene. The indicated validation AP50 metric is measured on the target domain."""
        header = "Applying the domain adaptation algorithm to the all-weather static street views."
        demo_header_holder = demo_header_ui(header, description)

        image_caption_holder = components.html("""
                            <h3>
                                <b style="color:blue">Baseline {} </b> (AP50 = {}%  on target)  &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  <b style="color:green">Adapted {} </b> (AP50 = {}%  on target) 
                            </h3>
                            """.format(model_name, baseline_ap, model_name, proposed_ap), height=40)

        condition = data_street_ui()

        chart_holder.empty()
        baseline_viz_holder.empty()
        proposed_viz_holder.empty()
        path = cfg.data.street[condition].path
        dataset = LoadImages(path)
        proposed_viz_holder = display_video(baseline_ckpt_run_id, proposed_ckpt_run_id, dataset, city_model=('City' in model_name) or ('Street' in model_name))


    elif app_mode == "Show experimental results":
        chart_holder = display_chart()
        baseline_viz_holder.empty()
        proposed_viz_holder.empty()
# @st.cache
def get_datasets(corruption, severity):
    clean_dataset = get_clean_voc_dataset()
    corrupt_dataset = get_corrupted_voc_dataset(corruption, severity)
    return zip(clean_dataset, corrupt_dataset)

def detect_images(ckpt_run_id, corruption='contrast', severity=3, city_model=False):
    # Get the model
    model_path = get_model_path(ckpt_run_id)
    model, device = get_model_and_device(model_path, 416)
    half = device.type != 'cpu'

    # Get names and colors
    if city_model == False:
        names =  model.module.names if hasattr(model, 'module') else model.names
    else:
        names = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    def run_and_plot_detections(img, im0s):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img, augment=False)[0]

        # Apply NMS
        conf_thres, iou_thres = 0.4, 0.5  # 0.5
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            im0 = im0s
            cv2.cvtColor(im0, cv2.COLOR_BGR2RGB, im0)

            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Plot box onto image
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
        return im0


    image_holder = st.empty()

    for tuple1, tuple2 in get_datasets(corruption, severity):
        _, img_1, im1s, _ = tuple1
        _, img_2, im2s, _ = tuple2

        im1 = run_and_plot_detections(img_1, im1s)
        im2 = run_and_plot_detections(img_2, im2s)

        time.sleep(2)

        image_holder.image([im1, im2], width=450)
    return image_holder


def display_video(ckpt_1, ckpt_2, dataset, city_model=False):
    # Get the model
    model_path_1 = get_model_path(ckpt_1)
    model_1, device_1 = get_model_and_device(model_path_1, 416)
    half = device_1.type != 'cpu'

    # Get names and colors
    if city_model == False:
        names = model_1.module.names if hasattr(model_1, 'module') else model_1.names
    else:
        names = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    model_path_2 = get_model_path(ckpt_2)
    model_2, device_2 = get_model_and_device(model_path_2, 416)
    half = device_2.type != 'cpu'

    def run_and_plot_detections(img, im0s):
        img = torch.from_numpy(img).to(device_1)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred_1 = model_1(img, augment=False)[0]
        pred_2 = model_2(img, augment=False)[0]
        # Apply NMS
        conf_thres, iou_thres = 0.4, 0.5  # 0.5
        pred_1 = non_max_suppression(pred_1, conf_thres, iou_thres, agnostic=False)
        pred_2 = non_max_suppression(pred_2, conf_thres, iou_thres, agnostic=False)

        im0_1 = im0s.copy()
        cv2.cvtColor(im0_1, cv2.COLOR_BGR2RGB, im0_1)
        im0_2 = im0s.copy()
        cv2.cvtColor(im0_2, cv2.COLOR_BGR2RGB, im0_2)

        # Process detections
        for i, det in enumerate(pred_1):  # detections per image

            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0_1.shape).round()

                # Plot box onto image
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0_1, label=label, color=colors[int(cls)], line_thickness=3)

        for i, det in enumerate(pred_2):  # detections per image

            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0_2.shape).round()

                # Plot box onto image
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0_2, label=label, color=colors[int(cls)], line_thickness=3)
        return im0_1, im0_2


    image_holder = st.empty()
    # image_holder_two = st.empty()
    # dataset = LoadImages(dataset_path, img_size=416)
    for tuple in dataset:
        _, img, im0s, _ = tuple

        im0_1, im0_2 = run_and_plot_detections(img, im0s)

        time.sleep(1.5)

        image_holder.image([im0_1, im0_2], width=450)
        # image_holder_two.image([im0_1, im0_2], width=450)
        # image_holder.image(im0_1, width=600)
    return image_holder


def compare_models(ckpt_1, ckpt_2, corruption='contrast', severity=3):
    # Get the model
    model_path_1 = get_model_path(ckpt_1)
    model_1, device_1 = get_model_and_device(model_path_1, 416)
    half = device_1.type != 'cpu'

    # Get names and colors
    names = model_1.module.names if hasattr(model_1, 'module') else model_1.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    model_path_2 = get_model_path(ckpt_2)
    model_2, device_2 = get_model_and_device(model_path_2, 416)
    half = device_2.type != 'cpu'

    def run_and_plot_detections(img, im0s):
        img = torch.from_numpy(img).to(device_1)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred_1 = model_1(img, augment=False)[0]
        pred_2 = model_2(img, augment=False)[0]
        # Apply NMS
        conf_thres, iou_thres = 0.4, 0.5  # 0.5
        pred_1 = non_max_suppression(pred_1, conf_thres, iou_thres, agnostic=False)
        pred_2 = non_max_suppression(pred_2, conf_thres, iou_thres, agnostic=False)

        # Process detections
        for i, det in enumerate(pred_1):  # detections per image

            im0_1 = im0s.copy()
            cv2.cvtColor(im0_1, cv2.COLOR_BGR2RGB, im0_1)

            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0_1.shape).round()

                # Plot box onto image
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0_1, label=label, color=colors[int(cls)], line_thickness=3)

        for i, det in enumerate(pred_2):  # detections per image

            im0_2 = im0s
            cv2.cvtColor(im0_2, cv2.COLOR_BGR2RGB, im0_2)

            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0_2.shape).round()

                # Plot box onto image
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0_2, label=label, color=colors[int(cls)], line_thickness=3)
        return im0_1, im0_2

    # description = """This demonstrates the effect of domain shift due to image corruptions on the performance of a baseline object detection model.
    #                         You can experiment with different corruption types and levels of severity by using the controls in the side bar."""
    # header = "Visualizing the effect of domain shift due to common image corruptions"
    # demo_header_ui(header, description)
    # components.html("""
    # <h3>
    #     <b style="color:green">Source Domain</b> (clean images)    &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <b style="color:red">Target Domain</b> (corrupted images)
    # </h3>
    # """, height=40)
    image_holder = st.empty()

    for tuple in get_corrupted_voc_dataset(corruption, severity):
        _, img, im0s, _ = tuple

        im0_1, im0_2 = run_and_plot_detections(img, im0s)

        time.sleep(2)

        image_holder.image([im0_1, im0_2], width=450)
    return image_holder


def display_results():
    display_chart()

def run_app():
    display_results()

def demo_header_ui(header, description):
    components.html(
        """
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="rindra">
        <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="rindra"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="rindra"></script>
        <div id="accordion">
          <div class="card">
            <div class="card-header" id="headingOne">
              <h5 class="mb-0">
                {} <button class="btn btn-link" data-toggle="collapse" data-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
                Show/Hide details.
                </button>
              </h5>
            </div>
            <div id="collapseOne" class="collapse show" aria-labelledby="headingOne" data-parent="#accordion">
              <p class="card-body">
                {}
              </p>
            </div>
          </div>
        </div>
        """.format(header, description),
        height=200,
    )



if __name__ == "__main__":
    st.beta_set_page_config(  # Alternate names: setup_page, page, layout
        layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
        initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
        page_title="Robust DA",
        page_icon=":shark:",
    )
    hide_streamlit_style = """
                <style>
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    main()


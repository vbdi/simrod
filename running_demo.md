# SimROD Demo application

You will need to install the pip library streamlit as explained in `docs/installation.md`. If not yet installed, run:

The pretrained models for the demo applications will be made available for download in the future. When available, copy them to [/home/USERNAME/tmp_guild/runs/](/home/USERNAME/tmp_guild/runs/) and the model definitions are available in the configuration file [src/yolo/cfg/app_config.yml](cfg/app_config.yml)


```
pip install streamlit
```

Then, you also need to change the configuration in `cfg/app_config.yml` which defines where the model checkpoints and data are stored. 

```yaml
MODEL_PATH: /home/USERNAME/
RUNS_DIR: /home/USERNAME/tmp_guild/runs

data:
  pascal:
    SOURCE_DIR: /home/USERNAME/data/voc_yolo/VOC2007/images/test_demo
    test:
      path: /home/USERNAME/data/voc_yolo/VOC2007/images/test_demo
  city:
    all:
      path: /home/USERNAME/data/rain_yolo/images/test_demo/
    clear:
      path: /home/USERNAME/data/city_yolo/images/val/
    day:
      path: /home/USERNAME/data/video/da/Raincouver_v1/test-1-rain-day.mp4
    dusk:
      path: /home/USERNAME/data/rain_yolo/images/test-4
    night:
      path: /home/USERNAME/data/rain_yolo/images/test-5

  street:
    clear:
      path: /home/USERNAME/data/street_yolo/images/test-clear/
    foggy:
      path: /home/USERNAME/data/street_yolo/images/test-fog/
```

It also defines all the models that are used for all parts of the demo. For example, the Pascal VOC models are listed in:

```
models:
  pascal:
    selections: ['Pascal S', 'Pascal M', 'Pascal L']
    limited_selections: ['Pascal M', 'Pascal L']
    baseline:
      small:
        id: 841b0d3fed3a41e1ae8cd5e84a9fc233
        ap:
          src: 75.87
          contrast: 42.02
      medium:
        id: 8fbf557d7ba34f5b9ac12b73d00dc6a1
        ap:
          src: 83.13
          contrast: 55.81
      large:
        id: f17e0b1be899485fa16778ec77df976f
        ap:
          src: 87.48
          contrast: 65.73
    contrast:
      small:
        id: e2aa73f019a6450385042b397d5af612
        ap: 74.75
      medium:
        id: fb30436c44db4dd8a2ef51e6b15e3e88
        ap: 80.27
      large:
        id: 90da0a2471914ae489a7b818fdbde332
        ap: 83.35
    frost:
      small:
        id:
        ap:
      medium:
        id: 29f307e791584f1d8e45d0c9138ff269
        ap: 75.97
      large:
        id: 7760b39317954090a65dbffe63f2199c
        ap: 81.13
    fog:
      small:
        id:
        ap:
      medium:
        id: f17df52dba7940d3ba4aa48d00f7a16e
        ap: 83.79
      large:
        id: 3d1acf8c7a394e16a164f0b4fb81da16
        ap: 86.70
    ...
```

Once the models and data are configured properly, run the demo application with:

```
streamlit run demo_app.py
```

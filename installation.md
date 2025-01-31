
## Conda environment setup

We use the following library for the specified purpose:

- imagecorruptions: for preparing the image corruption datasets
- guildai: for managing and tracking our experiments
- omegaconf: for configuring the experiments
- streamlit: for building our demo applications

Proper citations will be later added to our main paper as these tools are crucial for the ease of reproducibility of this work.

```
conda create -n yolo python=3.7
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
pip install imagecorruptions
pip install omegaconf
pip install guildai==0.7.0
pip install tidecv
pip install pycocotools==2.0.1
pip install matplotlib
conda install tqdm
pip install scikit-image==0.16.2
pip install streamlit
```
# Data preparation

This section gives details about how to structure the data folders, configure the train/val/test splits, format the ground-truth labels and generate pseudo-labels. 

## Data folder structure

At a high-level, the data folders are organized as follow:


```
voc_yolo
|- VOC_2007
    |- images
        |- trainval
        |- test
        |- test-contrast-1
        |- ...
        |- test-contrast-5
        |- ...
        |- test-fog-1
        |- ...
        |- test-fog-5
|- VOC_2012
    |- images
        |- trainval
        |- test
        |- trainval-contrast-1
        |- ...
        |- trainval-contrast-3
        |- ...
        |- test-contrast-1
        |- test-contrast-2
        |- test-contrast-3
        |- test-contrast-4
        |- test-contrast-5
        |- test-contrast-0
        |- ...
```

There are different dataset splits including the original `trainval` and `test` of `VOC2007`, the original `trainval` split of `VOC2023`. 
For the domain adaptation experiments, we also have the corrupted dataset splits, i.e. from the target domain. The corrupted training data splits' names in `VOC2012` follows the `trainval-<CORRUPTION_TYPE>-<SEVERITY_LEVEL>` convention. Similarly, the corrupted test data splits's names in `VOC2007` follows the `test-<CORRUPTION_TYPE>-<SEVERITY_LEVEL>`. 
For the corrupted data splits, the severity level varies from 1 to 5. For evaluation purpose, we also have test splits with severity `0`, which concatenates all corrupted images with severity levels between 1 and 5. 

Each split (e.g. trainval, test or test-contrast-3) must have both an `images/<SPLIT_NAME>` and `labels/<SPLIT_NAME>`, which contains respectively the JPEG/PNG files and the correspodning label files as TXT files. 
Importantly, the name of the JPEG and TXT files must be the same. 

As an example, below is a structure of the folders and files at a deeper level:

```
voc_yolo
|- VOC_2007
    |- images
        |- trainval
            |- 000001.jpg
            |- 000003.jpg
            |- 000005.jpg
            |- ...
        |- test
            |- 000002.jpg
            |- 000004.jpg
            |- ...
        |- test-contrast-1
            |- 000002.jpg
            |- 000004.jpg
            |- ...
        |- ...
        |- test-contrast-5
            |- 000002.jpg
            |- 000004.jpg
            |- ...
        |- test-contrast-0
            |- 000002-contrast-1.jpg
            |- 000002-contrast-2.jpg
            |- 000002-contrast-3.jpg
            |- 000002-contrast-4.jpg
            |- 000002-contrast-5.jpg
            |- 000004-contrast-1.jpg
            |- 000004-contrast-2.jpg
            |- 000004-contrast-3.jpg
            |- 000004-contrast-4.jpg
            |- 000004-contrast-5.jpg
            |
    |- labels
        |- trainval
            |- 000001.txt
            |- 000003.txt
            |- 000005.txt
            |- ...
        |- test
            |- 000002.txt
            |- 000004.txt
            |- ...
        |- test-contrast-1
            |- 000002.jpg
            |- 000004.jpg
            |- ...
        |- ...
        |- test-contrast-5
            |- 000002.txt
            |- 000004.txt
            |- ...
        |- test-contrast-0
            |- 000002-contrast-1.txt
            |- 000002-contrast-2.txt
            |- 000002-contrast-3.txt
            |- 000002-contrast-4.txt
            |- 000002-contrast-5.txt
            |- 000004-contrast-1.txt
            |- 000004-contrast-2.txt
            |- 000004-contrast-3.txt
            |- 000004-contrast-4.txt
            |- 000004-contrast-5.txt
            |...
        |- ...

|- VOC_2012
    |- images
        |- trainval
            |- 000001.jpg
            |- 000003.jpg
            |- 000005.jpg
            |- ...
        |- trainval-contrast-1
            |- 000001.jpg
            |- 000003.jpg
            |- 000005.jpg
        |- ...
        |- trainval-contrast-5
            |- 000001.jpg
            |- 000003.jpg
            |- 000005.jpg
        |- ...
        |- trainval-fog-1
            |- 000001.jpg
            |- 000003.jpg
            |- 000005.jpg
        |- ...
        |- trainval-fog-5
            |- 000001.jpg
            |- 000003.jpg
            |- 000005.jpg
        ...
    |- labels
        |- trainval
            |- 000001.txt
            |- 000003.txt
            |- 000005.txt
            |- ...
        |- trainval-contrast-1
            |- 000001.txt
            |- 000003.txt
            |- 000005.txt
        |- ...
        |- trainval-contrast-5
            |- 000001.txt
            |- 000003.txt
            |- 000005.txt
        |- ...
        |- trainval-fog-1
            |- 000001.txt
            |- 000003.txt
            |- 000005.txt
        |- ...
        |- trainval-fog-5
            |- 000001.txt
            |- 000003.txt
            |- 000005.txt
        ...

```

## Data configuration

The data configuration is specified by a YAML configuration file. Precisely, the data loader obtains the folder locations of the train/val/test splits specified by the YAML configuration, loads the labels from the labels folders and infers the image filenames from the labels' filename. 

An example of a data configuration in `data/pascal_c/pasca-source.yml`

```yaml
USERNAME: <define-your-user-name-here>

data_folder: /home/$USERNAME/data

corruption: contrast
severity: 3
ckpt_run_id: 0

test:
  severity: 0

# train data split
train: ${data_folder}/voc_yolo/VOC2007/labels/trainval

# validation data split
val: ${data_folder}/voc_yolo/VOC2007/labels/test

# clean test data (source domain)
clean_test: ${data_folder}/voc_yolo/VOC2007/labels/test

# corrupted test data (target domain)
corrupt_test: ${data_folder}/voc_yolo/VOC2007/labels/test-${corruption}-${test.severity}/
```

Another data configuration `data/pascal_c/pascal_adapt.yaml` that is by our domain adaptation method is shown below:

> Note that as long as you are using the GuildAI operation to run the experiment steps, you do not need to manually set the data configurations since they are already pre-configured in `guild.yml`.

## Label formatting

Each label file in `labels/<SPLIT_NAME>/` corresponds to one and only image. For example, the label file `000001.txt` below contains 2 lines which corresponds to 2 objects in the image.
Each line corresponds to the class index and the 4 normalized coordinates of the ground-truth bounding box in the YOLO format `(x_c, y_c, w, h)`.

```
> cd ~/data/voc_yolo/
> cat VOC2007/labels/test/0000001.txt

File:    VOC2007/labels/test/000001.txt
───────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1   │ 11 0.34135977337110485 0.609 0.4164305949008499 0.262
   2   │ 14 0.5070821529745043 0.508 0.9745042492917847 0.972
```



### Generate image corruption data

To generate the corrupted images, we use the pip library imagecorruptions. The data splits used for corrupted data are explained in the supplementary material document.

Once created, organize the folders in the same way as explained above. Note that the corrupted VOC2007 test data have the exact same ground-truth labels as the VOC2007 test data. Therefore, simply copy the labels from the clean `/labels/test` folder to `/labels/test-<CORRUPTION>-<SEVERITY>`. 

```
voc_yolo
|- VOC_2007
    |- images
        |- test
            |- 000002.jpg
            |- 000004.jpg
            |- ...
        |- test-contrast-1
            |- 000002.jpg
            |- 000004.jpg
            |- ...
        |- ...
        |- test-contrast-5
            |- 000002.jpg
            |- 000004.jpg
            |- ...
        |- test-contrast-0
            |- 000002-contrast-1.jpg
            |- 000002-contrast-2.jpg
            |- 000002-contrast-3.jpg
            |- 000002-contrast-4.jpg
            |- 000002-contrast-5.jpg
            |- 000004-contrast-1.jpg
            |- 000004-contrast-2.jpg
            |- 000004-contrast-3.jpg
            |- 000004-contrast-4.jpg
            |- 000004-contrast-5.jpg
            | ...
    |- labels
        |- test
            |- 000002.txt
            |- 000004.txt
            |- ...
        |- test-contrast-1
            |- 000002.jpg
            |- 000004.jpg
            |- ...
        |- ...
        |- test-contrast-5
            |- 000002.txt
            |- 000004.txt
            |- ...
        |- test-contrast-0
            |- 000002-contrast-1.txt
            |- 000002-contrast-2.txt
            |- 000002-contrast-3.txt
            |- 000002-contrast-4.txt
            |- 000002-contrast-5.txt
            |- 000004-contrast-1.txt
            |- 000004-contrast-2.txt
            |- 000004-contrast-3.txt
            |- 000004-contrast-4.txt
            |- 000004-contrast-5.txt
            |...
        |- ...
```
# Segmentation of bones on X-ray images and detection of cracks and fractures

## Formulation of the problem

It is necessary to develop a model that determines the presence of bones on an X-ray image and detects bone damage sites
____________

## Repo Structure

- experiments/ - notebooks
- fract_detect/ - Application code. Also available at [dockerhub](https://hub.docker.com/repository/docker/fdkonovalenko/ssd300_bones/general)
- pictures/ - images for readme.md

## Models

### Instance Segmentation  MASK-RCNN


The instance segmentation model could be used to determine the bounding boxes of individual bones and then classify the bones into intact and damaged individually. However, now it is not possible to prepare a dataset of sufficient size to train a classification model, and therefore a different approach with semantic segmentation was used at the first stage.
___________________


### Semantic segmentation of bones with subsequent decision-making on the need to detect the location of damage
------
- a reduced (224x224) image is supplied to the input
- a simple semantic segmentation model (SegNet or UNet architecture) processes the image
- if a mask of sufficient size is detected with sufficient confidence in the image, the user is notified to upload a full-size image
- after loading the full-size image, the bone damage detection model is launched

Masks were manually marked (CVAT) for individual bones in images from the Mura and FracAtlas datasets. A total of about 160 pictures.

<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/0.png" width="700" height="200">

*segmantation mask examples*

For semantic segmentation the SegNet pretained at [ADDI](https://www.fc.up.pt/addi/ph2%20database.html) dataset was used. The pretrained SegNet showed slightly better results than the non-pretrained one, as well as higher operating speed (with comparable quality) than UNet.

<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/5.png" width="700" height="200">

*ADDI data example*


<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/1.png" width="700" height="200">

*SegNet Training*


<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/3.png" width="500" height="500">

*SegNet Semantic Segmentation*

For subsequent detection of damage sites, the SSD300-VGG16 model was used, additionally trained on the FracAtlas dataset. Since the training sample was small, upsampling was used by reflecting images relative to the horizontal and vertical axes (the sample was increased by 4 times).

**training Results**
------
**Raw Dataset**

**Images - 717**

**Test images - 71**

|Size|Epochs|LR|Mean IoU|Maximun IoU|Comments|
|-|-|-|-|-|-|
|224x224|-|-|-|-|Mimimal size 300х300|
|300x300|15|3e-4|0,21|0,91||
|300x300|30|3e-4|0,22|0,88|Overfitting|
|300x300|10|3e-4|0,22|0,89||
|640x640|15|3e-4|0,27|0,79||
|640x640|30|3e-4|0,25|0,80|Overfitting|
|640x640|20|3e-4|0,21|0,82|Overfitting|
|**640x640**|**15**|**1e-4**|**0,34**|**0,86**||
|800x800|15|1e-4|0,29|0,86||
|1024x1024|20|1e-4|0,27|0,94|There are false positives at boundaries of bones|

**After Upsampling**

|Size|Epochs|LR|Mean IoU|Maximum IoU|Mean Map50|Maximum MaP50|Comments|
|-|-|-|-|-|-|-|-|
|640x640|15|1e-4|0,31|0,94|0,47|1,00||
|800x800|15|1e-4|0,32|0,89|0,47|1,00||




<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/4.png" width="900" height="600">

*The SSD300 prediction example. Blue -Preicted, Red - Ground Truth*

**YOLOv8 was also considered for detection**

The results for metrics and the presence of false positive alarms are comparable to the SSD300-VGG16. However, as the dataset size increases, it makes sense to compare the quality of these models again.

<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/9.png" width="900" height="600">

*YOLOv8 training results*

<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/10.png" width="900" height="600">

*Yolo detection example*

_____________
**As the result the SSD300-VGG-16 model, trained at the umsampled dataset with 640x640 px images was selected**
__________

### The X-Ray images processing application
-------
Based on FastAPI



<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/7.png" width="450" height="300">

*Segmentation (The image is not X-Ray image)*


<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/6.png" width="300" height="300">

*Segmentation (The image is X-Ray image)*


<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/8.png" width="300" height="300">

*Detection*

### Model Inference
------
```bash
docker pull fdkonovalenko/ssd300_bones:latest
docker run --rm -it -p 8010:8000 --name app fdkonovalenko/ssd300_bones

```
Then the model is available at localhost://8010
______

## The main problems

- Small Dataset
- To manually mark sites of bone damage, professional competence is required: it is often difficult to distinguish a crack by eye from a bone boundary or an image artifact

## Datasets


|Data|Link|Comments|
|--|--|--|
|MURA|[MURA](https://stanfordmlgroup.github.io/competitions/mura/)|только руки, разметка "есть дефект/нет дефекта"|
|roboflow|[Roboflow](https://universe.roboflow.com/nishanth/x-ray-bones-detector/dataset/4)|разметка по отдельным костям. только кисти рук|
|FracAtlas|[FracAtlas](https://github.com/XLR8-07/FracAtlas)|разметка с сегментацией и bbox только для повреждений костей. Кости всего тела. Есть бэйзлайн по обучению YOLO8 на поиск дефектов|

## References

- https://www.nature.com/articles/s41597-023-02432-4
- https://paperswithcode.com/datasets?q=Fracture%2FNormal+Shoulder+Bone+X-ray+Images+on+MURA&task=semantic-segmentation&page=1
- https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/README.md
- https://www.kaggle.com/code/alexj21/pytorch-eda-unet-from-scratch-finetuning
- https://github.com/qubvel/segmentation_models.pytorch
- https://pytorch.org/vision/main/auto_examples/others/plot_visualization_utils.html#sphx-glr-auto-examples-others-plot-visualization-utils-py
- https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- https://paperswithcode.com/paper/diagnose-like-a-radiologist-attention-guided
- https://debuggercafe.com/custom-backbone-for-pytorch-ssd/

# Result

- a model for bone sementation and detection of fracture sites and cracks has been developed
- the application is optimized: the resource-intensive detection model is initialized only if the “light” segmentation model detects bones in the image
- further ways of improvement:
   - additional training of models on new data
   - asynchronous image processing

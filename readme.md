# Сегментация костей на рентгеновских снимках и детекция трещин и переломов

## Постановка задачи

Необходимо разработать модель, которая на рентгеновском снимке определяет наличие костей и детектирует места повреждений костей
____________

## Структура репозитория

- experiments/ - ноутбуки с экспериментами
- test_segm/ - скрипты для запуска модели instance сегментации костей
- fract_detect/ - приложение по сегментации и детекции. Также образ доступен на [dockerhub](https://hub.docker.com/repository/docker/fdkonovalenko/ssd300_bones/general)
- pictures/ - изображения для readme.md

## Рассмотренные модели и подходы

### Instance сегментация  MASK-RCNN
-------
|Этап |Статус |Результат|Примечание|
|--|--|--|--|
|Сегментация костей (1 класс)| ✅ | скрипты для инференса в /test_segm, [веса модели](https://drive.google.com/file/d/1WvzMP-0dzdsu8cHjYdkwRVPfcYzbHP0p/view?usp=drive_link). Качество так себе: размеченных данных недостаточно| Вручную размечены снимки из датасета [MURA](https://stanfordmlgroup.github.io/competitions/mura/)|
|Сегментация костей и посторонних включений: имплантов, пластин (2 класса)|  ✅  | модель обучена, IoU до 0,8, однако из-за большого дисбаланса классов импланты предсказываются плохо | |
|Сегментация костей (1 класс)| ✅ | Значимого роста IoU нет:  в среднем на уровне 40...50 %, в лучшем случае до 90%. После 8..10 эпох начинается переобучение. На изображениях уменьшенного размера качество сегментации хуже: false positive на артефактах | то же, что и пункт 1, только с дополнительно размеченными 100 изображениями из MURA и из [FracAtlas](https://github.com/XLR8-07/FracAtlas). |

Модель instance сегментации могла бы быть использована для определения bounding boxes отдельных костей и последующей классификации костей на целые и поврежденные по отдельности. Однако сейчас нет возможности подготовить датасет достаточного размера для обучения модели классификации, а потому был применен другой подход с семантической сегментацией на первом этапе.
___________________


### Семантическая сегментация костей с последующим принятием решения о необходимости детекции  места повреждения
------
- на вход подается уменьшенное (224х224) изображение
- простая модель семантической сегментации (SegNet или UNet архитектура) обрабатывает изображение
- в случае обнаружения маски достаточного размера с достаточной уверенностью на изображении пользователь уведомляется о необходимости загрузить полноразмерное изображение
- после загрузки полноразмерного изображения запускается модель детекции повреждений костей

Вручную (CVAT) были размечены масками по отдельным костям на снимках из датасетов Mura и FracAtlas. Суммарно около 160 снимков. 

<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/0.png" width="700" height="200">

*Пример сегментационных масок*

Для семантической сегментации была использована модель SegNet, предобученная на датасете [ADDI](https://www.fc.up.pt/addi/ph2%20database.html) - это дерматологический датасет с различными дефектами кожи. Предобученная SegNet показала несколько лучшие результаты, чем не предобученная, а также большую скорость работы (при сравнимом качестве), чем UNet.


<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/5.png" width="700" height="200">

*Пример изображений из датасета ADDI*


<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/1.png" width="700" height="200">

*Обучение SegNet*


<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/3.png" width="500" height="500">

*Семантическая сегментация от SegNet*

Для последующей детекции мест повреждений использовалась модель SSD300-VGG16, дообученная на датасете FracAtlas. Поскольку обучающая выборка была мала, использовался апсемплинг путем отражения изображений относительно горизонтальной и вертикальной оси (выборка была увеличена в 4 раза).

**Результаты обучения**
------
**Без апсемплинга**

**Всего изображений - 717**

**В тестовой выборке - 71**

|Размер|Количество эпох|LR|Среднее значение IoU на тестовых данных|Максимальное значение IoU на тестовых данных|Примечание|
|-|-|-|-|-|-|
|224x224|-|-|-|-|Минимум 300х300|
|300x300|15|3e-4|0,21|0,91||
|300x300|30|3e-4|0,22|0,88|Переобучается|
|300x300|10|3e-4|0,22|0,89||
|640x640|15|3e-4|0,27|0,79||
|640x640|30|3e-4|0,25|0,80|Переобучается|
|640x640|20|3e-4|0,21|0,82|Переобучается|
|640x640|15|1e-4|0,34|0,86||
|800x800|15|1e-4|0,29|0,86||
|1024x1024|20|1e-4|0,27|0,94|Дает false positive на границы между костями и артефакты снимка|

**После апсемплинга**

**Всего изображений - 2868**

**В тестовой выборке - 284**

|Размер|Количество эпох|LR|Среднее значение IoU на тестовых данных|Максимальное значение IoU на тестовых данных|Среднее значение mAP50 на тестовых данных|Максимальное значение mAP50 на тестовых данных|Примечание|
|-|-|-|-|-|-|-|-|
|**640x640**|**15**|**1e-4**|**0,31**|**0,94**|**0,47**|**1,00**||
|800x800|15|1e-4|0,32|0,89|0,47|1,00||





<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/4.png" width="900" height="600">

*Пример детекции места повреждения SSD300. Синее - предсказание, красное - фактическое место повреждения (из разметки).*

**Также для детекции рассматривалась YOLOv8**

Результаты по метрикам и наличию false positive срабатываний сравнимы с SSD300-VGG16. Однако при увеличении размера датасета имеет смысл еще раз сравнить качество этих моделей.

<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/9.png" width="900" height="600">

*Результаты обучения YOLOv8*

<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/10.png" width="900" height="600">

*Пример детекции мест повреждения YOLOv8.*

_____________
**В итоге была взята модель SSD300-VGG-16, обученная на выборке увеличенного размера с размером изображений 640x640**
__________

### Приложение для обработки рентгеновских снимков 
-------
На основе FastAPI



<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/7.png" width="450" height="300">

*Сегментация (первый этап) - изображение не является ренгтеновским снимком*


<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/6.png" width="300" height="300">

*Сегментация (первый этап) - изображение является ренгтеновским снимком*


<img src="https://github.com/salfa-ru/doct24_neural-network/blob/Fedor_Konovalenko/bones_segmentation/pictures/8.png" width="300" height="300">

*Детекция (второй этап)*

### Загрузка и запуск модели
------
```bash
docker pull fdkonovalenko/ssd300_bones:latest
docker run --rm -it -p 8010:8000 --name app fdkonovalenko/ssd300_bones

```
После этого модель доступна по адресу localhost://8010
______

## Основные сложности

- мало данных
- для разметки вручную мест повреждения костей необходимы профессиональные компетенции: зачастую трещину на глаз сложно отличить от границы костей или артефакта снимка

## Датасеты


|Источник|Ссылка|Примечание|
|--|--|--|
|MURA|[MURA](https://stanfordmlgroup.github.io/competitions/mura/)|только руки, разметка "есть дефект/нет дефекта"|
|roboflow|[Roboflow](https://universe.roboflow.com/nishanth/x-ray-bones-detector/dataset/4)|разметка по отдельным костям. только кисти рук|
|FracAtlas|[FracAtlas](https://github.com/XLR8-07/FracAtlas)|разметка с сегментацией и bbox только для повреждений костей. Кости всего тела. Есть бэйзлайн по обучению YOLO8 на поиск дефектов|

## Использованные источники

- https://www.nature.com/articles/s41597-023-02432-4
- https://paperswithcode.com/datasets?q=Fracture%2FNormal+Shoulder+Bone+X-ray+Images+on+MURA&task=semantic-segmentation&page=1
- https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/README.md
- https://www.kaggle.com/code/alexj21/pytorch-eda-unet-from-scratch-finetuning
- https://github.com/qubvel/segmentation_models.pytorch
- https://pytorch.org/vision/main/auto_examples/others/plot_visualization_utils.html#sphx-glr-auto-examples-others-plot-visualization-utils-py
- https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- https://paperswithcode.com/paper/diagnose-like-a-radiologist-attention-guided
- https://debuggercafe.com/custom-backbone-for-pytorch-ssd/

# Результат

- разработана модель сементации костей и детекции мест переломов и трещин
- приложение оптимизировано: ресурсоемкая модель детекции инициализируется только в случае, если "легкая" модель сегментации обнаруживает кости на снимке
- дальнейшие пути совершенствования:
  - дообучение моделей на новых данных
  - асинхронная обработка изображений

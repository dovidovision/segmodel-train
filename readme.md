# Segmentation model
We use segmentation model to feed exact cat boundary to CLIP model

### Our segmentation model
- Head : Segformer
- Backbone : swinT
- library : mmsegmentation


## Requirements
```
pip install albumentations
pip install mmcv-full
```

```
git clone https://github.com/open-mmlab/mmsegmentation.git
mv ./mmseg ./mmsegmentation/
mv ./models ./mmsegmentation/
pip install -e mmsegmentation/
```

## Datasets
We fine-tune our model by using VOC2012 and oxford pet dataset
- In oxford dataset, We use only cat dataset. and merge outline segmentation map with inner segmentation map
- In VOC2012 dataset, We use all dataset. and We mark all object to 0 except cat.
    - We provide voc relabeling tool in this repo(voc_relabeling.py)

- Instead downloading and relabeling above dataset, You directly download our dataset at this link
    - https://drive.google.com/file/d/1KN1skxSur4PPcImxh110OsigIEEWfPhY/view?usp=sharing


## Structure
```
├──/segmodel-train
|   ├── /mmsegmention
|       ├── /models
|       ├── /data
|       ├── /mmseg
|       ├── /pretrain
|       ├── others...
|
```


## Train
__requirement to execute our model__
Download SwinT weights and convert to mmseg swinT weights
And save that weights in this/repo/mmsegmentation/pretrain
- Detail information is in mmsegmentation.

__Train__
```
# In this repo mmsegmentation directory
python tools/train.py models/swinT_segformer.py
```




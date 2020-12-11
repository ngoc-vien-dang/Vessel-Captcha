# Vessel-Captcha: An Efficient Learning Framework for Vessel Annotation and Segmentation

<img src="imgs/pipeline.png" >

This is the implementation of our Vessel-CAPTCHA framework for vessel brain MRI segmentation in Keras.

### 1. Preprocessing Data
#### Skull stripping:
Removing the skull from the MRI images by applying the mask.
```sh
python -m captcha.preprocessing.skull_stripping --original_data_dir <path-to-raw-dataset> --target_dir <path-to-skull-stripping-dir>
```
### 2. Generating Weak Annotations
#### Grid creating:
Creating the grid for weak annotation.
```sh
python -m captcha.preprocessing.grid --original_data <path-to-raw-dataset> --grid_filepath <path-to-grid-dir>
```
#### Rough-mask creating:
```sh
python -m captcha.preprocessing.rough_mask --patch_size 32 --clustering 'kmeans' --patch_annotation_dir <path-to-patch-annotation-dir> --rough_mask_dirh <path-to-rough-mask-dir>
```
### 3. Building a Training Dataset
Create a directory for image patches utilized for training.
#### 2D-PnetCl: 
```sh
python -m captcha.preprocessing.cls_patch_extraction --patch_size 32  --skull_stripping_dir <path-to-skull-stripping-volume-contain-patch-annotation-file> --patch_vessel_dir <path-to-extracted-patches>
```
#### 2D-WnetSeg: 
```sh
python -m captcha.preprocessing.seg_patch_extraction --skull_stripping_dir <path-to-skull-stripping-volume-contain-pixelwise-weak-annotation> --patch_extraction_dir <path-to-extracted-patches>
```
### 4. Training Models
##### 2D-PnetCl: 
We perform training a classification model using the following command:
```sh
python -m captcha.train_pnetcls --model_arch 'pnetcls' --patch_dir <path-to-extracted-patches>  --train_metadata_filepath <path-to-save-metadata> --model_filepath <path-to-save model>
```
##### Data Augmentation for 2D-WnetSeg (optional):
```sh
python -m captcha.pnetcls.prediction --patch_size 32 --train_non_label_dir <path-to-non-label-data> --train_metadata_filepath <path-to-load-metadata> --model_filepath <path-to-load-pnetcl-model> --grid_label_filepath <path-to-save-patch-label>
```
#### 2D-WnetSeg:
We perform training a segmentation model using the following command:
```sh
python -m captcha.train_wnetseg --skull_stripping_dir <path-to-extracted-patches> --model_arch  'wnetseg' --train_metadata_filepath <path-to-save-metadata> --model_filepath <path-to-save-model> --patch_size 96
```
### 5. Vessel Segmentation
We are now able to classify pixels with and  without vessel from the brain MRI volume using the trained segmentation model from the previous step. The command that helps us to do this is:
```sh
python -m captcha.predict_full_testset --test_set_dir <path-to-testset> --model_arch 'wnetseg' --patch_size 96 --train_metadata_filepath <path-to-load-metadata> --model_filepath <path-to-load-wnetseg-model> --prediction_filepath <patch-to-save-vessel-segmentation>
```
##### Applying filtering (optional):
```sh
python -m captcha.filter --patch_size 32 --wnetseg_prediction_dir <path-to-testset> --train_metadata_filepath <path-to-load-metadata> --model_filepath <path-to-load-pnetcl-model> --wnetseg_pnetcl_filepath <path-to-save-vessel-segmentation>
```
### 6. Evaluating Framework's Performance
```sh
python -m captcha.evaluation_segmentation --result_dir <path-to-prediction-dir> 
```

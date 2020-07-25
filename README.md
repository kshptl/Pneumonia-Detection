# Pneumonia Detection from Chest X-Rays

## Project Overview

In this project, data from the NIH Chest X-ray Dataset was analyzed and trained with a Convolutional Neural Network (CNN) to classify a given chest x-ray for the presence or absence of pneumonia. This project culminates in a model that can predict the presence of pneumonia with human radiologist-level accuracy that can be prepared for submission to the FDA for 510(k) clearance as software as a medical device.

## Pneumonia and X-Rays in the Wild

Chest X-ray exams are one of the most frequent and cost-effective types of medical imaging examinations. Deriving clinical diagnoses from chest X-rays can be challenging, however, even by skilled radiologists. 

When it comes to pneumonia, chest X-rays are the best available method for diagnosis. More than 1 million adults are hospitalized with pneumonia and around 50,000 die from the disease every year in the US alone. The high prevalence of pneumonia makes it a good candidate for the development of a deep learning application for two reasons: 1) Data availability in a high enough quantity for training deep learning models for image classification 2) Opportunity for clinical aid by providing higher accuracy image reads of a difficult-to-diagnose disease and/or reduce clinical burnout by performing automated reads of very common scans. 

The diagnosis of pneumonia from chest X-rays is difficult for several reasons: 
1. The appearance of pneumonia in a chest X-ray can be very vague depending on the stage of the infection
2. Pneumonia often overlaps with other diagnoses
3. Pneumonia can mimic benign abnormalities

For these reasons, common methods of diagnostic validation performed in the clinical setting are to obtain sputum cultures to test for the presence of bacteria or viral bodies that cause pneumonia, reading the patient's clinical history and taking their demographic profile into account, and comparing a current image to prior chest X-rays for the same patient if they are available. 

## About the Dataset

The dataset was curated by the NIH specifically to address the problem of a lack of large x-ray datasets with ground truth labels to be used in the creation of disease detection algorithms.

There are 112,120 X-ray images with disease labels from 30,805 unique patients in this dataset.  The disease labels were created using Natural Language Processing (NLP) to mine the associated radiological reports. The labels include 14 common thoracic pathologies: 
- Atelectasis 
- Consolidation
- Infiltration
- Pneumothorax
- Edema
- Emphysema
- Fibrosis
- Effusion
- Pneumonia
- Pleural thickening
- Cardiomegaly
- Nodule
- Mass
- Hernia 

The biggest limitation of this dataset is that image labels were NLP-extracted so there could be some erroneous labels but the NLP labeling accuracy is estimated to be >90%.

The original radiology reports are not publicly available but you can find more details on the labeling process [here.](https://arxiv.org/abs/1705.02315) 


### Dataset Contents: 

1. 112,120 frontal-view chest X-ray PNG images in 1024*1024 resolution (under images folder)
2. Meta data for all images (Data_Entry_2017.csv): Image Index, Finding Labels, Follow-up #,
Patient ID, Patient Age, Patient Gender, View Position, Original Image Size and Original Image
Pixel Spacing.


## Project Steps

### 1. Exploratory Data Analysis

The first part of this project will involves exploratory data analysis (EDA) to understand and describe the content and nature of the data. The EDA focused on the following aspects:

* The patient demographic data such as gender, age, patient position,etc. (as it is available)
* The x-ray views taken (i.e. view position)
* The number of cases including: 
    * number of pneumonia cases,
    * number of non-pneumonia cases
* The distribution of other diseases that are comorbid with pneumonia
* Number of disease per patient 
* Pixel-level assessments of the imaging data for healthy & disease states of interest (e.g. histograms of intensity values) and compare distributions across diseases.

### 2. Building and Training the Model

**Training and validating Datasets**

From the findings in the EDA, appropriate training and validation sets for classifying pneumonia were created, taking the following into consideration: 

* Distribution of diseases other than pneumonia that are present in both datasets
* Demographic information, image view positions, and number of images per patient in each set
* Distribution of pneumonia-positive and pneumonia-negative cases in each dataset

**Model Architecture**

Transfer learning was used from an existing CNN architecture (VGG16 pre-trained in the ImageNet dataset) to classify x-rays images for the presence of pneumonia. Fine-tuning was performed by selectively freezing and training some layers of the pre-trained network.


**Image Pre-Processing and Augmentation** 

Some amount of image preprocessing was performed prior to feeding them into the neural network for training and validating. This served the purpose of conforming to the model's architecture and of augmenting the training dataset for increasing the model performance. When performing image augmentation, only those parameters that reflected real-world differences that may be seen in chest X-rays were applied. 

**Training** 

In training the model, there were parameters that were tweaked to improve performance including: 
* Image augmentation parameters
* Training batch size
* Training learning rate 
* Inclusion and parameters of specific layers in your model 

 **Performance Assessment**

The model's performance was monitored over subsequent training epochs. Choosing the appropriate metrics upon which to monitor performance was critical to achieving a high performing model. 'Accuracy' was not the most appropriate statistic in this case. An F1-score provided a better comparison to human-reader-level classifications, described in [this paper](https://arxiv.org/pdf/1711.05225.pdf).

### 3. Clinical Workflow Integration 

The imaging data in the model was transformed from DICOM format into .png to help aid in the image pre-processing and model training steps of this project. In the real world, however, the pixel-level imaging data are contained inside of standard DICOM files.

For this project, a DICOM wrapper was created that took in a standard DICOM file and output data in the format accepted by the model. The wrapper included checks for the following: 
* Proper image acquisition type (i.e. X-ray)
* Proper image acquisition orientation (i.e. those present in your training data)
* Proper body part in acquisition

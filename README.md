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


### 4. FDA  Submission

The final steps of the project are derived from the FDA's official guidance on both the algorithm description and the algorithm performance assessment.

**1. General Information:**

* Intended Use statement for your model 
* Indications For Use that should include: 
    * Target population
    * When the device could be utilized within a clinical workflow
* Device limitations, including diseases/conditions/abnormalities for which the device has been found ineffective and should not be used
* Explainatino of how a false positive or false negative might impact a patient

**2. Algorithm Design and Function**

In this section, the _fully trained_ algorithm and the DICOM header checks were explained. A flow chart was included with the following: 

* Pre-algorithm checks performed on the DICOM
* Preprocessing steps performed the algorithm on the original images (e.g. normalization) (does *NOT* include augmentation)
* The architecture of the classifier


For each stage of the algorithm, a brief description of the design and function of the algorithm was given.

**3. Algorithm Training**

A description of the following parameters of the algorithm and how they were chosen: 

* Types of augmentation used during training
* Batch size
* Optimizer learning rate
* Layers of pre-existing architecture that were frozen
* Layers of pre-existing architecture that were fine-tuned
* Layers added to pre-existing architecture

The behavior of the following throughout training (using visuals):

* Training loss
* Validation loss 

The algorithm's final performance after training was complete by showing a precision-recall curve on your validation set.

Finally, the threshold for classification that was chosen and the corresponding F1 score, recall, and precision were provided.

**4. Databases**

A list of specific information about the training and validation datasets that were curated separately, including: 

* Size of the dataset
* The number of positive cases and the its radio to the number of negative cases
* The patient demographic data (as it is available)
* The radiologic techniques used and views taken
* The co-occurrence frequencies of pneumonia with other diseases and findings

**5. Ground Truth**

The methodology used to establish the ground truth can impact reported performance. Description of how the NIH created the ground truth for the data that was provided for this project. Description of the benefits and limitations of this type of ground truth.  

**6. FDA Validation Plan**

Description of how a FDA Validation Plan would be conducted for the algorithm, including: 

* The patient population to request imaging data from from your clinical partner, including:
    * Age ranges
    * Sex
    * Type of imaging modality
    * Body part imaged
    * Prevalence of disease of interest
    * Any other diseases that should be included _or_ excluded as comorbidities in the population

* A short explanation of how to obtain an optimal ground truth 
* A chosen performance standard based on [this paper.](https://arxiv.org/pdf/1711.05225.pdf)


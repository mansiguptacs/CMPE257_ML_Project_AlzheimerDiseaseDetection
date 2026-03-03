# CMPE255_MachineLearning_Project_AlzheimerDiseaseDetection

Interpretable and Deep Learning Based Detection of Alzheimer’s Disease from Brain MRI Data
Jainil Rana, Mansi Gupta, Mohit Barade
Problem Description
Alzheimer is a neurodegenerative disease that may result in memory loss, cognitive impairment, and loss of independence. One of the key challenges in managing Alzheimer is that the structural changes in the brain, such as tissue shrinkage and ventricular enlargement, begin years before any noticeable clinical symptoms appear. MRI scans can capture these early changes, but identifying them reliably and at scale is very difficult.
We plan to build machine learning models for Alzheimer disease detection. We want to explore how different clinical and imaging features contribute to the model’s predictions, rather than treating the model as a black box. This project will help in providing clearer insights into the decision making process behind each prediction by focusing on interpretability.
The objective of this project is to develop a two-phase model for detecting Alzheimer’s disease:
Phase 1: By applying interpretable machine learning with simple and meaningful features, and utilizing handcrafted imaging biomarkers that can be directly extracted from the associated MRI images.
Phase 2: Application of deep learning models for the identification of slight spatial patterns in brain structure.
This work can benefit researchers and students by demonstrating how interpretable and deep learning models can be applied to medical imaging data in a responsible and transparent way.
Primary Dataset
OASIS-1: Open Access Series of Imaging Studies https://www.oasis-brains.org/
Dataset Description
Number of subjects: ~400 individuals
Age range: 18–96 years
Data types:
Structural T1-weighted MRI scans (3D NIfTI format)
Clinical and demographic data in CSV format
Key features available:
Age, sex, education, socioeconomic status
Cognitive assessment scores (MMSE)
Clinical Dementia Rating (CDR)
Precomputed brain volume measures (e.g., normalized whole brain volume)
Target variable: Dementia status
Methodology
Anticipated Preprocessing
Missing data treatment (as in, subjects without clinical scores)
Filtering for subjects with valid labels for supervised learning
Normalizing and Standardizing Numerical Features
Feature extraction based on volumetric and intensity values from the MRI images
Alignment and masking of MRI images for consistent analysis
Phase 1: Baseline for interpretable Machine Learning and designing handcrafted imaging biomarkers
Train classical machine learning models such as Random Forests and Logistic Regression using clinical and volumetric variables, such as age and brain volume.
Emphasize model interpretability with feature importance and coefficients. This will set up a robust and interpretable baseline and help us determine the underlying features with signals.
Preprocess raw MRI images with Python neuroimaging libraries: Nibabel and Nilearn.
Derive basic structural biomarkers with image processing, such as the ventricle-to-brain volume ratio.
Integrate these handcrafted features into traditional ML models and check whether the domain inspired imaging biomarkers improve model performance.
Phase 2: Deep Learning Enhancement
Training a 3D Convolutional Neural Network on MRI volumes.
Employing architectures like 3D-ResNet or 3D-DenseNet.
Let the model itself find the spatial patterns of neurodegeneration.
Compare deep learning outcomes with previous interpretable models.
This multi-phased strategy should be effective because it reflects the thought process, from global indicators to the specific biomarkers, down to subtle spatial patterns.
Expected Challenges
Class imbalance: As the number of dementia patients is less than the number of non-demented patients, this may cause the model to be overfit. We are planning to use techniques such as class weighting, balanced sampling , and robust evaluation metrics like F1-score and Precision-Recall AUC.
Confounding impact of age: Since normal aging also causes structural brain changes that may resemble early Alzheimer’s patterns. To reduce this effect, age will be included as a feature in baseline models and we will conduct age-aware evaluations to ensure that age-related effects do not influence predictions.
​MRI Heterogeneity: This refers to variations in image quality. We will apply standard preprocessing techniques such as normalization, spatial alignment and brain masking to minimize variability across scans.
Also, there is a trade-off between interpretability and accuracy, particularly when we are using deep learning models. We will maintain interpretable baseline models and use them as reference points when evaluating the performance gains of more complex methods.
Evaluation Metrics
For validation, we will apply stratified cross-validation for traditional machine learning models and ensure a subject-level train, validation, and test split to prevent data leakage. 
We are planning to measure model performance using Accuracy, F1-score, ROC-AUC, and Precision-Recall AUC. Accuracy is a general performance metric. Since our data is imbalanced, F1-score and Precision-Recall AUC are more informative. We will use ROC-AUC to see if our model can really tell the difference between dementia and non-dementia cases when we try thresholds for the model.
We will also compare the performance of models developed in Phase 1 and Phase 2 to check improvements at each stage.
Team Roles:
Data Handling and Preprocessing: Jainil, Mansi
Model Development: Mansi, Mohit
Evaluation & analysis: Mohit, Jainil
Documentation & reporting: All team members
References:
Marcus, D. S., et al., “Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults,” Journal of Cognitive Neuroscience, 2007.
OASIS Brain Project. https://www.oasis-brains.org/

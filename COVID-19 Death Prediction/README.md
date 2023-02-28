# COVID-19 Death Prediction

A Deep Learning Model is developed in this project to predict death risk of COVID-19 patients.

## [Dataset](https://github.com/ahsan-83/Machine-Learning-Projects/tree/main/COVID-19%20Death%20Prediction/datasets)

[COVID-19 Dataset](https://www.kaggle.com/datasets/meirnizri/covid19-dataset) provided by the Mexican government. 

- Dataset consists of 21 unique features and of unique patients. 
- In the Boolean features, 1 means "yes" and 2 means "no". values as 97 and 99 are missing data.

**Dataset Informations**

- sex: 1 for female and 2 for male.
- age: of the patient.
- classification: covid test findings. Values 1-3 means that the patient was diagnosed with covid and otherwise not.
- patient type: type of care the patient received in the unit. 1 for returned home and 2 for hospitalization.
- pneumonia: whether the patient already have air sacs inflammation or not.
- pregnancy: whether the patient is pregnant or not.
- diabetes: whether the patient has diabetes or not.
- copd: Indicates whether the patient has Chronic obstructive pulmonary disease or not.
- asthma: whether the patient has asthma or not.
- inmsupr: whether the patient is immunosuppressed or not.
- hypertension: whether the patient has hypertension or not.
- cardiovascular: whether the patient has heart or blood vessels related disease.
- renal chronic: whether the patient has chronic renal disease or not.
- other disease: whether the patient has other disease or not.
- obesity: whether the patient is obese or not.
- tobacco: whether the patient is a tobacco user.
- usmr: Indicates whether the patient treated medical units of the first, second or third level.
- medical unit: type of institution of the National Health System that provided the care.
- intubed: whether the patient was connected to the ventilator.
- icu: Indicates whether the patient had been admitted to an Intensive Care Unit.
- date died: If the patient died indicate the date of death, and 9999-99-99 otherwise.

## Data Preprocessing and Visualization

**Death Distribution**

![](https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/COVID-19%20Death%20Prediction/resources/death_distribution.png)

- Death and alive case do not have even distribution.

**Feature-Death Bar Plot**

![](https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/COVID-19%20Death%20Prediction/resources/feature_death_barplot.png)

- More male died compared to female but overall, SEX feature had very little impact on covid deaths.
- INTUBED, ICU and PREGNANT feature has too many missing values.

**Age-Death Distribution**

![](https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/COVID-19%20Death%20Prediction/resources/age_death_distribution.png)

- Patients are roughly between 20-60 years old.
- The older patients are more likely to die compared to younger ones.

**Feature Correlations**

![](https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/COVID-19%20Death%20Prediction/resources/correlation_mat.png)

- COPD, ASTHMA, INMSUPR, OTHER_DISEASE, CARDIOVASCULAR, OBESITY, RENAL_CHRONIC, TOBACCO features has lower correlations.

## [Logistic Regression Model](https://github.com/ahsan-83/Machine-Learning-Projects/tree/main/COVID-19%20Death%20Prediction/notebook)

- Deep Learning Logistic Regression Model used for COVID-19 Death Prediction 
- Model contains 3 Dense layers with [128, 64, 32] units and RELU activation and 1 Dense layer unit with Sigmoid activation
- Batch Size : 512
- Learning Rate : 0.01
- Optimization Algo : Adam
- Loss : Binary Crossentropy

**Model Evaluation Metrics**

Model | Accuracy | Precision | Recall | F1 Score
--- | --- | --- | --- |--- 
LR Model | 0.938971 | 0.638825 | 0.369167 | 0.467926 

**Model Confusion Matrix**

![](https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/COVID-19%20Death%20Prediction/resources/LR_model_confusion_mat.png)

**Model Analysis**

- Logistic Regression Model achieved 93% test accuracy
- F1 Score is 47 which means we predicted the patients who survived well but we can't say the same thing for dead patients.

**Model Undersampling**

- Dataset is unbalanced in terms of dead and alive patients.
- Dataset is undersampled by keeping all dead patient records while shrinking alive patient records.

![](https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/COVID-19%20Death%20Prediction/resources/death_distribution_undersample.png)

**Model Evaluation Metrics for Undersampled Data**

Model | Accuracy | Precision | Recall | F1 Score
--- | --- | --- | --- |--- 
LR Model Under Sampled | 0.910423 | 0.882462 | 0.947661 | 0.913900 

**Model Confusion Matrix for Undersampled Data**

![](https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/COVID-19%20Death%20Prediction/resources/LR_undersample_model_confusion_mat.png)

## Model Comparison

![](https://github.com/ahsan-83/Machine-Learning-Projects/blob/main/COVID-19%20Death%20Prediction/resources/model_comparison.png)

- F1 Score increased from 47 to 91 after Under Sampling Data.
- Also Pression and Recall is higher in Under Sampled Model.


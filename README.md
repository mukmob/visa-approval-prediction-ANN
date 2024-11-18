# US Visa Approval Prediction

`MLOps Production Ready Machine Learning Classification Project On Visa Approval Prediction`

# Project Overview

### 1. Understanding the Problem Statement

* OFLC gives job certification applications for employers seeking to bring foreign workers into the United States and grants certifications.
* As In last year the count of employees were huge so OFLC needs Machine learning models to shortlist visa applicants based on their previous data.

**In this project i'm going to use the data given to build a Classification model:**

* This model is to check if Visa get approved or not based on the given dataset.
* This can be used to Recommend a suitable profile for the applicants for whom the visa should be certified or denied based on the certain criteria which influences the decision.



Features:

- case_id: ID of each visa application
- continent: Information of continent the employee
- education_of_employee: Information of education of the employee
- has_job_experience: Does the employee has any job experience? Y= Yes; N = No
- requires_job_training: Does the employee require any job training? Y = Yes; N = No
- no_of_employees: Number of employees in the employer's company
- yr_of_estab: Year in which the employer's company was established
- region_of_employment: Information of foreign worker's intended region of employment in the US.
- prevailing_wage: Average wage paid to similarly employed workers in a specific occupation in the area of intended employment. The purpose of the prevailing wage is to ensure that the foreign worker is not underpaid compared to other workers offering the same or similar service in the same area of employment.
- unit_of_wage: Unit of prevailing wage. Values include Hourly, Weekly, Monthly, and Yearly.
- full_time_position: Is the position of work full-time? Y = Full Time Position; N = Part Time Position
- case_status: Flag indicating if the Visa was certified or denied


### 2. Understanding the Solution

#### Solution scope:

1. This can be used on real life by US Visa Applicants so that they can improve their `Resume` and `Criteria` for the approval process.

2. Also this model can be used by US Visa officers to filter the candidate for visa approval automatically with involvement of much manual work or manpower, this overcome the manual task.

#### Solution Approch:

1. Machine Learning: ML Classification Algorithms
2. Deep Learning: Custom ANN with sigmoid activation function

#### Solution Procosed:

I will be using Machine Learning approcah to solve this problem.

1. Load the dataset from Database
2. Perform EDA and Feature Engineering to select the desirable features.
3. Fit the ML Classification Algorithms and find out which one performs better.
4. Select top performing few and tune with hyperparameters like Gridsearch CV.
5. Select the best model based on desired metrices.

### 3. Code Understanding and walkthrough

Use OOPs concept to make code clean and DRY

### 4. Understanding the Deployment

1. Docker
2. Cloud Services like AWS
3. Adding self Hosted Runner
4. MLFlow
5. Data Version Control (DVC)
6. Evidentlyai

#### Tools and dataset links

- Flowchart: https://whimsical.com/
- MLOPs Tool: https://www.evidentlyai.com/
- MongoDB: https://account.mongodb.com/account/login
- Data link: https://www.kaggle.com/datasets/moro23/easyvisa-dataset

## How to run?

```bash
conda create -n venv_visa python=3.11 -y
```

```bash
conda activate venv_visa
```

```bash
pip install -r requirements.txt
```

#### Insert all install packages with its version at requirement.txt

```bash
pip list
pip freeze > requirements.txt
```

## Workflow:

1. constants
2. entity
3. components
4. pipeline
5. Main file

### Export the environment variable

```bash


export MONGODB_URL="mongodb+srv://<username>:<password>...."

export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>


```

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

    #with specific access

    1. EC2 access : It is virtual machine

    2. ECR: Elastic Container registry to save your docker image in aws


    #Description: About the deployment

    1. Build docker image of the source code

    2. Push your docker image to ECR

    3. Launch Your EC2

    4. Pull Your image from ECR in EC2

    5. Lauch your docker image in EC2

    #Policy:

    1. AmazonEC2ContainerRegistryFullAccess

    2. AmazonEC2FullAccess

## 3. Create ECR repo to store/save docker image

    - Save the URI: 315865595366.dkr.ecr.us-east-1.amazonaws.com/visarepo

## 4. Create EC2 machine (Ubuntu)

## 5. Open EC2 and Install docker in EC2 Machine:

    #optinal

    sudo apt-get update -y

    sudo apt-get upgrade

    #required

    curl -fsSL https://get.docker.com -o get-docker.sh

    sudo sh get-docker.sh

    sudo usermod -aG docker ubuntu

    newgrp docker

# 6. Configure EC2 as self-hosted runner:

    setting>actions>runner>new self hosted runner> choose os> then run command one by one

# 7. Setup github secrets:

- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_DEFAULT_REGION
- ECR_REPO

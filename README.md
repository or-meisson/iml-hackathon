# Hackathon 2023 - Challenge 2: Breast Cancer Attribute Detection

## Project Overview
This repository contains the submission by Or Meissonnier, Shaked Mishory, Asaf Feldman, and Lotem Rozner for the "Introduction to Machine Learning (67577)" course's Hackathon at Hebrew University. The challenge focuses on detecting various attributes of breast cancer from anonymized patient data, aiming to aid in early and accurate diagnosis.

## Challenge Description
The task involves predicting specific medical characteristics of breast cancer using machine learning models based on provided patient data. This includes:
- **Part 1:** Predicting metastases sites.
- **Part 2:** Predicting tumor size.
- **Bonus Part 3:** Performing unsupervised data analysis to uncover interesting data trends.

## Data
The dataset consists of 65,798 anonymized records split into training and testing sets. Each record includes 34 features detailing patient visits, treatments, and outcomes.

## Repository Contents
- `Project.pdf`: Detailed challenge description and dataset features.
- Python scripts for preprocessing data, training models, and conducting unsupervised analysis.

## Models and Techniques Used
- **Random Forest and Decision Trees:** Used for predicting metastases sites and tumor sizes.
- **K-Means Clustering and PCA:** Applied in the unsupervised analysis to identify patterns and trends in the data.
- **Preprocessing Techniques:** Data cleaning, normalization, and transformation to prepare for model training.
- 
## Output Files
- `1.csv`: Contains predictions for the location of metastases as per Part 1 of the challenge.
- `2.csv`: Contains predictions for tumor size as per Part 2 of the challenge.


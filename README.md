# End-to-End Machine Learning Project: California Housing Prediction with Sklearn Pipeline

## Overview 
This project demonstrates a structured approach to data preprocessing, feature engineering, and model selection. The goal is to showcase key techniques in **exploratory data analysis (EDA), data transformation, and model fine-tuning**. The entire process is streamlined using **pipelines**, ensuring reproducibility and scalability. 

## 📊 Data Exploration & Transformation
A thorough **exploratory data analysis (EDA)** was performed to understand data distributions, relationships, and potential feature transformations. Key techniques used:

- **🌱 Handling Features with Heavy Tail:**
  - Used `SimpleImputer` to fill missing values.
  - Applied `StandardScaler` for feature scaling.
  - Used **log transformation** to handle heavy-tailed distributions.

- **🛠️ Handling Features with Multimodal Distribution:**
  - Used `SimpleImputer` with **most_frequent** strategy for missing categorical values.
  - Applied **OneHotEncoder** for categorical encoding.
  - Measured **cluster similarity** using **RBF kernel transformation**.

- **🌍 Geographical Data Processing:**
  - Applied **KMeans clustering** to identify spatial clusters.

- **🔄 Feature Engineering:**
  - Created **new attributes** through meaningful combinations.
  - Incorporated **cluster similarity** features to enhance data representation.

- **🎨 Visualization:**
  - Examined **geographical latitude and longitude** patterns with clustering.
  - Visualized **RBF kernel similarity effects** on the transformed feature space.
  
  <p align="center">
    <img src="California_Housing_Regression/images/geographical_cluster.png" alt="Geographical Latitude and Longitude Clustering" width="400"/>
  </p>
  
  <p align="center">
    <img src="California_Housing_Regression/images/rbf_similarity.png" alt="RBF Kernel Similarity Visualization" width="400"/>
  </p>

## 🛠️ Data Transformation Pipeline
To ensure consistency across training and testing datasets, a **Scikit-Learn Pipeline** was implemented. The pipeline integrates:

- **📝 Feature Engineering & Attribute Combination:**
  - Created new ratio-based attributes:
    - `bedrooms_per_room = total_bedrooms / total_rooms`
    - `rooms_per_house = total_rooms / households`
    - `people_per_house = population / households`

- **🔄 Handling Features with Heavy Tail:**
  - Applied **log transformation** to features such as `total_bedrooms`, `total_rooms`, `population`, `households`, and `median_income`.
  
  <p align="center">
    <img src="California_Housing_Regression/images/log_transformation.png" alt="Log Transformation to Deal with Heavy Tail" width="800"/>
  </p>

- **🌍 Geographical Feature Transformation:**
  - Used **KMeans clustering** to assign spatial clusters based on latitude and longitude.
  - Applied **RBF kernel transformation** to measure similarity to cluster centers.

- **🔢 Categorical Feature Encoding:**
  - Used `OneHotEncoder` to handle categorical features after imputing missing values with the most frequent category.

- **💪 Numerical Feature Scaling & Handling Missing Values:**
  - Used `SimpleImputer` for missing values.
  - Applied `StandardScaler` for feature scaling.
  
This approach eliminates data leakage risks and allows seamless integration of preprocessing steps into the machine learning workflow.

<p align="center">
  <img src="California_Housing_Regression/images/pipeline.png" alt="Final Pipeline Structure" width="1100"/>
</p>

## 🌟 Model Experimentation & Fine-Tuning
Multiple models were tested to identify the best-performing one. The following steps were taken:

- **📈 Baseline Model Testing:**
  - Experimented with **Linear Regression, Decision Tree, and Random Forest** models.
  
- **🎯 Hyperparameter Tuning:**
  - Applied **Grid Search** and **Randomized Search** to optimize model parameters.
  - Evaluated model performance using cross-validation to prevent overfitting.
  
  <p align="center">
    <img src="California_Housing_Regression/images/Grid_search.png" alt="Grid Search" width="1100"/>
     <img src="California_Housing_Regression/images/random_search.png" alt="Random Search" width="1100"/>
  </p>
  
  

- **🔬 Final Model Selection:**
  - Chose the most promising model based on accuracy metric
  - Conducted error analysis and feature importance studies.

## 📝 Key Takeaways
- A structured **preprocessing pipeline** improves model reproducibility and efficiency.
- **Feature engineering techniques** like attribute combination and cluster similarity enhance model performance.
- **Hyperparameter tuning** is essential for optimizing predictive models.

This project serves as a demonstration of **data transformation, model selection, and systematic experimentation** in machine learning. 🚀


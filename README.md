# Customer Churn Analysis using Machine Learning

## 📌 Project Overview
Customer churn occurs when existing customers stop doing business with a company. Since acquiring new customers is 5–7 times more expensive than retaining existing ones, predicting churn is crucial for business growth.

This project analyzes customer behavior data, identifies key factors driving churn, and builds machine learning models to predict customers likely to churn. With these insights, businesses can design proactive retention strategies such as targeted campaigns, personalized offers, and improved customer support.

---

## 🚀 Features
- Data preprocessing and handling class imbalance using **SMOTE**.
- Exploratory data analysis (EDA) with **visualizations**.
- Implementation of multiple ML models:
  - Decision Tree
  - Random Forest
  - XGBoost
- Model evaluation using:
  - Accuracy
  - Confusion Matrix
  - Classification Report
- Saving trained models with **Pickle** for future use.

---

## 📂 Dataset
The project uses a CSV file:

- **Customer_Churn_Prediction.csv**  
  Make sure to place the dataset in the project directory before running the notebook.

---

## 🛠️ Installation & Requirements

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/customer-churn-analysis.git
cd customer-churn-analysis
pip install -r requirements.txt
Requirements
Python 3.8+

numpy

pandas

matplotlib

seaborn

scikit-learn

imbalanced-learn

xgboost

You can also install them manually:

bash
Copy code
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
#📊 Usage
Run the Jupyter notebook to execute the analysis:

bash
Copy code
jupyter notebook Customer_Churn_Analysis_using_ML.ipynb
Steps performed:

##Import dependencies.

Load dataset (Customer_Churn_Prediction.csv).

Perform EDA and preprocessing.

Train multiple machine learning models.

Evaluate performance.

Save the best-performing model.

##📈Results
Random Forest and XGBoost generally outperform Decision Trees in accuracy.

The confusion matrix and classification report highlight the trade-off between precision and recall.

Feature importance helps identify key drivers of churn.

##💡 Future Improvements
Hyperparameter tuning with GridSearchCV/RandomizedSearchCV.

Experiment with deep learning models.

Deploy the model as a REST API or web app (e.g., Flask/Streamlit).









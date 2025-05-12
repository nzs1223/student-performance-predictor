# student-performance-predictor

Student Performance Predictor
This project focuses on predicting the performance of students based on various attributes using machine learning. The goal is to predict whether a student will pass or fail based on features like age, school, family background, study time, and more.

Dataset
The dataset used in this project is the Student Performance Dataset from the UCI Machine Learning Repository, which contains data from students' academic performance in secondary education. The dataset includes various features like age, gender, school, family size, and grades in multiple subjects.

Dataset source: https://archive.ics.uci.edu/dataset/320/student+performance
Project Structure
bash
Copy
Edit
student-performance-predictor/
├── student-mat.csv        # The dataset used in the project
├── student_performance_predictor.ipynb  # Jupyter notebook containing the analysis
├── README.md              # This file
Steps Performed
1. Data Loading
We loaded the dataset into a pandas DataFrame from a CSV file.

2. Data Preprocessing
We checked for missing values and cleaned the dataset.

Converted categorical variables into numerical values using label encoding.

Split the dataset into features (X) and target variable (y), where y is whether the student passes or fails.

3. Model Training
We used the RandomForestClassifier to train a model to predict student performance.

The dataset was split into a training and testing set using an 80-20 split.

4. Model Evaluation
The model's performance was evaluated using:

Classification Report for precision, recall, and F1-score.

Confusion Matrix to visualize the true positives, true negatives, false positives, and false negatives.

ROC AUC Score to evaluate the model's ability to distinguish between classes.

5. Results
After training and evaluating the model, the following metrics were obtained:

Accuracy: 95%

Precision (for passing students): 0.93

Recall (for passing students): 0.98

F1-Score (for passing students): 0.95

ROC AUC Score: 0.948

6. Conclusion
The model performs well in predicting student performance, with an overall accuracy of 95%. It is particularly good at identifying students who are likely to pass, with high recall and F1-score values.

Technologies Used
Python

Pandas

NumPy

scikit-learn

Jupyter Notebook

Installation
Clone this repository to your local machine.

bash
Copy
Edit
git clone https://github.com/yourusername/student-performance-predictor.git
Navigate to the project directory.

bash
Copy
Edit
cd student-performance-predictor
Install the necessary libraries.

bash
Copy
Edit
pip install -r requirements.txt
Open the Jupyter notebook and run the cells to execute the code.

bash
Copy
Edit
jupyter notebook student_performance_predictor.ipynb

# Sonar Rock vs Mine Prediction

This project aims to build a machine learning model that can accurately distinguish between sonar signals bouncing off a rock or a mine-like object. The model is trained on a dataset containing 60 numerical features extracted from sonar return signals, along with a label indicating whether the object is a rock or a mine.

## Dataset

The dataset used in this project is the [Sonar Data Set](http://archive.ics.uci.edu/ml/datasets/Sonar) from the UCI Machine Learning Repository. It contains 208 instances with 60 features and a binary label (R for rock or M for mine).

## Approach

The project follows these steps:

1. **Data Preprocessing**: The dataset is loaded into a Pandas DataFrame, and the features (X) and labels (y) are separated.
2. **Exploratory Data Analysis**: Basic statistical analysis is performed on the dataset to gain insights into the data distribution.
3. **Data Splitting**: The dataset is split into training and test sets using scikit-learn's `train_test_split` function, with a 10% test set size and stratified sampling to maintain the class balance.
4. **Model Training**: A Logistic Regression model from scikit-learn is trained on the training data.
5. **Model Evaluation**: The trained model's performance is evaluated on both the training and test sets using accuracy as the metric.
6. **Prediction System**: A simple predictive system is implemented, where the user can input a set of 60 numerical features, and the model predicts whether the object is a rock or a mine.

## Requirements

- Python 3.x
- NumPy
- Pandas
- scikit-learn

## Usage

1. Clone the repository: `git clone https://github.com/your-username/sonar-rock-mine-prediction.git`
2. Navigate to the project directory: `cd sonar-rock-mine-prediction`
3. Open the Jupyter Notebook: `jupyter notebook SONAR Rock vs Mine Prediction.ipynb`
4. Run the notebook cells sequentially to train the model and make predictions.

## Results

The Logistic Regression model achieved an accuracy of approximately 83% on the training data and 76% on the test data. These results showcase the model's capability to distinguish between rocks and mines based on sonar return signals.

## Future Work

- Explore and compare the performance of other machine learning algorithms on this dataset.
- Perform feature selection or engineering to improve model performance.
- Investigate techniques for handling potential class imbalance in the dataset.

## Credits

The dataset used in this project is from the UCI Machine Learning Repository: [Sonar Data Set](http://archive.ics.uci.edu/ml/datasets/Sonar).

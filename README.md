# Classification-and-Regression-Models-Using-Deep-Learning
Here's a README file based on the information you provided:

---

# CS 5588 Big Data Analytics and Application
## Hands-On Session Plan: Classification and Regression Models

### Date: January 30, 2025

### Overview
In this session, students will learn to build and train feedforward neural networks for both classification and regression tasks using PyTorch. The session includes an instructor-led task using the **Diabetes Dataset** and a student activity using the **California Housing Dataset**. The session concludes with documenting the work in GitHub and completing a reflective survey.

### Submission Requirements:
- **Survey Form**: Complete the survey at the end of the session.
- **GitHub Repository**: Submit completed notebooks with detailed documentation.

---

### Objectives:
- Build and train feedforward neural networks for classification and regression tasks.
- Utilize GPU acceleration in **Google Colab** for faster model training.
- Document and share work using **GitHub**.

### Datasets:
#### 1. Diabetes Dataset (Instructor-Led):
- **Features**: 10 baseline variables for 442 diabetes patients (e.g., age, BMI, blood pressure).
- **Tasks**:
    - **Regression**: Predict the progression of diabetes.
    - **Classification**: Classify patients as "low progression" or "high progression".
- **Dataset URL**: [Diabetes Dataset on Kaggle](https://www.kaggle.com/)

#### 2. California Housing Dataset (Student Activity):
- **Features**: Census data (e.g., median income, housing median age, average occupancy).
- **Tasks**:
    - **Regression**: Predict median house values.
    - **Classification**: Classify houses as "low value" or "high value".
- **Dataset URL**: [California Housing Dataset on Kaggle](https://www.kaggle.com/)

---

### Session Flow:

#### 1. Google Colab Setup:
- Initialize **Google Colab** and enable **GPU acceleration**.
- Verify GPU availability with `torch.cuda.is_available()`.

#### 2. Diabetes Dataset (Instructor-Led):
**Regression Task**:
- **Objective**: Predict diabetes progression.
    - Load data using `load_diabetes()` from `sklearn.datasets`.
    - Preprocess with `StandardScaler`.
    - Split into training/test sets and convert to PyTorch tensors.
    - Build a feedforward neural network.
    - Use **MSELoss** and **Adam optimizer** for training (100 epochs).
    - Evaluate with **Mean Squared Error** and **R-squared** metrics.
    - Visualize predictions vs. true values.

**Classification Task**:
- **Objective**: Classify patients as "low progression" or "high progression".
    - Use binary labels (1: high progression, 0: low progression).
    - Use the same neural network architecture with softmax activation.
    - Use **CrossEntropyLoss** and **Adam optimizer** for training (100 epochs).
    - Evaluate with metrics like accuracy, precision, recall, F1-score, and confusion matrix.

#### 3. California Housing Dataset (Student Activity):
**Regression Task**:
- **Objective**: Predict median house values.
    - Load data using `fetch_california_housing()` from `sklearn.datasets`.
    - Preprocess and split data, converting to PyTorch tensors.
    - Replicate neural network from Diabetes regression task.
    - Train using **MSELoss** and **Adam optimizer** (100 epochs).
    - Evaluate with **Mean Squared Error** and **R-squared** metrics.

**Classification Task**:
- **Objective**: Classify houses as "low value" or "high value".
    - Use binary labels based on median house value.
    - Replicate neural network from Diabetes classification task.
    - Train using **CrossEntropyLoss** and **Adam optimizer** (100 epochs).
    - Evaluate with metrics like accuracy, precision, recall, F1-score, and confusion matrix.

---

---

### Wrap-Up:
#### Key Takeaways:
- Experience in building feedforward neural networks for regression and classification.
- Understanding of evaluation metrics like **MSE**, **R-squared**, **accuracy**, **precision**, **recall**, and **F1-score**.
- Familiarity with **GitHub** for professional documentation and sharing work.

#### Survey:
- Complete a short survey reflecting on:
    - Challenges faced.
    - Lessons learned.
    - Suggestions for improving future sessions.

---

### Resources:
- [Google Colab](https://colab.research.google.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [Kaggle - Diabetes Dataset](https://www.kaggle.com/)
- [Kaggle - California Housing Dataset](https://www.kaggle.com/)

---

Let me know if you'd like to adjust or add anything!

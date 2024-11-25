Sentiment Analysis Using Feedforward Neural Network and Word2Vec Embeddings

Overview
This project builds a deep learning model to classify sentiments (positive or negative) from text data using a Feedforward Neural Network (FNN) and Word2Vec embeddings. The model is trained on the provided dataset (SentimentData.csv), which contains textual reviews and their corresponding sentiment labels.

Requirements
The project requires the following libraries and tools to run:

Python (3.x)
Jupyter Notebook (for running the .ipynb file)
Required Python libraries:
pandas
numpy
matplotlib
seaborn
scikit-learn
gensim
tensorflow or keras
You can install the required libraries using the following command:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn gensim tensorflow

Files in the Project
Deeplearning.ipynb: Jupyter Notebook containing the implementation of:

Data preprocessing
Word2Vec embeddings
Feedforward Neural Network model
Model training and evaluation
Visualization of evaluation metrics (AUC-ROC, Precision-Recall curve, etc.)
SentimentData.csv: The dataset used for training and testing the model.

Columns:
review: The text review.
sentiment: The sentiment label (positive or negative).
Steps to Run the Project
Step 1: Prepare the Environment
Ensure all required libraries are installed.
Place the files Deeplearning.ipynb and SentimentData.csv in the same directory.
Step 2: Open the Jupyter Notebook
Launch Jupyter Notebook:
bash
Copy code
jupyter notebook
Open the Deeplearning.ipynb file.
Step 3: Run the Notebook
Run each cell in sequence to:
Load and preprocess the dataset.
Train the Word2Vec model (or load pre-trained embeddings).
Train the Feedforward Neural Network model.
Evaluate the model and visualize metrics.
Step 4: View Outputs
Outputs include:
Accuracy, Precision, Recall, F1-score
AUC-ROC curve
Precision-Recall curve
Confusion Matrix
Results
The project provides insights into the effectiveness of combining Word2Vec embeddings with a Feedforward Neural Network for sentiment classification.

Model Accuracy: 0.85
AUC-ROC Score: 0.93
F1 Score: 0.85
Notes
For issues with compatibility or library versions, ensure TensorFlow is updated:
bash
Copy code
pip install --upgrade tensorflow
Modify hyperparameters in the notebook (e.g., learning rate, batch size, or embedding size) to experiment with model performance.
Authors
Group Members: [Add your names here]
Tasks: [Briefly describe which member handled specific tasks]

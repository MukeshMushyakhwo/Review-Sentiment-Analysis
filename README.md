
# Review Sentiment Analysis ( Model Evaluation)

Sentiment analysis involves determining the sentiment (positive or negative) expressed in textual data. Sentiment analysis plays a crucial role in understanding public opinions and emotions from text data, making it a valuable tool in analyzing movie reviews. In this project, it analyze movie reviews to classify them as positive or negative based on their sentiment. We will explore two different models: Logistic Regression and Long Short-Term Memory (LSTM), showcasing their implementation, training, evaluation, and sentiment prediction. A comprehensive comparison of their performance will be presented.


## Logistic Regression Model
The Logistic Regression model is a classical machine learning algorithm that is widely used for binary classification tasks, such as sentiment analysis. In this section, we'll guide you through the process of building, training, and evaluating a Logistic Regression model for sentiment analysis.

Build and train a Logistic Regression model:
Fit the model using the TF-IDF transformed training data.
Evaluate the model's performance:
Calculate accuracy and F1-score metrics.
Generate a confusion matrix and classification report:
Visualize the model's performance.

* F1-score: 86%
* Confusion matrix
![](https://github.com/MukeshMushyakhwo/Review-Sentiment-Analysis/blob/main/Evaluaiton%20graph/lr_cm.png?raw=true)



## LSTM 

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture designed to address the vanishing gradient problem and capture long-range dependencies in sequential data. LSTMs are particularly effective for tasks involving sequential data, such as natural language processing (NLP), speech recognition, time series prediction, and more. They have the ability to learn and remember information over long periods of time, making them well-suited for tasks that involve analyzing sequences with dependencies that span across multiple time steps.

Build and train an LSTM model using TensorFlow. Implement early stopping to prevent overfitting during training. Monitor the training and validation loss and accuracy to select the best epoch.

* Accuracy: 82%

* Confusion Matrix

![](https://github.com/MukeshMushyakhwo/Review-Sentiment-Analysis/blob/main/Evaluaiton%20graph/lstm_cm.png?raw=true)

***
* Training and Validation loss 
![](https://github.com/MukeshMushyakhwo/Review-Sentiment-Analysis/blob/main/Evaluaiton%20graph/lstm_trian_val_loss.png?raw=true)


The illustration represents the training and validation loss of an LSTM model:

The loss metric indicates how well the model's predicted probabilities match the true labels. Lower loss values suggest that the model's predictions are closer to the ground truth.

In Epoch 1, the training loss is relatively high at 5.7554, and the validation loss is 0.7161.
The model quickly reduces the loss over subsequent epochs. By Epoch 11, the training loss is 0.3595, and the validation loss is 0.4763.

The model demonstrates significant improvements in both accuracy and loss as the training progresses. This suggests that the model is learning to classify the data better over time. However, it's worth noting that the validation loss starts to increase after a certain point (Epoch 7) while the training loss continues to decrease. This could be an indication of overfitting, where the model begins to perform well on the training data but struggles to generalize to new, unseen data.

## Model Evaluation
The Logistic Regression model outperforms the LSTM model in terms of accuracy, with an accuracy of 0.86408 compared to the LSTM model's accuracy of 0.8218. This indicates that the Logistic Regression model was better at correctly classifying sentiments on the given dataset.

## Conclusion
In this sentiment analysis project, we have walked through the entire process of analyzing movie reviews to determine sentiment using two distinct models: Logistic Regression and LSTM. By following the detailed steps provided in this documentation, you can preprocess data, train models, evaluate their performance, and make sentiment predictions on new input text. Additionally, the comparison between models offers insights into their respective capabilities and potential applications.

# Streamlit Web Application
Logistic Regression model was dumbed due to it's high accuracy to perform sentiment classification on streamlit.

**Install Streamlit**

If you haven't already, you need to install Streamlit. You can do this using pip:

`pip install streamlit`


**Run the Streamlit App**:

To run your Streamlit app, open a terminal and navigate to the directory containing your app.py file. Then, run the following command:

`streamlit run app.py`


### Streamlit output

**Positive Review**
![](https://github.com/MukeshMushyakhwo/Review-Sentiment-Analysis/blob/main/output%20screenshot/positive.png?raw=true)

**Negative Review**
![](https://github.com/MukeshMushyakhwo/Review-Sentiment-Analysis/blob/main/output%20screenshot/negative.png?raw=true)



# Multi-Class Image Classification using EfficientNetV2

In this project, I've implemented a multi-class image classification task using the EfficientNetV2B3 model. I've broken down the process into several steps and provided detailed explanations along the way.

## 1. Importing Libraries
I started by importing all the necessary libraries for data manipulation, visualization, model building, and evaluation. These include NumPy, pandas, OpenCV (cv2), TensorFlow, and others.

## 2. Reading Data
Next, I mounted my Google Drive and extracted the dataset from a ZIP file using the `ZipFile` module. Then, I looped through the training and test directories to count the number of images in each folder and printed out the counts.

## 3. Data Visualization
To gain insights into the data, I visualized a random selection of images from the training dataset along with their corresponding labels using Matplotlib.

## 4. Splitting Dataset
I defined the input shape, number of classes, and batch size. Then, I loaded the training, validation, and test datasets using `tf.keras.preprocessing.image_dataset_from_directory`, specifying the image size, batch size, and validation split.

## 5. Early Stopping and Model Checkpoint
To prevent overfitting and save the best model during training, I set up EarlyStopping and ModelCheckpoint callbacks to monitor the validation loss and accuracy.

## 6. Model Building
I imported the EfficientNetV2B3 model from Keras applications and added it as a layer to a Sequential model. Then, I added a Flatten layer followed by a Dense layer with softmax activation for classification. Finally, I compiled the model using Adam optimizer and sparse categorical crossentropy loss.

## 7. Model Training and Validation
I trained the model for 15 epochs using the training and validation datasets, utilizing the EarlyStopping and ModelCheckpoint callbacks to monitor performance and save the best model.

## 8. Model Evaluation and Visualization
After training, I evaluated the model's performance on both the training and validation datasets. I visualized the training and validation accuracy and loss over epochs using Matplotlib to assess model performance.

## 9. Loading the Best Model
Finally, I loaded the best model saved during training using the ModelCheckpoint callback for further evaluation and testing.

## 10. Testing
After training and validating the model, I proceeded to test its performance on the unseen test dataset. Here's what I did:

- **Model Predictions:** I used the loaded model to make predictions on the test set (`test_set`).
  
- **Conversion to Class Labels:** The predicted probabilities were converted to class labels using `np.argmax()` to determine the class with the highest probability.

- **Evaluation:** I evaluated the model's performance on the test set and obtained the final test accuracy.

- **Classification Report:** To further assess the model's performance, I generated a classification report using `classification_report` from scikit-learn. This report provides precision, recall, F1-score, and support for each class.

- **Confusion Matrix:** Additionally, I calculated the confusion matrix to visualize the model's performance across different classes.

--- 

## Conclusion
This project demonstrates a comprehensive approach to multi-class image classification using the EfficientNetV2B3 model.


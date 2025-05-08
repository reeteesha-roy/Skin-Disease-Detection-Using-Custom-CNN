# Objective

A deep learning-based project that classifies skin diseases using a custom Convolutional Neural Network (CNN). This project explores how well a CNN can learn from image data alone—leveraging data augmentation, multiple convolutional blocks, and extensive evaluation.The main aim of this project was to build and evaluate a custom CNN model capable of accurately classifying images of 20 different skin diseases using the "20 Skin Diseases Dataset" and analyze how CNNs perform on complex medical image data.

### Dataset Used: [20 Skin Diseases Dataset](https://www.kaggle.com/datasets/haroonalam16/20-skin-diseases-dataset)   

### Structure:
- Images are organized into folders per class.
- The dataset contains over 10,000+ high-resolution skin images.
- Each image is labeled by disease type.

# Project Overview

### 1. Data Loading & Preprocessing:
   
- Images loaded and resized to 192x192.
- Labels created using one-hot encoding.
- Data split into training and validation sets.
- Visualized random image-label pairs for verification.

### 2. Data Augmentation:

- Rotation, flipping, zoom, shear, and shifts applied using ImageDataGenerator.
- Normalization (rescaling) applied for better training convergence.

### 3. Model Architecture:
- Custom CNN with 4 convolutional blocks.
- Final layers include flattening, fully connected layers, and softmax output.
- Regularization through max-pooling and ReLU activation.

### 4. Training:
- Trained for 20 epochs using Adamax optimizer and categorical crossentropy.
- Used ModelCheckpoint to save model after each epoch.

### 5. Evaluation:
- Plotted training/validation accuracy and loss.
- Visualized predictions on test images.
- Generated and visualized a confusion matrix.

# Dependencies & Libraries Used

- numpy : Array operations
- pandas : Data manipulation (minor use)
- matplotlib, seaborn: Data and result visualization
- opencv-python (cv2): Image loading and processing
- tensorflow: Building and training the CNN
- scikit-learn: Model evaluation (train_test_split, confusion_matrix)

Install libraries using:
<pre> 'pip install numpy pandas matplotlib seaborn opencv-python tensorflow scikit-learn' </pre>

# Model Performance:

- Final Training Accuracy: 95.5%
- Final Validation Accuracy: 96.9% (subject to variation per run)
- Loss Trends: Converged smoothly with minimal overfitting (1.2%)
- Confusion Matrix: Reveals confusion between similar-looking diseases (visualized below)


# Observation and Conclusion:

CNNs are powerful in extracting spatial hierarchies in medical images. Data augmentation significantly boosts generalization in limited-data scenarios. Deeper convolutional stacks (like 4 blocks here) help the model grasp complex patterns, especially in high-res images.Even a custom CNN, without transfer learning, can perform well with proper tuning and preprocessing.

# Results

### Training Accuracy & Loss
![image](https://github.com/user-attachments/assets/2aafedc8-8bf0-4cda-a47f-550ccbff5595)

### Validation Accuracy & Loss
![image](https://github.com/user-attachments/assets/77d117ae-0f77-4de6-8e8a-e3e3723b2919)

### Predicted Images
![image](https://github.com/user-attachments/assets/e1f57eca-5f3c-497b-b46a-b9c4d8cb8b3c)

### Confusion Matrix
![image](https://github.com/user-attachments/assets/d84feb25-ee4e-467b-8009-bcb2250e9e09)


> Precautions: Medical datasets often have class imbalance and inter-class similarity, which challenges even robust CNNs. 

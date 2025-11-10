# week-2-AI-Based-Crop-Disease-Prediction
This project leverages the PlantVillage Dataset (Resized 224×224) to develop an AI-driven system for monitoring crop health. The dataset contains thousands of leaf images categorized as healthy or diseased, making it ideal for training and testing deep learning models in plant disease recognition.

Introduction

Agriculture plays a vital role in the economy of developing countries like India.
Crop diseases cause significant losses in yield and quality every year. Traditionally, farmers identify plant diseases manually through visual inspection, which is time-consuming, inaccurate, and depends on expert knowledge.

With advancements in Artificial Intelligence (AI) and Deep Learning, image-based disease prediction has become an efficient and reliable method. This project focuses on developing a deep learning model that can automatically identify crop diseases from leaf images.

2️⃣ Objective

The main objective of this project is to:

Detect whether a plant leaf is healthy or diseased.

Identify the specific type of disease affecting the plant.

Assist farmers with early diagnosis to improve crop yield and reduce losses.

3️⃣ Dataset Description

The project uses the PlantVillage Dataset, which is a large collection of images of healthy and diseased crop leaves.

Key points:

It contains around 54,000+ images of different crops.

Each crop has multiple disease categories plus a healthy category.

The images are taken under controlled conditions with a plain background.

The dataset includes crops like apple, tomato, potato, corn, grape, and others.

4️⃣ Methodology
(a) Data Collection

Images are collected from the PlantVillage dataset on Kaggle. Each image is labeled with the crop name and disease type.

(b) Data Preprocessing

Before training, images are processed to make them suitable for AI models:

Resizing: All images are resized to a uniform dimension (e.g., 224×224 pixels).

Normalization: Pixel values are scaled between 0 and 1 for faster convergence.

Augmentation: Random transformations (rotation, zoom, flipping) are applied to increase dataset diversity and prevent overfitting.

Splitting: Dataset is divided into training, validation, and testing sets.

(c) Feature Extraction

Deep learning models automatically extract important features such as:

Leaf color patterns

Texture differences

Shape and spot variations
These features help in distinguishing between healthy and diseased leaves.

(d) Model Building

A Convolutional Neural Network (CNN) is used for image classification.
Alternatively, Transfer Learning models such as EfficientNet, ResNet, or MobileNet can be used to improve accuracy.

The model consists of:

Convolutional Layers – extract features from images

Pooling Layers – reduce spatial dimensions

Flattening Layer – converts 2D features into 1D

Fully Connected Layers – classify images into respective disease categories

Softmax Layer – outputs probability scores for each class

(e) Model Training

The model is trained on labeled images to minimize the categorical cross-entropy loss using an optimizer (like Adam).
During training, the model learns patterns that distinguish different diseases.

(f) Model Evaluation

After training, the model is tested using unseen images to measure:

Accuracy

Precision

Recall

Confusion Matrix

These metrics help evaluate the performance and reliability of the model.

(g) Prediction

For any new leaf image:

The image is preprocessed (resized and normalized).

It is passed through the trained model.

The model predicts the probable disease category of the leaf.

5️⃣ Results and Discussion

The model achieves high classification accuracy (up to 95–99%) on validation data.

The use of transfer learning helps achieve good results even with limited computational resources.

The system can distinguish multiple diseases of the same crop effectively.

However, performance may drop slightly in real-world conditions due to background noise and lighting differences.

6️⃣ Applications

Farmers: Early disease detection using mobile apps.

Agricultural Research: Monitoring crop health and disease spread.

Smart Farming: Integrating with IoT sensors and drone imaging.

Government / NGOs: Crop disease mapping for preventive measures.

7️⃣ Advantages

Fast and accurate disease identification

Reduces dependency on expert supervision

Supports real-time monitoring through AI integration

Scalable for multiple crops and regions

8️⃣ Limitations

Model performance may depend on lighting, camera quality, and background.

The dataset images are often captured in controlled environments, not real fields.

Requires large and balanced datasets for best performance.

9️⃣ Future Scope

Improve real-world accuracy using field images and domain adaptation.

Use object detection (YOLO, Mask R-CNN) for localizing diseased regions on leaves.

Build a mobile application for farmers to capture and analyze leaf images instantly.

Integrate with IoT-based smart farming systems for automatic monitoring.

🔟 Conclusion

AI-based crop disease prediction provides a powerful and scalable solution to the challenges of manual disease detection.
By using deep learning models trained on large datasets like PlantVillage, we can achieve accurate, fast, and cost-effective disease identification — contributing significantly to smart agriculture and food security.


# output 
<img width="1190" height="590" alt="4c2d99e1-ef9d-49eb-9630-57162a069622" src="https://github.com/user-attachments/assets/745a871b-f6f2-4008-8300-d6d10e9195e3" />
<img width="636" height="658" alt="11eb6709-d123-4023-b065-11c46cfa6278" src="https://github.com/user-attachments/assets/7475064c-0342-42ee-9fa2-479fa701ad0e" />
<img width="959" height="985" alt="3472f6b7-b728-4cde-98db-89fa4f60ef77" src="https://github.com/user-attachments/assets/c34af10e-6e31-4fb1-abea-e282ef037404" />

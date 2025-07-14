# Cat-vs-Dog-image-classification

This project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images of dogs and cats.

## 📘 Project Description

The objective is to train a deep learning model that can accurately distinguish between dog and cat images using supervised learning techniques.

## 🛠️ Technologies Used

- Python 🐍
- Google Colab ☁️
- TensorFlow / Keras
- NumPy & Matplotlib
- OpenCV / PIL (for image processing)

## 🧠 Model Architecture

- **Input Layer**: 256x256 RGB Images
- **Conv2D + MaxPooling** layers
- **Flatten** layer
- **Dense** layers with `ReLU` activation
- **Output Layer** with `Sigmoid` activation

## 📂 Dataset

- 2 Classes: Dog, Cat
- Data source: [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data) (or custom-uploaded images)
- Each image labeled accordingly

## 🚀 How to Run

1. Open the notebook `Dog_Vs_Cat_Image_Classification.ipynb` in Google Colab
2. Upload training and testing image folders
3. Run each cell to:
   - Preprocess data
   - Define and compile the CNN model
   - Train the model
   - Evaluate performance

## 📊 Future Improvements

- Data Augmentation
- Use Pretrained Models (Transfer Learning with ResNet, VGG, etc.)
- Deploy model as a web app using Flask or Streamlit

## 👩‍💻 Author

Pooja R B  



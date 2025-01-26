Potato Disease Classification with Flask and CNN
Project Overview
This project is designed to classify potato leaf diseases using a custom Convolutional Neural Network (CNN) and a Flask web application. The system allows farmers to upload images of potato leaves, and it classifies the leaves into categories: healthy, early blight, and late blight. The results are displayed with confidence levels, and feedback can be provided by farmers.
#MAIN AGENDA : This project compares the two model : 1) inside training folder: libraryused.ibynb that used all the tensorflows libraries to train the model.
                                                      2) inside trainmodel folder: train_model.py that used the partial ml libraries and self custom cnn using custom serializable.

Features
Image Classification: The system uses a trained CNN model to classify uploaded potato leaf images into three categories: Healthy, Early Blight, and Late Blight.
Admin Dashboard: Admins can log in to view uploaded images, classification results, and statistics. They can also view feedback from farmers.
Farmer Feedback: Farmers can submit feedback regarding the classification results, which are stored in the database.
Charts: Admins can view classification trends through a bar chart showing the number of classifications for each disease type.
Installation
Requirements
Python 3.x
TensorFlow
Flask
SQLite
Matplotlib
Other dependencies (specified in requirements.txt)
Steps to Set Up
Clone this repository:

bash
Copy
Edit
git clone https://github.com/yourusername/potato-disease-classification.git
cd potato-disease-classification
Create a virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # For Windows, use venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Set up the SQLite database:

bash
Copy
Edit
python create_db.py
Train the CNN model (if you haven't already) by running:

bash
Copy
Edit
python train.py
Run the Flask application:

bash
Copy
Edit
python app.py
Visit the app in your browser:

bash
Copy
Edit
http://127.0.0.1:5000/
File Descriptions
app.py: The main Flask application file that handles routing, file uploads, and database interactions.
train.py: Script to train the CNN model using the provided dataset of potato leaf images.
static/uploads/: Directory where uploaded images are stored.
static/charts/: Directory where classification trend charts are saved.
templates/: Directory containing HTML files used by Flask to render pages.
Database Structure
The project uses SQLite for storing the following data:

uploads: Stores information about each uploaded image, including the farmer's name, image path, classification result, and confidence score.
feedback: Stores feedback messages submitted by farmers.
admin: Stores admin credentials for login purposes.
Model Overview
The CNN model is a custom architecture built using TensorFlow and Keras. It consists of three convolutional layers, each followed by max pooling and dropout layers to prevent overfitting. Data augmentation is applied during training to increase the model's robustness.

Usage
Uploading Images
Farmers can visit the home page, upload an image of a potato leaf, and enter their name.
The image will be classified as healthy, early blight, or late blight, and the classification result will be displayed with the confidence percentage.
Admin Login
Admins can log in using a username and password stored in the database.
After logging in, admins can view uploaded images, classification statistics, and feedback from farmers.
Feedback
Farmers can provide feedback about the classification results. This feedback is stored in the database and can be accessed by the admin.
Model Training
The model is trained on a dataset of potato leaf images from various sources, including healthy leaves and those affected by early blight and late blight. The model architecture includes custom convolutional layers and regularization techniques to ensure good generalization to unseen data.

Training Data
The dataset is expected to be structured in the following way:

markdown
Copy
Edit
Plantvillage/
    Healthy/
        healthy_image_1.jpg
        healthy_image_2.jpg
        ...
    Early_Blight/
        early_blight_image_1.jpg
        early_blight_image_2.jpg
        ...
    Late_Blight/
        late_blight_image_1.jpg
        late_blight_image_2.jpg
        ...

  #for datasets
  you can download the dataset of PlantVillage from kaggle or go to tags the click releases and click assets, you will find PlantVillage.zip file extract it and kept it under training folders.

  #FOR TRAINED MODEL
  1. The trained model of libraryused.ibynb of training folder is inside the tags/releases/assets as model.zip, extract it and kept it under training folder.
  2. The trained model using custom serializable i.e train_model.py is inside the tags/release/assets as custom_plant_disease_model.zip, extract it and keep outside.


#PROJECT DEMONSTRATION 
1. Home page
![Screenshot (276)](https://github.com/user-attachments/assets/f3bdbc45-10bd-4260-8ba9-eea841f9076a)

2. Results of custom serializable objects model(using custom_plant_disease_model.keras)
   ![image](https://github.com/user-attachments/assets/6ad5033a-84db-48c6-816c-129c46793a68)

   ![image](https://github.com/user-attachments/assets/4b1b33e6-3b99-45f4-a673-748ad8ad6aa2)

3. results of using model.keras trained file with all ml libraries:
    ![image](https://github.com/user-attachments/assets/e26480a2-5694-416e-bb17-2b9f69ac1e71)

   ![Screenshot (282)](https://github.com/user-attachments/assets/1f31a975-0c47-4a52-a96a-c749e17b6180)






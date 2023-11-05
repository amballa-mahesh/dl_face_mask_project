"# dl_face_mask_project" 
Readme file.

LIVE HOST WEB ADDRESS: https://face-mask-detection-ggc0.onrender.com	

This the Facemask Detector.

Steps involved in creating this model are - 

EDA - 

reading data of the directory containing the image files using Pandas
converting dependent variable to numerics
data spliting to dependent and independent variables
changing the dimensions and preprocess them for the pre trained DL models like Mobilenetv2



Model Creation-

data spliting to dependent and independent variables
create the Deeplearing model using mobelnetv2
train the model using early stopping and save the model for the feature reuse



Prediction:
using the best trained model to perdict the test data of images
finding the model performance using the accuracy score, confusion matix and classification report.

Creation of User GUI-

Using the flask library we created the use GUI with HTML and CSS.
Deploy this model in local server.
get the values of the feilds selected by the user by flask
check the user can access the live feed so that the prediction happens on the rectangular box using the xml file to create the border
get the predictions from the model.


Using the logging

We will write back the logs to the logs.log file

Updating the data to mysql.

from the front end user interface get the values of selected feilds and save them back to local database by python mysql connector as well as the cassandra data base.
downloadt the data from the database and share....

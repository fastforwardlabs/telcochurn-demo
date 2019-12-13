# Refractor (or churnexplainer)

This is the CML port of the Refractor prototype which is part of the [Interpretability
report from Cloudera Fast Forward Labs](https://clients.fastforwardlabs.com/ff06/report).

## CML Applications: Train and inspect a new model locally

This project uses the Applications feature of CML (>=1.2) and CDSW (>=1.7) to instantiate a UI frontend for visual interpretability and decision management.  

### Train a predictor model
A model has been pre-trained and placed in the models directory.  
Start a Python 3 Session with at least 8GB of memory and __run utils/setup.py__.  This will create the minimum setup to use existing, pretrained models.  

If you want to retrain the model start a Session with at least 8GB memory and run:  
```!pip3 install -r utils/requirements3.txt```

After installing requirements run the train.py code to train a new model.  

The model artifact will be saved in the models directory named after the datestamp, dataset and algorithm of training (ie. 20191120T161757_ibm_linear). The default settings will create a linear regression model against the IBM telco dataset. However, the code is vary modular and can train multiple model types against essentially any tabular dataset.  

Look at train_multiple.py for examples.  


### Deploy a Predictor and Explainer models
Go to the **Models** section and create a new predictor model with the following:
* **Name**: Predictor
* **Description**: Predict customer churn
* **File**: predictor.py
* **Function**: predict
* **Input**: 
`{"StreamingTV":"No","MonthlyCharges":70.35,"PhoneService":"No","PaperlessBilling":"No","Partner":"No","OnlineBackup":"No","gender":"Female","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"}`  
* **Kernel**: Python 3

If you created your own model (see above)
* Click on "Set Environment Variables" and add:
  * **Name**: CHURN_MODEL_NAME
  * **Value**: 20191120T161757_ibm_linear  **your model name from above**
  Click "Add" and "Deploy Model"

Create a new Explainer model with the following:

* **Name**: Explainer
* **Description**: Explain customer churn prediction
* **File**: explainer.py
* **Function**: explain
* **Input**: `{"StreamingTV":"No","MonthlyCharges":70.35,"PhoneService":"No","PaperlessBilling":"No","Partner":"No","OnlineBackup":"No","gender":"Female","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"}`
* **Kernel**: Python 3

If you created your own model (see above)
* Click on "Set Environment Variables" and add:
  * **Name**: CHURN_MODEL_NAME
  * **Value**: 20191120T161757_ibm_linear  **your model name from above**
  Click "Add" and "Deploy Model"

In the deployed Explainer model -> Settings note (copy) the "Access Key" (ie. mukd9sit7tacnfq2phhn3whc4unq1f38)


### Instatiate the flask UI application
From the Project level click on "Open Workbench" (note you don't actually have to Launch a session) in order to edit a file.
Select the flask/single_view.html file and paste the Access Key in at line 19. 
Save and go back to the Project.


Go to the **Applications** section and select "New Application" with the following:
* **Name**: Visual Churn Analysis
* **Subdomain**: telco-churn
* **Script**: flask_app.py
* **Kernel**: Python 3
* **Engine Profile**: 1vCPU / 2 GiB Memory  

If you created your own model (see above)
* Add Environment Variables:  
  * **Name**: CHURN_MODEL_NAME  
  * **Value**: 20191120T161757_ibm_linear  **your model name from above**  
  Click "Add" and "Deploy Model"  
  
  

After the Application deploys, click on the blue-arrow next to the name.  The initial view is a table of rows selected at random from the dataset.  This shows a global view of which features are most important for the predictor model.


Clicking on any single row will show a "local" interpretabilty of a particular instance.  Here you 
can see how adjusting any one of the features will change the instance's churn prediction.


** Don't forget** to stop your Models and Experiments once you are done to save resources for your colleagues.


## Additional options
By default this code trains a linear regression model against the IBM dataset.
There are other datasets and other model types as well.  Set the Project environment variables to try other 
datasets and models:  
Name              Value  
CHURN_DATASET     ibm (default) | breastcancer | iris  
CHURN_MODEL_TYPE  linear (default) | gb | nonlinear | voting  


**NOTE** that not all of these options have been fully tested so your mileage may vary.

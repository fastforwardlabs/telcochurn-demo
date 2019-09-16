# Refractor (or churnexplainer)

This is the CDSW port of the Refractor prototype the is part of the [Interpretability
report from Cloudera Fast Forward Labs](https://clients.fastforwardlabs.com/ff06/report).

## OCRC Cluster: Train and inspect a new model locally

This processes uses the Experiments interface to instantiate an UI frontend for the Interpretability.  
This isn't necessarily what the Experiments were meant for but it works quite well.   
The Jobs interface works as well but has less options for passing parameters at runtime.  

### Set the Engine
If using this project on the OCRC cluster you can use the *mgregory/flask-lime:latest* engine which has the 
pre-requisite packages already installed.  This saves time in the demo.


In the Project -> Settings -> Engine tab select "Base Image V6 with sudo, flaskand lime for interpretability, mgregory/flask-lime:latest" 

### Train a predictor model
The default settings will create a linear regression model against the IBM dataset.


Go to the **Jobs** section and run the "Train Model" job.  
In the History of that Job you can see the name of the predictor model that was trained.
(ie. 20181128T220521_ibm_linear_c74c820).  You should also see this model in the "models" directory of the filesystem.
Note (copy) this model name to be used in the next section.

### Create an Explainer model for this predictor
Go to the **Models** section and create a new model with the following:
* **Name**: Explainer
* **Description**: Explainer
* **File**: explainer.py
* **Function**: explain
* **Input**: `{"StreamingTV":"No","MonthlyCharges":70.35,"PhoneService":"No","PaperlessBilling":"No","Partner":"No","OnlineBackup":"No","gender":"Female","Contract":"Month-to-month","TotalCharges":1397.475,"StreamingMovies":"No","DeviceProtection":"No","PaymentMethod":"Bank transfer (automatic)","tenure":29,"Dependents":"No","OnlineSecurity":"No","MultipleLines":"No","InternetService":"DSL","SeniorCitizen":"No","TechSupport":"No"}`
* **Kernel**: Python 3
* Click on "Set Environment Variables" and add:
  * **Name**: CHURN_MODEL_NAME
  * **Value**: 20181128T220521_ibm_linear_c74c820  **your model name from above**
  Click "Add" and "Deploy Model"


In the deployed Explainer model -> Settings note (copy) the "Access Key" (ie. mukd9sit7tacnfq2phhn3whc4unq1f38)


### Instatiate the flask UI application
From the Project level click on "Open Workbench" (note you don't actually have to Launch a session) in order to edit a file.
Select the flask/single_view.html file and paste the Access Key in at line 19. 
Save and go back to the Project.


Go to the **Experiments** section and select "Run Experiment" with the following:
* **Script**: flask_app_exp.py
* **Arguments**: num_rows model_name
  * num_rows is the number of rows to display in the global explainability table (ie. 20)
  * model_name is the name of the model that was built above or a different model (ie. 20181128T220521_ibm_linear_c74c820)
* **Kernel**: Python 3


In the main Experiments section the "APP_URL" field will have the link to the Flask App.  The initial
 view is a table of qty num_rows selected at random from the dataset.  This shows a global view of
 which features are most important for the predictor model.


Clicking on any single row will show a "local" interpretabilty of a particular instance.  Here you 
can see how adjusting any one of the features will change the instance's churn prediction.


** Don't forget** to stop your Models and Experiments once you are done to save resources for your colleagues.


## Other clusters: Dependencies
To build a local model on non-OCRC clusters you need to first build the dependencies in 
the `requirements.txt` file.

In a workbench session or terminal window run:

`pip3 install -r requirements.txt`

Or you can run the "Install Pre-reqs" Job.

## Interactive Flask Session

As an alternative to the Experiments feature you can launch the web app in an interactive Session. At the project 
level, you will need to set the CHURN_MODEL_NAME environment variable to the name of your predictor model.  
Open and run the `flask_app.py` file. This will start a long running flask process that will display some data 
from the churn explainer model in table form. There is a link to the table view page at the end of the `flask_app.py`
output called: _Open Table View_

Click on "Interrupt" at the top when you are done with the app.

## Additional options
By default this code trains a linear regression model against the IBM dataset.
There are other datasets and other model types as well.  Set the Project environment variables to try other 
datasets and models:
Name              Value
CHURN_DATASET     ibm (default) | breastcancer | iris 
CHURN_MODEL_TYPE  linear (default) | gb | nonlinear | voting

**NOTE** that not all of these options have been fully tested so your mileage may vary.


## Todo and Next Steps
* Decouple predictor from explainer to allow generic re-use.
* Add interface to allow human-entry of unique identifier instead of picking a single instance.
* Find a way to not require hard-coded API key in single_view.html
# This code will build a new model into the ~/models directory
# To use this new model, change and save the CHURN_MODEL_NAME environment 
# variable

from churnexplainer import train
train.train_and_explain_and_save()
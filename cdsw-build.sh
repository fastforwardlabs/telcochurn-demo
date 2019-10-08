#!/bin/bash

pip3 install -r requirements.txt

if [ ! -d "models" ] 
then
  mkdir models
fi

CHURN_MODEL_FILE="$CHURN_MODEL_NAME.pkl"

if [ -f $CHURN_MODEL_FILE ]
then 
  mkdir models/$CHURN_MODEL_NAME
  mv $CHURN_MODEL_FILE models/$CHURN_MODEL_NAME
fi

#echo "models/$CHURN_MODEL_NAME/$CHURN_MODEL_FILE"
#ls echo models/$CHURN_MODEL_NAME/$CHURN_MODEL_FILE
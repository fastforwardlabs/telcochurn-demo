import os
import importlib

dataset = os.environ.get('CHURN_DATASET', 'ibm')
load_dataset = (importlib
                .import_module('churnexplainer.data.' + dataset)
                .load_dataset)

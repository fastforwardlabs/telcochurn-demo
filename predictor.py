import os
import sys
from collections import ChainMap
from pandas.io.json import dumps as jsonify

sys.path.append("/home/cdsw") 

from churnexplainer.utils import log_environment
from churnexplainer.explainedmodel import ExplainedModel



em = ExplainedModel(os.environ['CHURN_MODEL_NAME'])

def predict(args):
    #data = dict(ChainMap(request.args, em.default_data))
    data = dict(ChainMap(args, em.default_data))
    data = em.cast_dct(data)
    probability, explanation = em.explain_dct(data)
    return jsonify({'probability': probability})
  

from flask import Flask,send_from_directory,request
import logging
from pandas.io.json import dumps as jsonify
import os
import random
from IPython.display import Javascript,HTML
import cdsw
from collections import ChainMap
from churnexplainer.utils import log_environment
from churnexplainer.explainedmodel import ExplainedModel
from urllib.parse import urljoin

# This reduces the the output to the 
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

#SAMPLE_SIZE = 25
SAMPLE_SIZE = int(sys.argv[1])
CHURN_MODEL_NAME = str(sys.argv[2])

em = ExplainedModel(CHURN_MODEL_NAME)

app= Flask(__name__,static_url_path='')
@app.route('/')
def home():
    return "<script> window.location.href = '/flask/table_view.html'</script>"


@app.route('/flask/<path:path>')
def send_file(path):
    return send_from_directory('flask', path)

def explainid_non_flask(N):
    customer_data = dataid_non_flask(N)[0]
    customer_data.pop('id')
    customer_data.pop('Churn probability')
    data = em.cast_dct(customer_data)
    probability, explanation = em.explain_dct(data)
    return {'data': dict(data),
                    'probability': probability,
                    'explanation': explanation,
           'id':int(N)}  

def dataid_non_flask(N):
    customer_id = em.data.index.dtype.type(N)
    customer_df = em.data.loc[[customer_id]].reset_index()
    return customer_df.to_dict(orient='records')

@app.route('/sample_table')
def sample_table():
  #N = request.args.get('N', 10, int)
  sample_ids = random.sample(range(1,len(em.data)),SAMPLE_SIZE)
  sample_table = []
  for ids in sample_ids:
    sample_table.append(explainid_non_flask(str(ids)))
  return jsonify(sample_table)

@app.route("/explain")
def explain():
    data = dict(ChainMap(request.args, em.default_data))
    data = em.cast_dct(data)
    probability, explanation = em.explain_dct(data)
    return jsonify({'data': dict(data),
                    'probability': probability,
                    'explanation': explanation})
  
@app.route('/explainid')
def explainid():
    customer_data = dataid(request.args['id'])[0]
    customer_data.pop('id')
    customer_data.pop('Churn probability')
    data = em.cast_dct(customer_data)
    probability, explanation = em.explain_dct(data)
    return {'data': dict(data),
                    'probability': probability,
                    'explanation': explanation}
  
@app.route("/data")
def data():
    N = request.args.get('N', 10, int)
    data = em.data.sample(N)
    return data.reset_index().to_json(orient='records')

  
@app.route("/modelname")
def modelname():
    return jsonify({'modelname': em.model_name})


@app.route("/dataset")
def dataset():
    return jsonify({'dataset': em.dataset})


@app.route("/dataid")
def dataid():
    customer_id = em.data.index.dtype.type(request.args['id'])
    customer_df = em.data.loc[[customer_id]].reset_index()
    return customer_df.to_json(orient='records')


@app.route("/features")
def features():
    response = {
        'id': em.data.index.name or 'index',
        'label': em.label_name,
        'features': list(em.default_data.keys())
    }
    return jsonify(response)


@app.route("/categories")
def categories():
    return jsonify({feat: dict(enumerate(cats))
                   for feat, cats in em.categories.items()})

@app.route("/stats")
def stats():
    return jsonify(em.stats)


@app.route("/size")
def size():
    return jsonify({'size': len(em.data)})


@app.route("/default")
def default():
    return jsonify(em.default_data)


HTML("<a href='https://{}.{}'>Open Table View</a>".format(os.environ['CDSW_ENGINE_ID'],os.environ['CDSW_DOMAIN']))

APP_URL = urljoin('https://'+os.environ['CDSW_ENGINE_ID']+'.'+os.environ['CDSW_DOMAIN'],'')
cdsw.track_metric("APP_URL", APP_URL)

if __name__=="__main__":
    app.run(host=os.environ['CDSW_IP_ADDRESS'], port=os.environ['CDSW_PUBLIC_PORT'])
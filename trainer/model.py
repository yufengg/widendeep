from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
#from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
#from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils


tf.logging.set_verbosity(tf.logging.ERROR)

CONTINUOUS_COLUMNS =  ["I"+str(i) for i in range(1,14)] # 1-13 inclusive
CATEGORICAL_COLUMNS = ["C"+str(i) for i in range(1,27)] # 1-26 inclusive
LABEL_COLUMN = ["clicked"]

TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
# TEST_DATA_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

def generate_input_fn(filename):
  def _input_fn():
    BATCH_SIZE = 40
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TextLineReader()
    key, value = reader.read_up_to(filename_queue, num_records=BATCH_SIZE)

    # 1 int label, 13 ints, 26 strings
    cont_defaults = [ [0] for i in range(1,14) ]
    cate_defaults = [ [0] for i in range(1,27) ]
    label_defaults = [ [0] ]
    column_headers = TRAIN_DATA_COLUMNS
    record_defaults = [ label_defaults + cont_defaults + cate_defaults ]

    columns = tf.decode_csv(
        value, record_defaults=record_defaults, field_delim='\t')

    features = dict(zip(column_headers, columns))

    # save our label
    labels = features.pop(LABEL_COLUMN)
    
    for feature_name in CATEGORICAL_COLUMNS:
      features[feature_name] = tf.expand_dims(features[feature_name], -1)

    return features, labels

  return _input_fn


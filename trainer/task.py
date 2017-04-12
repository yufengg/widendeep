# Copyright 2016 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils


tf.logging.set_verbosity(tf.logging.INFO) # Set to INFO for tracking training, default is WARN. ERROR for least messages

print("Using TensorFlow version %s" % (tf.__version__))

CONTINUOUS_COLUMNS =  ["I"+str(i) for i in range(1,14)] # 1-13 inclusive
CATEGORICAL_COLUMNS = ["C"+str(i) for i in range(1,27)] # 1-26 inclusive
LABEL_COLUMN = ["clicked"]

TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

BATCH_SIZE = 40

def generate_input_fn(filename, batch_size=BATCH_SIZE):
  def _input_fn():
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TextLineReader()
    # Reads out batch_size number of lines
    key, value = reader.read_up_to(filename_queue, num_records=batch_size)
    
    # 1 int label, 13 ints, 26 strings
    cont_defaults = [ [0] for i in range(1,14) ]
    cate_defaults = [ [" "] for i in range(1,27) ]
    label_defaults = [ [0] ]
    column_headers = TRAIN_DATA_COLUMNS
    record_defaults = label_defaults + cont_defaults + cate_defaults

    # Decode CSV data that was just read out. 
    columns = tf.decode_csv(
        value, record_defaults=record_defaults)
    
    # features is a dictionary that maps from column names to tensors of the data.
    # income_bracket is the last column of the data. Note that this is NOT a dict.
    all_columns = dict(zip(column_headers, columns))
    
    # Save the label column
    # dict.pop() returns the popped array of label values
    labels = all_columns.pop(LABEL_COLUMN[0])
    
    # the remaining columns are our features
    features = all_columns

    # Sparse categorical features must be represented with an additional dimension. 
    # There is no additional work needed for the Continuous columns; they are the unaltered columns.
    # See docs for tf.SparseTensor for more info
    for feature_name in CATEGORICAL_COLUMNS:
      features[feature_name] = tf.expand_dims(features[feature_name], -1)

    return features, labels

  return _input_fn


def build_feature_cols():
  # Sparse base columns.
  wide_columns = []
  for name in CATEGORICAL_COLUMNS:
      wide_columns.append(tf.contrib.layers.sparse_column_with_hash_bucket(name, hash_bucket_size=1000))

  # Continuous base columns.
  deep_columns = []
  for name in CONTINUOUS_COLUMNS:
    deep_columns.append(tf.contrib.layers.real_valued_column(name))

  # No transformations.
  # Embed wide columns into deep columns
  for col in wide_columns:
    deep_columns.append(tf.contrib.layers.embedding_column(col, dimension=8))

  return wide_columns, deep_columns

def build_model(model_type, model_dir, wide_columns, deep_columns):
  runconfig = tf.contrib.learn.RunConfig(
    save_checkpoints_secs=120,
    save_checkpoints_steps = None
  )
  m = None
  # Linear Classifier
  if model_type == 'WIDE':
    m = tf.contrib.learn.LinearClassifier(
      config=runconfig,
      model_dir=model_dir, 
      feature_columns=wide_columns)

  # Deep Neural Net Classifier
  elif model_type == 'DEEP':
    m = tf.contrib.learn.DNNClassifier(
      config=runconfig,
      model_dir=model_dir,
      feature_columns=deep_columns,
      hidden_units=[100, 70, 50, 25])

  # Combined Linear and Deep Classifier
  elif model_type == 'WIDE_AND_DEEP':
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
      config=runconfig,
      model_dir=model_dir,
      linear_feature_columns=wide_columns,
      dnn_feature_columns=deep_columns,
      dnn_hidden_units=[100, 70, 50, 25])

  return m


def build_estimator(model_type='WIDE_AND_DEEP', model_dir=None):
  if model_dir is None:
    model_dir = 'models/model_' + model_type + '_' + str(int(time.time()))
    print("Model directory = %s" % model_dir)

  wide_columns, deep_columns = build_feature_cols()
  m = build_model(model_type, model_dir, wide_columns, deep_columns)
  print('estimator built')
  return m


# All categorical columns are strings for this dataset
def column_to_dtype(column):
    if column in CATEGORICAL_COLUMNS:
        return tf.string
    else:
        return tf.float32

"""
  This function maps input columns (feature_placeholders) to 
  tensors that can be inputted into the graph 
  (similar in purpose to the output of our input functions)
  In this particular case, we need to accomodate the sparse fields (strings)
  so we have to do a slight modification to expand their dimensions, 
  just like in the input functions
"""
def serving_input_fn():
    feature_placeholders = {
        column: tf.placeholder(column_to_dtype(column), [None])
        for column in FEATURE_COLUMNS
    }
    # DNNCombinedLinearClassifier expects rank 2 Tensors, 
    # but inputs should be rank 1, so that we can provide 
    # scalars to the server
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    
    return input_fn_utils.InputFnOps(
        features, # input into graph
        None,
        feature_placeholders # tensor input converted from request 
    )
    
def generate_experiment(output_dir, train_file, test_file, model_type):
  def _experiment_fn(output_dir):
    train_input_fn = generate_input_fn(train_file)
    eval_input_fn = generate_input_fn(test_file)
    my_model = build_estimator(model_type=model_type, 
                               model_dir=output_dir)

    experiment = tf.contrib.learn.Experiment(
      my_model,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      train_steps=1000
      ,
      export_strategies=[saved_model_export_utils.make_export_strategy(
        serving_input_fn,
        default_output_alternative_key=None
      )]
    )
    return experiment

  return _experiment_fn


def train_and_eval(job_dir=None):
  print("Begin training and evaluation")

  # if local eval and no args passed, default
  if job_dir is None: job_dir = 'models/' 

  # Ensure path has a '/' at the end
  if job_dir[-1] != '/': job_dir += '/'

  # dataset-uploader/criteo-kaggle/small_version
  gcs_base = 'https://storage.googleapis.com/' # No need to change
  gcs_path = 'dataset-uploader/criteo-kaggle/medium_version/' # Path to the folder with the files
  trainfile = 'train.csv'
  testfile  = 'eval.csv'
  local_path = 'dataset_files'
  train_file = base.maybe_download(
    trainfile, local_path, gcs_base + gcs_path + trainfile)
  test_file = base.maybe_download(
    testfile, local_path, gcs_base + gcs_path + testfile)

  training_mode = 'learn_runner'
  train_steps = 1000
  test_steps = 100
  model_type = 'DEEP'

  model_dir = job_dir + 'model_' + model_type + '_' + str(int(time.time()))
  print("Saving model checkpoints to " + model_dir)
  export_dir = model_dir + '/exports'

  # Manually train and export model
  if training_mode == 'manual':
    # In this function, editing below here is unlikely to be needed
    m = build_estimator(model_type, model_dir)

    m.fit(input_fn=generate_input_fn(train_file), steps=train_steps)
    print('fit done')

    results = m.evaluate(input_fn=generate_input_fn(test_file), steps=test_steps)
    print('evaluate done')

    print('Accuracy: %s' % results['accuracy'])

    export_folder = m.export_savedmodel(
      export_dir_base = export_dir,
      input_fn=serving_input_fn
    )

    print('Model exported to ' + export_dir)

  elif training_mode == 'learn_runner':
    # use learn_runner
    experiment_fn = generate_experiment(
      model_dir, train_file, test_file, model_type)

    metrics, output_folder = learn_runner.run(experiment_fn, model_dir)

    print('Accuracy: {}'.format(metrics['accuracy']))
    print('Model exported to {}'.format(output_folder))


def version_is_less_than(a, b):
    a_parts = a.split('.')
    b_parts = b.split('.')
    
    for i in range(len(a_parts)):
        if int(a_parts[i]) < int(b_parts[i]):
            print('{} < {}, version_is_less_than() returning False'.format(
              a_parts[i], b_parts[i]))
            return True
    return False

def get_arg_parser():
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=False
  )

  return parser

if __name__ == "__main__":
  print("TensorFlow version {}".format(tf.__version__))
  required_tf_version = '1.0.0'
  if version_is_less_than(tf.__version__ , required_tf_version):
    raise ValueError('This code requires tensorflow >= ' + str(required_tf_version))

  parser = get_arg_parser()
  args = parser.parse_args()
  train_and_eval(args.job_dir)

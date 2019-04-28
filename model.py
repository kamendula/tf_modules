#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import iris_data
import glob
import random
from datetime import date, timedelta
from DeepFM import DeepFM
from BaseModel import BaseModel
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def build_model_fn(features, labels, mode, params):
    model = DeepFM(features, labels, mode, params).build_model()
    return model


def main(argv):
    args = parser.parse_args(argv[1:])
    params = load_base_params("./base_conf")

    print (params["dt_dir"])
    if params["dt_dir"] == "":
        params["dt_dir"] = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    params["model_dir"] = params["model_dir"] + "_" + params["dt_dir"]

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()
    params["model_type"] = "DNN"

    #------init Envs------
    print ("data dir is ", params["data_dir"])
    tr_files = glob.glob("%s/tr*libsvm" % params["data_dir"])
    random.shuffle(tr_files)
    print("tr_files:", tr_files)
    va_files = glob.glob("%s/va*libsvm" % params["data_dir"])
    print("va_files:", va_files)
    te_files = glob.glob("%s/te*libsvm" % params["data_dir"])
    print("te_files:", te_files)

    config = tf.estimator.RunConfig().replace( \
            session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':int(params["num_threads"])}),
            log_step_count_steps=int(params["log_steps"]), save_summary_steps=int(params["log_steps"]))
    params["estimator_config"] = config
    params["filename"] = {}
    params["filename"]["train"] = tr_files
    params["filename"]["eval"] = va_files
    params["filename"]["test"] = te_files
    #DeepFM = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=params, config=config)
    #deepFm_model = build_model_fn(features, labels, params)
    classifier = tf.estimator.Estimator(model_fn=build_model_fn, model_dir=params["model_dir"], params=params, config=config)

    # Train the Model.
    classifier.train(
        input_fn=lambda:get_input_fn_2(params), 
    steps=args.train_steps)
    sys.exit()

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:get_input_fn("test"))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

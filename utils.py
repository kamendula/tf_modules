#encoding:utf-8

import sys
import iris_data
import tensorflow as tf
import configparser

def get_input_fn(mode = "train", batch_size=5):
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()
    """An input function for training"""
    # Convert the inputs to a Dataset.
    if mode == "train":
        dataset = tf.data.Dataset.from_tensor_slices((dict(train_x), train_y))
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((dict(test_x), test_y))
        dataset = dataset.batch(batch_size)


    # Return the dataset.
    return dataset

def get_input_fn_2(params):
    if params["task_type"]:
        filenames = params["filename"]["train"]
    #filenames = params["filename"]
    print('Parsing', filenames)
    batch_size = int(params["batch_size"])
    num_epochs = int(params["num_epochs"])
    perform_shuffle=False

    def decode_libsvm(line):
        #columns = tf.decode_csv(value, record_defaults=CSV_COLUMN_DEFAULTS)
        #features = dict(zip(CSV_COLUMNS, columns))
        #labels = features.pop(LABEL_COLUMN)
        columns = tf.string_split([line], ' ')
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        splits = tf.string_split(columns.values[1:], ':')
        id_vals = tf.reshape(splits.values,splits.dense_shape)
        feat_ids, feat_vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        #feat_ids = tf.reshape(feat_ids,shape=[-1,FLAGS.field_size])
        #for i in range(splits.dense_shape.eval()[0]):
        #    feat_ids.append(tf.string_to_number(splits.values[2*i], out_type=tf.int32))
        #    feat_vals.append(tf.string_to_number(splits.values[2*i+1]))
        #return tf.reshape(feat_ids,shape=[-1,field_size]), tf.reshape(feat_vals,shape=[-1,field_size]), labels
        return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(500000)    # multi-thread pre-process then prefetch

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) # Batch size to use

    #return dataset.make_one_shot_iterator()
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    #return tf.reshape(batch_ids,shape=[-1,field_size]), tf.reshape(batch_vals,shape=[-1,field_size]), batch_labels
    return batch_features, batch_labels
    


def get_feature_columns_list(feature_list):
    feature_columns = []
    for key in train_x.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))
    return feature_columns

def load_base_params(base_conf):
    config = configparser.ConfigParser()
    config.read("base.conf")
    params = {}
    raw_params=dict(config.items("default"))
    for k, v in raw_params.items():
        items = v.split(",")
        if len(items) == 0:
            params[k] = ""
        elif len(items) == 1:
            params[k] = items[0]
        else:
            if items[1] == "int":
                params[k] = int(items[0])
            elif items[1] == "float":
                params[k] = float(items[0])
            elif items[1] == "bool":
                params[k] = (items[0] == "True")
            else:
                params[k] = items[0]

    return params


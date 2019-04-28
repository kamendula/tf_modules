#encoding:utf-8

import sys
import iris_data
import tensorflow as tf
import configparser
from utils import *

class BaseModel(object):
    def __init__(self, params):
        #------hyperparameters----
        self.params = params
        self.field_size = params["field_size"]
        self.feature_size = params["feature_size"]
        self.embedding_size = params["embedding_size"]
        self.l2_reg = params["l2_reg"]
        self.learning_rate = params["learning_rate"]
        #batch_norm_decay = params["batch_norm_decay"]
        #optimizer = params["optimizer"]
        self.layers  = map(int, params["deep_layers"].split(':'))
        self.dropout = map(float, params["dropout"].split(':'))
        self.model_type = self.params["model_type"]
        self.batch_norm = self.params["batch_norm"]

    def build_model(self):
        if self.params["model_type"] == "DNN":
            feature_columns = self.build_columns()
            classifier = tf.estimator.DNNClassifier(
                feature_columns=feature_columns,
                # Two hidden layers of 10 nodes each.
                hidden_units=[10, 10],
                # The model must choose between 3 classes.
                n_classes=3)
        return classifier 

    def build_columns(self):
        feature_columns = []
        for key in self.features:
            feature_columns.append(tf.feature_column.numeric_column(key=key))
        return feature_columns

class DeepFM(BaseModel):
    def __init__(self, features, labels, mode, params):
        super(DeepFM, self).__init__(params)
        self.features = features
        self.labels = labels
        self.mode = mode
        self.weights = {}

    def build_features(self):
        feat_ids  = self.features['feat_ids']
        self.feat_ids = tf.reshape(feat_ids,shape=[-1,self.field_size])
        feat_vals = self.features['feat_vals']
        shape = feat_vals.get_shape()
        self.feat_vals = tf.reshape(feat_vals,shape=[-1,self.field_size])

    def build_loss(self, y):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=self.labels)) + \
            self.l2_reg * tf.nn.l2_loss(self.weights["FM_W"]) + \
                self.l2_reg * tf.nn.l2_loss(self.weights["FM_V"])
        return loss

    def init_weights(self):
        FM_B = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
        FM_W = tf.get_variable(name='fm_w', shape=[self.feature_size], initializer=tf.glorot_normal_initializer())
        FM_V = tf.get_variable(name='fm_v', shape=[self.feature_size, self.embedding_size], initializer=tf.glorot_normal_initializer())
        self.weights["FM_B"] = FM_B
        self.weights["FM_W"] = FM_W
        self.weights["FM_V"] = FM_V

    def build_model(self):
        model_fn = self._build_model()
        return model_fn

    def _build_model(self):
        self.init_weights()
        self.build_features()
        pred, y = self.build_graph()

        ####predict, end here.
        predictions={"prob": pred}
        export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
        # Provide an estimator spec for `ModeKeys.PREDICT`
        if self.mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                    mode=self.mode,
                    predictions=predictions,
                    export_outputs=export_outputs)

        loss = self.build_loss(y)

        # Provide an estimator spec for `ModeKeys.EVAL`
        ####eval, end here.
        eval_metric_ops = {
            "auc": tf.metrics.auc(self.labels, pred)
        }
        if self.mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                    mode=self.mode,
                    predictions=predictions,
                    loss=loss,
                    eval_metric_ops=eval_metric_ops)
        optimizer = self.build_optimizer()
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        # Provide an estimator spec for `ModeKeys.TRAIN` modes
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(
                    mode=self.mode,
                    predictions=predictions,
                    loss=loss,
                    train_op=train_op)

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = tf.contrib.layers.batch_norm(x, decay=self.params["batch_norm_decay"], center=True, scale=True, updates_collections=None, is_training=True,  reuse=None, scope=scope_bn)
        bn_infer = tf.contrib.layers.batch_norm(x, decay=self.params["batch_norm_decay"], center=True, scale=True, updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
        z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
        return z


    def build_graph(self):
        #------build f(x)------
        with tf.variable_scope("First-order"):
            feat_wgts = tf.nn.embedding_lookup(self.weights["FM_W"], self.feat_ids)              # None * F * 1
            y_w = tf.reduce_sum(tf.multiply(feat_wgts, self.feat_vals),1)

        with tf.variable_scope("Second-order"):
            embeddings = tf.nn.embedding_lookup(self.weights["FM_V"], self.feat_ids)             # None * F * K
            feat_vals = tf.reshape(self.feat_vals, shape=[-1, self.field_size, 1])
            embeddings = tf.multiply(embeddings, feat_vals)                 #vij*xi
            sum_square = tf.square(tf.reduce_sum(embeddings,1))
            square_sum = tf.reduce_sum(tf.square(embeddings),1)
            y_v = 0.5*tf.reduce_sum(tf.subtract(sum_square, square_sum),1)	# None * 1

        with tf.variable_scope("Deep-part"):
            if  self.params["batch_norm"]:
                #normalizer_fn = tf.contrib.layers.batch_norm
                #normalizer_fn = tf.layers.batch_normalization
                if self.mode == tf.estimator.ModeKeys.TRAIN:
                    train_phase = True
                    #normalizer_params = {'decay': batch_norm_decay, 'center': True, 'scale': True, 'updates_collections': None, 'is_training': True, 'reuse': None}
                else:
                    train_phase = False
                    #normalizer_params = {'decay': batch_norm_decay, 'center': True, 'scale': True, 'updates_collections': None, 'is_training': False, 'reuse': True}
            else:
                normalizer_fn = None
                normalizer_params = None

            deep_inputs = tf.reshape(embeddings,shape=[-1,self.field_size*self.embedding_size]) # None * (F*K)
            for i in range(len(self.layers)):
                #if FLAGS.batch_norm:
                #    deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' %i)
                    #normalizer_params.update({'scope': 'bn_%d' %i})
                deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=self.layers[i], \
                    #normalizer_fn=normalizer_fn, normalizer_params=normalizer_params, \
                    weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg), scope='mlp%d' % i)
                if self.batch_norm:
                    deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' %i)   #放在RELU之后 https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu
                if self.mode == tf.estimator.ModeKeys.TRAIN:
                    deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=self.dropout[i])                              #Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)
                    #deep_inputs = tf.layers.dropout(inputs=deep_inputs, rate=dropout[i], training=mode == tf.estimator.ModeKeys.TRAIN)

            y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity, \
                    weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg), scope='deep_out')
            y_d = tf.reshape(y_deep,shape=[-1])
            #sig_wgts = tf.get_variable(name='sigmoid_weights', shape=[layers[-1]], initializer=tf.glorot_normal_initializer())
            #sig_bias = tf.get_variable(name='sigmoid_bias', shape=[1], initializer=tf.constant_initializer(0.0))
            #deep_out = tf.nn.xw_plus_b(deep_inputs,sig_wgts,sig_bias,name='deep_out')

        with tf.variable_scope("DeepFM-out"):
            #y_bias = FM_B * tf.ones_like(labels, dtype=tf.float32)  # None * 1  warning;这里不能用label，否则调用predict/export函数会出错，train/evaluate正常；初步判断estimator做了优化，用不到label时不传
            y_bias = self.weights["FM_B"] * tf.ones_like(y_d, dtype=tf.float32)      # None * 1
            y = y_bias + y_w + y_v + y_d
            pred = tf.sigmoid(y)
            return pred, y

    def build_optimizer(self):
        if self.params["optimizer"] == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        if self.params["optimizer"] == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8)
        if self.params["optimizer"] == 'Momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95)
        if self.params["optimizer"] == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(self.learning_rate)
        return optimizer

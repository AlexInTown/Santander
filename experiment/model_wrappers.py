# -*- coding: utf-8 -*-
__author__ = 'AlexInTown'

import numpy as np
import time
import copy


class XgboostModel:
    def __init__(self, model_params, train_params=None, test_params=None):
        self.model_params = model_params
        if train_params:
            self.train_params = train_params
        else:
            self.train_params = {"num_boost_round": 300 }
        self.test_params = test_params
        fname_parts = ['xgb']
        fname_parts.extend(['{0}#{1}'.format(key, val) for key,val in model_params.iteritems()])
        self.model_out_fname = '-'.join(fname_parts)

    def fit(self, X, y):
        """Fit model."""
        import xgboost as xgb
        dtrain = xgb.DMatrix(X, label=np.asarray(y))
        #bst, loss, ntree = xgb.train(self.model_params, dtrain, num_boost_round=self.train_params['num_boost_round'])
        self.bst = xgb.train(self.model_params, dtrain, num_boost_round=self.train_params['num_boost_round'])
        #self.loss = loss
        #self.ntree = ntree
        #print loss, ntree

    def predict(self, X):
        """Predict using the xgb model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        import xgboost as xgb
        dtest = xgb.DMatrix(X)
        return self.bst.predict(dtest)

    def to_string(self):
        return self.model_out_fname


class SklearnModel:
    def __init__(self, model_params):
        self.model_params = copy.deepcopy(model_params)
        self.model_class = model_params['model_type']
        del self.model_params['model_type']
        fname_parts = [self.model_class.__name__]
        fname_parts.extend(['{0}#{1}'.format(k,v) for k,v in model_params.iteritems()])
        self.model = self.model_class(**self.model_params)
        self.model_out_fname = '-'.join(fname_parts)

    def fit(self, X, y):
        """Fit model."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predict using the sklearn model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        return self.model.predict_proba(X)[:, 1]

    def to_string(self):
        return self.model_out_fname


class LasagneModel:

    def __init__(self, model_params):
        import lasagne
        import theano.tensor as T
        self.model_params = model_params
        self.batch_size = model_params['batch_size']
        self.var_input = T.matrix('var_input')
        self.var_targets = T.ivector('var_targets')
        l_in = lasagne.layers.InputLayer(shape=(None, model_params['in_size']), input_var=self.var_input)

        # Apply 20% dropout to the input data:
        l_in_drop = lasagne.layers.DropoutLayer(l_in, p=model_params['in_dropout'])

        # Add a fully-connected layer of 800 units, using the linear rectifier, and
        # initializing weights with Glorot's scheme (which is the default anyway):
        l_hid1 = lasagne.layers.DenseLayer(
                l_in_drop, num_units=model_params['hid_size'],
                nonlinearity=model_params['nonlinearity'],
                W=lasagne.init.GlorotUniform())

        # We'll now add dropout of 50%:
        l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=model_params['hid_dropout'])

        # Another 800-unit layer:
        l_hid2 = lasagne.layers.DenseLayer(
                l_hid1_drop, num_units=model_params['hid_size'],
                nonlinearity=model_params['nonlinearity'])

        # 50% dropout again:
        l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=model_params['hid_dropout'])

        # Finally, we'll add the fully-connected output layer, of 10 softmax units:
        l_out = lasagne.layers.DenseLayer(
                l_hid2_drop, num_units=2,
                nonlinearity=lasagne.nonlinearities.softmax)

        # Each layer is linked to its incoming layer(s), so we only need to pass
        # the output layer to give access to a network in Lasagne:
        self.network = l_out

    @classmethod
    def iterate_minibatches(cls, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def fit(self, X, y):
        """Fit model."""
        X = np.asarray(X)
        y = np.asarray(y)
        import lasagne
        import theano
        import theano.tensor as T
        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(self.network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, self.var_targets)
        loss = loss.mean()

        # TODO We could add some weight decay as well here, see lasagne.regularization.

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params(self.network, trainable=1)
        updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate=self.model_params['learning_rate'], momentum=0.9)

        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        self.test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(self.test_prediction, self.var_targets)
        test_loss = test_loss.mean()
        # As a bonus, also create an expression for the classification accuracy:
        #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.vector),
        #                  dtype=theano.config.floatX)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        self.train_fn = theano.function([self.var_input,self.var_targets], loss, updates=updates, allow_input_downcast=1)

        # Compile a second function computing the validation loss:
        self.val_fn = theano.function([self.var_input, self.var_targets], test_loss)

        # Finally, launch the training loop.
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(self.model_params['num_epochs']):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(X, y, self.batch_size, shuffle=1):
                inputs, targets = batch
                train_err += self.train_fn(inputs, targets)
                train_batches += 1
            # Then we print the results for this epoch:
            # print("Epoch {} of {} took {:.3f}s".format(
            #         epoch + 1, self.model_params['num_epochs'], time.time() - start_time))
            # print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            # And a full pass over the validation data:
            X_val = None
            y_val = None
            if 'x_val' in self.model_params:
                X_val = self.model_params['x_val']
            if 'y_val' in self.model_params:
                y_val = self.model_params['y_val']
            if X_val and y_val:
                val_err = 0
                val_acc = 0
                val_batches = 0
                for batch in self.iterate_minibatches(X_val, y_val, self.batch_size, shuffle=1):
                    inputs, targets = batch
                    err, acc = self.val_fn(inputs, targets)
                    val_err += err
                    val_acc += acc
                    val_batches += 1
                # print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                # print("  validation accuracy:\t\t{:.2f} %".format(
                #     val_acc / val_batches * 100))
                pass
        pass

    def predict(self, X):
        """Predict using the sklearn model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        X = np.asarray(X)
        import theano
        import theano.tensor as T
        test_fn = theano.function([self.var_input], self.test_prediction, allow_input_downcast=1)
        return test_fn(X)[:, 1]


    def to_string(self):
        return self.model_out_fname


from __future__ import print_function, absolute_import

import lasagne
import numpy
import time
import json
import h5py

import theano
import theano.tensor as T

from lifelines.utils import concordance_index

from deepsurv.deepsurv_logger import DeepSurvLogger

from lasagne.regularization import regularize_layer_params, l1, l2
from lasagne.nonlinearities import rectify,selu

class DeepSurv:
    def __init__(self, n_in,
    learning_rate, hidden_layers_sizes = None,
    lr_decay = 0.0, momentum = 0.9,
    L2_reg = 0.0, L1_reg = 0.0,
    activation = "rectify",
    dropout = None,
    batch_norm = False,
    standardize = False,
    ):

        self.X = T.fmatrix('x')  # patients covariates
        self.E = T.ivector('e') # the observations vector

        # Default Standardization Values: mean = 0, std = 1
        self.offset = numpy.zeros(shape = n_in, dtype=numpy.float32)
        self.scale = numpy.ones(shape = n_in, dtype=numpy.float32)

        # self.offset = theano.shared(numpy.zeros(shape = n_in, dtype=numpy.float32))
        # self.scale = theano.shared(numpy.ones(shape = n_in, dtype=numpy.float32))

        network = lasagne.layers.InputLayer(shape=(None,n_in), input_var = self.X)
        network_2 = lasagne.layers.InputLayer(shape=(None,n_in), input_var = self.X)

        # if standardize:
        #     network = lasagne.layers.standardize(network,self.offset,
        #                                         self.scale,
        #                                         shared_axes = 0)
        self.standardize = standardize

        if activation == 'rectify':
            activation_fn = rectify
        elif activation == 'selu':
            activation_fn = selu
        else:
            raise IllegalArgumentException("Unknown activation function: %s" % activation)

        # Construct Neural Network
        for n_layer in (hidden_layers_sizes or []):
            if activation_fn == lasagne.nonlinearities.rectify:
                W_init = lasagne.init.GlorotUniform()
            else:
                # TODO: implement other initializations
                W_init = lasagne.init.GlorotUniform()

            network = lasagne.layers.DenseLayer(
                network, num_units = n_layer,
                nonlinearity = activation_fn,
                W = W_init
            )
            
            network_2 = lasagne.layers.DenseLayer(
                network_2, num_units = n_layer,
                nonlinearity = activation_fn,
                W = W_init
            )

            if batch_norm:
                network = lasagne.layers.batch_norm(network)
                network_2 = lasagne.layers.batch_norm(network_2)

            if not dropout is None:
                network = lasagne.layers.DropoutLayer(network, p = dropout)
                network_2 = lasagne.layers.DropoutLayer(network_2, p = dropout)

        # Combine Linear to output Log Hazard Ratio - same as Faraggi
        network = lasagne.layers.DenseLayer(
            network, num_units = 1,
            nonlinearity = lasagne.nonlinearities.linear,
            W = lasagne.init.GlorotUniform()
        )
        
        network_2 = lasagne.layers.DenseLayer(
            network_2, num_units = 1,
            nonlinearity = lasagne.nonlinearities.linear,
            W = lasagne.init.GlorotUniform()
        )
        
        self.network_1 = network
        self.network_2 = network_2
        self.network_combine = lasagne.layers.ConcatLayer([network, network_2])
        
        self.params = lasagne.layers.get_all_params(self.network_combine, trainable = True)
        
        self.hidden_layers = lasagne.layers.get_all_layers(self.network_combine)[1:]
        
        # Relevant Functions
        self.partial_hazard = T.exp(self.risk(self.network_1, deterministic = True)) # e^h(x)
       

        # Store and set needed Hyper-parameters:
        self.hyperparams = {
            'n_in': n_in,
            'learning_rate': learning_rate,
            'hidden_layers_sizes': hidden_layers_sizes,
            'lr_decay': lr_decay,
            'momentum': momentum,
            'L2_reg': L2_reg,
            'L1_reg': L1_reg,
            'activation': activation,
            'dropout': dropout,
            'batch_norm': batch_norm,
            'standardize': standardize
        }

        self.n_in = n_in
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.L2_reg = L2_reg
        self.L1_reg = L1_reg
        self.momentum = momentum
        self.restored_update_params = None

    def _negative_log_likelihood(self, network, E, deterministic = False):
        
        risk = self.risk(network, deterministic)
        hazard_ratio = T.exp(risk)
        log_risk = T.log(T.extra_ops.cumsum(hazard_ratio))
        uncensored_likelihood = risk.T - log_risk
        censored_likelihood = uncensored_likelihood * E
        num_observed_events = T.sum(E)
        neg_likelihood = -T.sum(censored_likelihood) / num_observed_events
        return neg_likelihood

    def _get_loss_updates(self,
    L1_reg = 0.0, L2_reg = 0.001,
    update_fn = lasagne.updates.nesterov_momentum,
    max_norm = None, deterministic = False,
    momentum = 0.9,
    **kwargs):
        
        
        loss = (
            self._negative_log_likelihood(self.network_1, self.E, deterministic)
            + self._negative_log_likelihood(self.network_2, self.E, deterministic)
            + regularize_layer_params(self.network_1, l1) * L1_reg
            + regularize_layer_params(self.network_1, l2) * L2_reg
            + regularize_layer_params(self.network_2, l1) * L1_reg
            + regularize_layer_params(self.network_2, l2) * L2_reg
            + (regularize_layer_params(self.network_1, l2) - regularize_layer_params(self.network_2, l2)) * L2_reg
        )

        if max_norm:
            grads = T.grad(loss,self.params)
            scaled_grads = lasagne.updates.total_norm_constraint(grads, max_norm)
            updates = update_fn(
                scaled_grads, self.params, **kwargs
            )
        else:
            updates = update_fn(
                loss, self.params, **kwargs
            )

        if momentum:
            updates = lasagne.updates.apply_nesterov_momentum(updates, 
                self.params, self.learning_rate, momentum=momentum)

        # If the model was loaded from file, reload params
        if self.restored_update_params:
            for p, value in zip(updates.keys(), self.restored_update_params):
                p.set_value(value)
            self.restored_update_params = None

        # Store last update function to be later saved
        self.updates = updates

        return loss, updates

    def _get_train_valid_fn(self,
    L1_reg, L2_reg, learning_rate,
    **kwargs):

        loss, updates = self._get_loss_updates(
            L1_reg, L2_reg, deterministic = False,
            learning_rate=learning_rate, **kwargs
        )
        train_fn = theano.function(
            inputs = [self.X, self.E],
            outputs = loss,
            updates = updates,
            name = 'train'
        )

        valid_loss, _ = self._get_loss_updates(
            L1_reg, L2_reg, deterministic = True,
            learning_rate=learning_rate, **kwargs
        )

        valid_fn = theano.function(
            inputs = [self.X, self.E],
            outputs = valid_loss,
            name = 'valid'
        )
        return train_fn, valid_fn
    

    def get_concordance_index(self, x, t, e, **kwargs):
        
        compute_hazards = theano.function(
            inputs = [self.X],
            outputs = -self.partial_hazard
        )
        partial_hazards = compute_hazards(x)

        return concordance_index(t,
            partial_hazards,
            e)

    def _standardize_x(self, x):
        return (x - self.offset) / self.scale

    # @TODO: implement for varios instances of datasets
    def prepare_data(self,dataset):
        if isinstance(dataset, dict):
            x, e, t = dataset['x'], dataset['e'], dataset['t']

        if self.standardize:
            x = self._standardize_x(x)

        # Sort Training Data for Accurate Likelihood
        sort_idx = numpy.argsort(t)[::-1]
        x = x[sort_idx]
        e = e[sort_idx]
        t = t[sort_idx]

        return (x, e, t)

    def train(self,
    train_data, valid_data= None,
    n_epochs = 500,
    validation_frequency = 250,
    patience = 2000, improvement_threshold = 0.99999, patience_increase = 2,
    logger = None,
    update_fn = lasagne.updates.nesterov_momentum,
    verbose = True,
    **kwargs):
        
        if logger is None:
            logger = DeepSurvLogger('DeepSurv')

        # Set Standardization layer offset and scale to training data mean and std
        if self.standardize:
            self.offset = train_data['x'].mean(axis = 0)
            self.scale = train_data['x'].std(axis = 0)

        x_train, e_train, t_train = self.prepare_data(train_data)

        if valid_data:
            x_valid, e_valid, t_valid = self.prepare_data(valid_data)

        # Initialize Metrics
        best_validation_loss = numpy.inf
        best_params = None
        best_params_idx = -1

        # Initialize Training Parameters
        lr = theano.shared(numpy.array(self.learning_rate,
                                    dtype = numpy.float32))
        momentum = numpy.array(0, dtype= numpy.float32)

        train_fn, valid_fn = self._get_train_valid_fn(
            L1_reg=self.L1_reg, L2_reg=self.L2_reg,
            learning_rate=lr,
            momentum = momentum,
            update_fn = update_fn, **kwargs
        )

        start = time.time()
        for epoch in range(n_epochs):
            # Power-Learning Rate Decay
            lr = self.learning_rate / (1 + epoch * self.lr_decay)
            logger.logValue('lr', lr, epoch)

            if self.momentum and epoch >= 10:
                momentum = self.momentum

            loss = train_fn(x_train, e_train)

            logger.logValue('loss', loss, epoch)
            # train_loss.append(loss)

            ci_train = self.get_concordance_index(
                x_train,
                t_train,
                e_train,
            )
            logger.logValue('c-index',ci_train, epoch)
            # train_ci.append(ci_train)

            if valid_data and (epoch % validation_frequency == 0):
                validation_loss = valid_fn(x_valid, e_valid)
                logger.logValue('valid_loss',validation_loss, epoch)

                ci_valid = self.get_concordance_index(
                    x_valid,
                    t_valid,
                    e_valid
                )
                logger.logValue('valid_c-index', ci_valid, epoch)

                if validation_loss < best_validation_loss:
                    # improve patience if loss improves enough
                    if validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, epoch * patience_increase)

                    best_params = [param.copy().eval() for param in self.params]
                    best_params_idx = epoch
                    best_validation_loss = validation_loss

            if verbose and (epoch % validation_frequency == 0):
                logger.print_progress_bar(epoch, n_epochs, loss, ci_train)

            if patience <= epoch:
                break

        if verbose:
            logger.logMessage('Finished Training with %d iterations in %0.2fs' % (
                epoch + 1, time.time() - start
            ))
        logger.shutdown()

        # Return Logger.getMetrics()
        # metrics = {
        #     'train': train_loss,
        #     'best_params': best_params,
        #     'best_params_idx' : best_params_idx,
        #     'train_ci' : train_ci
        # }
        # if valid_data:
        #     metrics.update({
        #         'valid' : valid_loss,
        #         'valid_ci': valid_ci,
        #         'best_valid_ci': max(valid_ci),
        #         'best_validation_loss':best_validation_loss
        #     })
        logger.history['best_valid_loss'] = best_validation_loss
        logger.history['best_params'] = best_params
        logger.history['best_params_idx'] = best_params_idx

        return logger.history

    def to_json(self):
        return json.dumps(self.hyperparams)

    def save_model(self, filename, weights_file = None):
        with open(filename, 'w') as fp:
            fp.write(self.to_json())

        if weights_file:
            self.save_weights(weights_file)

    def save_weights(self,filename):
        def save_list_by_idx(group, lst):
            for (idx, param) in enumerate(lst):
                group.create_dataset(str(idx), data=param)

        weights_out = lasagne.layers.get_all_param_values(self.network_combine, trainable=False)
        if self.updates:
            updates_out = [p.get_value() for p in self.updates.keys()]
        else:
            raise Exception("Model has not been trained: no params to save!")

        # Store all of the parameters in an hd5f file
        # We store the parameter under the index in the list
        # so that when we read it later, we can construct the list of
        # parameters in the same order they were saved
        with h5py.File(filename, 'w') as f_out:
            weights_grp = f_out.create_group('weights')
            save_list_by_idx(weights_grp, weights_out)

            updates_grp = f_out.create_group('updates')
            save_list_by_idx(updates_grp, updates_out)

    def load_weights(self, filename):
        def load_all_keys(fp):
            results = []
            for key in fp:
                dataset = fp[key][:]
                results.append((int(key), dataset))
            return results

        def sort_params_by_idx(params):
            return [param for (idx, param) in sorted(params, 
            key=lambda param: param[0])]

        # Load all of the parameters
        with h5py.File(filename, 'r') as f_in:
            weights_in = load_all_keys(f_in['weights'])
            updates_in = load_all_keys(f_in['updates'])

        # Sort them according to the idx to ensure they are set correctly
        sorted_weights_in = sort_params_by_idx(weights_in)
        lasagne.layers.set_all_param_values(self.network_combine, sorted_weights_in, 
            trainable=False)

        sorted_updates_in = sort_params_by_idx(updates_in)
        self.restored_update_params = sorted_updates_in

    def risk(self, network, deterministic = False):
        
        return lasagne.layers.get_output(network,
                                        deterministic = deterministic)


import numpy as np
from sklearn.metrics import log_loss
from scikits.statsmodels.tools import categorical
from sklearn import preprocessing
import random
import theano
from theano import tensor as T
from .nnet_updates import Optimizers_update



class Regression(object):
    
    def __init__(self, X, Y, X_test, Y_test, iters, learning_rate = 0.01, optimizer = 'rmsprop', batch_size = None, L1 = 0.001, L2 = 0.001, maximize = False, early_stopping_rounds = 10,
                 
                 weights_initialization = 'uniform', objective = 'categorical_crossentropy', linear_regression = False, add_bias = True, custom_eval = None):
      
        
        self.X_dat = X
        self.y_dat = Y
        self.X_test = X_test
        self.Y_test = Y_test
        self.iters = iters
        self.learning_rate = learning_rate
        self.weights_initialization = weights_initialization
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.L1 = L1
        self.L2 = L2
        self.maximize = maximize
        self.early_stopping_rounds = early_stopping_rounds
        self.objective = objective
        self.linear_regression = linear_regression
        self.add_bias = add_bias
        self.custom_eval = custom_eval
        

    @staticmethod
    def initialize_weights(shape, inp, outp, weights_f):            
        
        if weights_f == 'uniform':
            
            init_w = theano.shared(np.asarray(np.random.uniform(low=-0.1, high=0.1, size = shape), dtype=theano.config.floatX))
            
        if weights_f == 'normal':
            
            init_w = theano.shared(np.asarray(np.random.normal(loc=0.0, scale=0.1, size = shape), dtype=theano.config.floatX))
            
        if weights_f == 'glorot_uniform':
            
            init_w = theano.shared(np.asarray(np.random.uniform(low = -np.sqrt(6. / (inp + outp)), high = np.sqrt(6. / (inp + outp)), size = shape), dtype=theano.config.floatX))   # http://nbviewer.jupyter.org/github/vkaynig/ComputeFest2015_DeepLearning/blob/master/Fully_Connected_Networks.ipynb
            
        return init_w



    @staticmethod
    def objectives(output, target, name, nrows_X):
        
        if name == 'categorical_crossentropy':                                           # negative log-likelihood
            
            return T.nnet.categorical_crossentropy(output, target)
            
        if name == 'binary_crossentropy':
            
            return T.nnet.binary_crossentropy(output, target)
            
        if name == 'mean_squared_error':
            
            return T.sqr(output - target)/float(nrows_X)                                   # mean-square-error divided by number of rows of train-data [ see linear regression ]
        
        if name == 'root_mean_squared_error':
            
            return T.sqrt(T.sqr(output - target)/float(nrows_X))
            
        if name == 'mean_squared_logarithmic_error':                                         # https://github.com/fchollet/keras/blob/master/keras/objectives.py
            
            return T.sqr(T.log(T.clip(output, 1.0e-7, np.inf) + 1.) - T.log(T.clip(target, 1.0e-7, np.inf) + 1.)).mean(axis=-1)    


    @staticmethod
    def evaluate_early_stopping(y_true, y_pred, linear_regression):
        
        if not linear_regression:
            
            out = log_loss(y_true, y_pred)                                                  # log-loss
            
        else:
            
            out = np.mean([(y_pred[i] - y_true[i])**2 for i in range(len(y_true))])         # mse
    
        return out
        
    @staticmethod
    def shared_dataset(data_shared, borrow=True):
        
        shared_x = theano.shared(np.asarray(data_shared, dtype=theano.config.floatX), borrow=borrow)
        
        return shared_x


    def fit(self):
        
        #if self.batch_size is not None:
            
        index = T.lscalar('index') 
            
        # create shared data-sets in case of mini-batch
        train_X = self.shared_dataset(self.X_dat)       
        train_y = self.shared_dataset(self.y_dat)
        test_X = self.shared_dataset(self.X_test)
        
        if self.batch_size is not None:
            
            n_train_batches = train_X.get_value(borrow=True).shape[0] / self.batch_size
            n_test_batches = test_X.get_value(borrow=True).shape[0] / self.batch_size
        
        X = T.matrix()
        
        if self.linear_regression:
            
            Y = T.fvector()
        
        else:
            
            Y = T.matrix()
        
        
        if self.linear_regression:             
            
            self.w = self.initialize_weights((self.X_dat.shape[1]), self.X_dat.shape[1], 1, self.weights_initialization)            # initialize weights for the parameters ( linear regression )
            
            if self.add_bias:
                
                self.b = theano.shared(np.asarray(0, dtype=theano.config.floatX))                                                       # initialize bias to zero ( linear regression -- a single value )
            
                py_x = T.dot(X, self.w) + self.b                                                                                             # get predictions for linear regression
            
            else:
                
                py_x = T.dot(X, self.w)
            
        else:
            
            self.w = self.initialize_weights((self.X_dat.shape[1], self.y_dat.shape[1]), self.X_dat.shape[1], self.y_dat.shape[1], self.weights_initialization)       # initialize weights for the parameters ( logistic regression )
            
            if self.add_bias:
                
                self.b = theano.shared(np.zeros((self.y_dat.shape[1],), dtype=theano.config.floatX))                                    # initialize bias to zeros ( logistic regression -- a numpy array )
                
                py_x = T.nnet.softmax(T.dot(X, self.w) + self.b)                                                                                         # get probability predictions
                
            else:
                
                py_x = T.nnet.softmax(T.dot(X, self.w))
        
        
        cost = T.mean(self.objectives(py_x, Y, self.objective, self.X_dat.shape[0]))                                                    # objective function
        
        
        if self.L1 > 0.0 or self.L2 > 0.0:                                         # L1, L2 regularization [ when both used then 'elastic-net' ]
        
            if self.add_bias:
                
                reg_param_L1  = abs(T.sum(self.w) + T.sum(self.b))                               # L1 regrularization
                
                reg_param_L2 = T.sum(T.sqr(self.w)) + T.sum(T.sqr(self.b))                       # L2 regularization
    
                cost = cost + self.L1 * reg_param_L1 + self.L2 * reg_param_L2
                
            else:
                
                reg_param_L1  = abs(T.sum(self.w))                                         # L1 regrularization
                
                reg_param_L2 = T.sum(T.sqr(self.w))                                        # L2 regularization
    
                cost = cost + self.L1 * reg_param_L1 + self.L2 * reg_param_L2
        
        if self.add_bias:
            
            Params = [self.w, self.b]
            
        else:
            
            Params = [self.w]
        
       
        if self.batch_size is None:
            
            train = theano.function(inputs = [index], outputs = cost, 
                                    
                                    updates = Optimizers_update(cost, Params, self.learning_rate, self.optimizer).run_optimizer(),

                                    givens = { X: train_X[0:index], Y: train_y[0:index] }, allow_input_downcast = True)         # Compile [ call external class Optimizers_update ]

            predict_valid = theano.function(inputs = [index], outputs = py_x, givens = { X: test_X[0:index]}, allow_input_downcast = True)
           
        else:
            
            train = theano.function(inputs = [index], outputs = cost, updates = Optimizers_update(cost, Params, self.learning_rate, self.optimizer).run_optimizer(),
                                    
                                    givens = { X: train_X[index * self.batch_size: (index + 1) * self.batch_size],
                                                          
                                               Y: train_y[index * self.batch_size: (index + 1) * self.batch_size]}, allow_input_downcast = True)
            
            predict_valid = theano.function(inputs = [index], outputs = py_x, givens = { X: test_X[index * self.batch_size: (index + 1) * self.batch_size]}, allow_input_downcast = True)     # prediction function for validation set
        
        
        self.predict = theano.function(inputs = [X], outputs = py_x)                                                                    # predictions function

        early_stopping = []                                                                                                             # early stopping
    
        consecutive_increases_OR_decreases = 0 
        
        for i in range(self.iters):
            
            if self.batch_size is None:

                cost_train = train(self.X_dat.shape[0])
                
                if self.custom_eval is None:
                    
                    cost_valid = self.evaluate_early_stopping(self.Y_test, self.predict(self.X_test), self.linear_regression)
                    
                else:
                    
                    cost_valid = self.custom_eval[0](self.Y_test, self.predict(self.X_test))
                
            else:
                
                for batch_index_train in range(n_train_batches):

                    cost_train = train(batch_index_train)
                    
                if self.custom_eval is None:
                    
                    cost_valid = np.mean([self.evaluate_early_stopping(self.Y_test[batch_index_test * self.batch_size: (batch_index_test + 1) * self.batch_size], predict_valid(batch_index_test), self.linear_regression) for batch_index_test in range(n_test_batches)])
            
                else:
                    
                    cost_valid = np.mean([self.custom_eval[0](self.Y_test[batch_index_test * self.batch_size: (batch_index_test + 1) * self.batch_size], predict_valid(batch_index_test)) for batch_index_test in range(n_test_batches)])
            
            try:
                
                if self.custom_eval is None:
                    
                    print 'iter', str(i+1), '  train_loss ', str(np.round(cost_train, 3)), '  test_loss ', str(np.round(cost_valid, 3))
                    
                else:
                    
                    print 'iter', str(i+1), '  train_loss ', str(np.round(cost_train, 3)), '  test_' + self.custom_eval[1], ' ', str(np.round(cost_valid, 3))
            
            except:
                
                ValueError
            
            early_stopping.append(cost_valid)
            
            if not self.maximize:
                
                change_sign = len(early_stopping) >= 2 and early_stopping[-1] > early_stopping[-2]
                increase = 'increases'
                    
            else:
                
                change_sign = len(early_stopping) >= 2 and early_stopping[-1] < early_stopping[-2]
                decrease = 'decreases'
                    
            if change_sign:
                
                consecutive_increases_OR_decreases +=1
            else:
                consecutive_increases_OR_decreases = 0
                    
                
            if (consecutive_increases_OR_decreases >= self.early_stopping_rounds):
                
                if not self.maximize:
                    
                    print 'regression stopped after ', str(consecutive_increases_OR_decreases),' consecutive ', increase,' of loss and ',str(i+1),' Epochs'
                
                    break
                
                else:
                    print 'regression stopped after ', str(consecutive_increases_OR_decreases), ' consecutive ', decrease, ' of loss and ', str(i+1), ' Epochs'
                
                    break
                
                
            if np.isinf(cost_valid) or np.isnan(cost_valid):
                
                print 'Inf or nan values present after', str(i), 'Epochs'
                
                break

            
            
    def PREDICT(self, variable):                                                       # predictions function to predict unknown data
        
        preds = self.predict(variable)
        
        return preds
        
        
        
    def WeightsBias(self):
        
        if self.add_bias:
            
            weights, bias = self.w.get_value(), self.b.get_value()
            
            return weights, bias
            
        else:
            
             weights = self.w.get_value()
             
             return weights


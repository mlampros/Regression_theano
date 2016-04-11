from collections import OrderedDict
import theano
import theano.tensor as T


class Optimizers_update(object):
    
    def __init__(self, Cost, Params, Learning_rate, opt_name):

        self.Cost = Cost
        self.Params = Params
        self.Learning_rate = Learning_rate
        self.opt_name = opt_name

    @staticmethod
    def sgd(Cost, Params, Learning_rate):                                         # based on https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py  with minor changes
        
        Grads = T.grad(Cost, Params)    
        
        updates = OrderedDict()
    
        for param, grad in zip(Params, Grads):
            
            updates[param] = param - Learning_rate * grad
    
        return updates     
    
    
    def run_optimizer(self):
        
        if self.opt_name == 'sgd':
            
            opt_method = self.sgd(self.Cost, self.Params, self.Learning_rate)
             
        return opt_method            


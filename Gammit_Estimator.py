import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import gamma
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.special import gamma as gamma_func


class Gammit_Estimator():
    def import_data(self,x,y,T=None):
        self.x = x
        self.y = y
        self.T = T
        return self

    def _create_likelihood_function(self,form = "gamma"):
        if form == "gamma":
            if self.T is None:
                def ell(param):
                    alpha = param[:-1]
                    beta = param[-1]
                    return -np.mean(gamma.logpdf(self.y,self.x @ alpha, scale =  beta), axis = 0)
            else:
                def ell(param):
                    alpha = param[:-1]
                    beta = param[-1]
                    return -np.mean(gamma.logpdf(self.y,self.x @ alpha, scale = beta)*(self.y < self.T) 
                                    + gamma.logsf(self.T,self.x@alpha,scale = beta)*(self.y>=self.T), axis = 0)
        elif form == "normal":
            if self.T is None:
                def ell(param):
                    alpha = param[:-1]
                    sigma = param[-1]
                    return -np.mean(norm.logpdf(self.y,self.x @ alpha, scale =  sigma), axis = 0)
            else:
                def ell(param):
                    alpha = param[:-1]
                    sigma = param[-1]
                    return -np.mean(norm.logpdf(self.y,self.x @ alpha, scale = sigma)*(self.y < self.T) 
                                    + norm.logsf(self.T,self.x@alpha,scale = sigma)*(self.y>=self.T), axis = 0)
        return ell
    def estimate_mean_effect(self,n_points = 10,boundaries = (0.5,5),form = "gamma"):
        ell = self._create_likelihood_function(form = form)
        check_points = np.linspace(boundaries[0],boundaries[1],n_points)
        outputs = np.zeros((n_points,n_points,n_points))
        # Do a grid search first
        #print("Finding start point for optimization:")
        for i_1,a_1 in enumerate(check_points):
            #print(f"\r{i_1*10}% completed.",end = "")
            for i_2,a_2 in enumerate(check_points):
                for i_3,b in enumerate(check_points):
                        outputs[i_1,i_2,i_3] = ell(np.array([a_1,a_2,b]))

        #print("\r100% completed.")
        val = np.argmin(outputs)
        coor = [(val//(n_points**2)),(val//(n_points))%n_points,val%n_points]
        points = [check_points[coor[i]] for i in range(3)]

        # Now use a more robust minimizer
        param = minimize(ell,np.array(points))
        self.alpha_hat = param.x[:-1]
        self.beta_hat = param.x[-1]
        self.V = param.hess_inv
        if form == "gamma":
            return self.alpha_hat[1]*self.beta_hat
        elif form == "normal":
            return self.alpha_hat[1]
    
    def compute_standard_error(self):
        x = self.x
        y = self.y
        n = len(x)
        T = self.T
        alpha_hat = self.alpha_hat
        beta_hat = self.beta_hat
        
        V =self.V
        J = np.array([[0,beta_hat,alpha_hat[1]]])
        variance = np.squeeze(J@V@J.T)
        standard_error = np.sqrt(variance/n)
        return standard_error
        

        

                


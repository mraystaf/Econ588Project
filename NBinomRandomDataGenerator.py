import numpy as np
from scipy.stats import nbinom, norm, uniform, logistic
from matplotlib import pyplot as plt
import pandas as pd
from AbstractRandomDataGenerator import AbstractRandomDataGenerator

class NBinomRandomDataGenerator(AbstractRandomDataGenerator):
    def __init__(self, size, x_distribution='uniform', x_loc=30000, x_scale=100000, beta=np.array([0, -4e-5])):
        self.size = size
        self.x_distribution = x_distribution
        self.x_loc = x_loc
        self.x_scale = x_scale
        self.beta = beta.reshape(-1,1).T
        return None

    def _generate_uncensored_data(self):
        """Creates the x-values and the uncensored y-values (which I'll call y_star)"""
        # Generate the x data
        if self.x_distribution == 'uniform':
            x = uniform.rvs(loc=self.x_loc, scale=self.x_scale, size=(self.size,1))
        elif self.x_distribution == 'normal':
            x = norm.rvs(loc=self.x_loc, scale=self.x_scale, size=(self.size,1))
        else:
            raise ValueError("Unrecognized x distribution. Allowed distributions are \'uniform\' and \'normal\'")
        X = np.hstack((np.ones_like(x), x))

        # Generate the p parameter for the negative binomial based on the x's and beta
        # p represents the probability of success (aka, the probability that your job application leads to an offer)
        # I decided to use a sigmoid
        p = logistic.cdf(self.beta@(X.T)) # Probability of success

        # Generate y_star (uncensored data)
        # Each individual has their own nbinom distribution that they face, with their own distinct probability (which, in our perfect world, depends entirely upon their desired salary)
        # Therefore, yi should be a random draw from a nbinom with n=n, p=np.exp(beta@(x.T)) / (np.exp(beta@(x.T)) + 1)

        # NOTE: There are some problems when p is too small
        try:
            y_star = nbinom.rvs(n=1, p=p)
        except ValueError:
            raise ValueError("The probability values are too close to zero. The y_star values are approaching infinity. Consider a beta value closer to 0")

        return np.squeeze(x.T), y_star
    
    def _generate_plots(self, x, y_star, y):
        fig, axs = plt.subplots(1,2, figsize=(16,8))

        # Plot y_star against x
        axs[0].plot(x, y_star, 'ro', markersize=2)
        axs[0].set_xlabel("x (desired starting salary)")
        axs[0].set_ylabel("$y^*$ (number of applications)")
        axs[0].set_title("Uncensored plot")

        # Plot y against x
        axs[1].plot(x, y, 'ro', markersize=2)
        axs[1].set_xlabel("x (desired starting salary)")
        axs[1].set_ylabel("$y$ (number of applications)")
        axs[1].set_title("Censored plot")
        # Share y-axis
        axs[1].sharey(axs[0])
        plt.suptitle(f"Randomly Generated Data (NegBin, {self.percent_censored * 100}% Censored)")

        plt.tight_layout()
        plt.show()


    def createUniversallyCensoredData(self, T, generate_plots=False):
        """Creates data where all the y-values are capped at T"""

        # Obtain uncensored data
        x, y_star = self._generate_uncensored_data()

        # Cap all of the y data at T_i
        y = np.array([np.min([yi, T]) for yi in y_star])
        censor_mask = T <= y_star

        # Build the dataframe to return
        data = pd.DataFrame()
        data['Desired Starting Salary ($)'] = x
        data['Rejected Job Applications'] = y
        data['Censored'] = censor_mask.astype(int)

        if generate_plots:
            self._generate_plots(x=x, y_star=y_star, y=y)

        return data
    
    def createVaryingCensoredData(self, percent_censored=0.3, generate_plots=False):
        self.percent_censored = percent_censored

        # Obtain uncensored data
        x, y_star = self._generate_uncensored_data()

        # Randomly select y_star values and censor them by a random value
        # Choose the observations to be censored
        censor_mask = np.random.choice(self.size, round(percent_censored * self.size), replace=False) 
        y = y_star.copy()
        # Values selected to be censored are multiplied by a number between 0 and 1
        y[censor_mask] = (uniform.rvs(loc=0, scale=1, size=(censor_mask.size,)) * y_star[censor_mask]).round()

        # Build the dataframe to return
        data = pd.DataFrame()
        data['Desired Starting Salary ($)'] = x
        data['Rejected Job Applications'] = y
        # data['Censored'] = (y < y_star).astype(int) # TOOD: This doesn't handle the edge case well
        data['Censored'] = 0
        data.loc[censor_mask, 'Censored'] = 1

        if generate_plots:
            self._generate_plots(x=x, y_star=y_star, y=y)

        return data    
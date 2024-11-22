import numpy as np
from scipy.stats import nbinom, norm, binom, uniform
from matplotlib import pyplot as plt
import pandas as pd
from abc import ABC, abstractmethod

class AbstractRandomDataGenerator:
    @abstractmethod
    def __init__(self):
        pass

    def _generate_uncensored_data(self):
        pass

    def _generate_plots(self, x, y_star, y):
        pass

    def createUniversallyCensoredData(self):
        pass

    def createVaryingCensoredData(self):
        pass
        
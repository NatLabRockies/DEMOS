from .. import modelmanager
from .template import TemplateStep
from scipy.special import softmax
import pandas as pd
import numpy as np

@modelmanager.template
class MultinomialLogitStep(TemplateStep):
    def __init__(self, tables=None, model_expression=None, filters=None, out_tables=None,
                 out_column=None, out_filters=None, name=None, tags=[], model_coeffs=None, spec_names=None):
        # Parent class can initialize the standard parameters
        TemplateStep.__init__(self, tables=tables, model_expression=model_expression,
                              filters=filters, out_tables=out_tables, out_column=out_column,
                              out_transform=None, out_filters=out_filters, name=name, tags=tags)

        self.coeffs = pd.DataFrame(model_coeffs)
        self.variable_names = spec_names

    @classmethod
    def from_dict(cls, d):
        obj = cls(tables=d['tables'], out_tables=d['out_tables'], name=d['name'], filters=d['filters'], model_coeffs=d['model_coeffs'], \
                  spec_names=d['spec_names'])
        return obj

    def to_dict(self):
        #TODO generate a dict from self obj
        pass

    def fit(self):
        #TODO add training process
        pass

    def predict(self, data):
        if self.coeffs is None:
            raise Exception("coeffs in MNL fail to be loaded.")
        data = data.loc[:, self.variable_names]
        utils = np.dot(data, self.coeffs)
        base_util = np.zeros(utils.shape[0])
        utils = np.column_stack((base_util, utils))
        probabilities = softmax(utils, axis=1)
        s = probabilities.cumsum(axis=1)
        r = np.random.rand(probabilities.shape[0]).reshape((-1, 1))
        choices = (s < r).sum(axis=1)
        return pd.Series(index=data.index, data=choices)

    def run(self, data):
        """Function to run simulation of the MNL model

            Args:
                data (_type_): _description_
                coeffs (_type_): _description_

            Returns:
                Pandas Series: Pandas Series of the outcomes of the simulated model
            """
        return self.predict(data)
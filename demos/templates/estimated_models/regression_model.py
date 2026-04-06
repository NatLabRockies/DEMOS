from __future__ import print_function

import numpy as np
import patsy
import pandas as pd
from statsmodels.api import OLS

import orca

from .. import modelmanager
from ..utils.misc import get_data
from .template import TemplateStep


@modelmanager.template
class RegressionStep(TemplateStep):
    """
    A class for building ordinary least squares (OLS) regression model steps. This
    extends TemplateStep, where some common functionality is defined. Estimation is
    handled by Statsmodels and simulation is handled within this class.

    Expected usage:
    - create a model object
    - specify some parameters
    - run the `fit()` method
    - iterate as needed

    Then, for simulation:
    - specify some simulation parameters
    - use the `run()` method for interactive testing
    - use `modelmanager.register()` to save the model to Orca and disk
    - registered steps can be accessed via ModelManager and Orca

    All parameters listed in the constructor can be set directly on the class object,
    at any time.

    Parameters
    ----------
    tables : str or list of str, optional
        Name(s) of Orca tables to draw data from. The first table is the primary one.
        Any additional tables need to have merge relationships ("broadcasts") specified
        so that they can be merged unambiguously onto the first table. Among them, the
        tables must contain all variables used in the model expression and filters. The
        left-hand-side variable should be in the primary table. The `tables` parameter is
        required for fitting a model, but it does not have to be provided when the object
        is created.

    model_expression : str, optional
        Patsy formula containing both the left- and right-hand sides of the model
        expression: http://patsy.readthedocs.io/en/latest/formulas.html
        This parameter is required for fitting a model, but it does not have to be
        provided when the object is created.

    filters : str or list of str, optional
        Filters to apply to the data before fitting the model. These are passed to
        `pd.DataFrame.query()`. Filters are applied after any additional tables are merged
        onto the primary one. Replaces the `fit_filters` argument in UrbanSim.

    out_tables : str or list of str, optional
        Name(s) of Orca tables to use for simulation. If not provided, the `tables`
        parameter will be used. Same guidance applies: the tables must be able to be
        merged unambiguously, and must include all columns used in the right-hand-side
        of the model expression and in the `out_filters`.

    out_column : str, optional
        Name of the column to write simulated values to. If not provided, the left-hand-
        side variable from the model expression will be used.

    out_filters : str or list of str, optional
        Filters to apply to the data before simulation. If not provided, no filters will
        be applied. Replaces the `predict_filters` argument in UrbanSim.

    name : str, optional
        Name of the model step, passed to ModelManager. If none is provided, a name is
        generated each time the `fit()` method runs.

    tags : list of str, optional
        Tags, passed to ModelManager.

    """

    def __init__(
        self,
        tables=None,
        model_expression=None,
        filters=None,
        out_tables=None,
        out_column=None,
        out_filters=None,
        name=None,
        tags=[],
    ):

        # Parent class can initialize the standard parameters
        TemplateStep.__init__(
            self,
            tables=tables,
            model_expression=model_expression,
            filters=filters,
            out_tables=out_tables,
            out_column=out_column,
            out_transform=None,
            out_filters=out_filters,
            name=name,
            tags=tags,
        )

        # Placeholders for model fit data, filled in by fit() or from_dict()
        self.summary_table = None
        self.fitted_parameters = None

    @classmethod
    def from_dict(cls, d):
        """
        Create an object instance from a saved dictionary representation.

        Parameters
        ----------
        d : dict

        Returns
        -------
        RegressionStep

        """
        obj = cls(
            tables=d["tables"],
            model_expression=d["model_expression"],
            filters=d["filters"],
            out_tables=d["out_tables"],
            out_column=d["out_column"],
            out_filters=d["out_filters"],
            name=d["name"],
            tags=d["tags"],
        )

        obj.summary_table = d["summary_table"]
        obj.fitted_parameters = d["fitted_parameters"]

        return obj

    def to_dict(self):
        """
        Create a dictionary representation of the object.

        Returns
        -------
        dict

        """
        d = TemplateStep.to_dict(self)

        # Add parameters not in parent class
        d.update(
            {
                "summary_table": self.summary_table,
                "fitted_parameters": self.fitted_parameters,
            }
        )
        return d

    def fit(self):
        """
        Fit the model; save and report results. Uses the Statsmodels OLS class with
        default estimation settings.

        The `fit()` method can be run as many times as desired. Results will not be saved
        with Orca or ModelManager until the `register()` method is run.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        df = get_data(
            tables=self.tables,
            filters=self.filters,
            model_expression=self.model_expression,
        )

        m = OLS.from_formula(data=df, formula=self.model_expression)
        results = m.fit()

        self.name = self._generate_name()
        self.summary_table = str(results.summary())
        print(self.summary_table)

        self.fitted_parameters = results.params.tolist()  # params is a pd.Series

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate continuous predictions from fitted OLS parameters.

        Parameters
        ----------
        data : pd.DataFrame
            Input data containing the right-hand-side variables of the model expression.

        Returns
        -------
        pd.Series
            Predicted values with the same index as `data`.

        """
        data = data.sort_index(axis=0)
        rhs = self.model_expression.split("~", 1)[1]
        dm = patsy.dmatrix(data=data, formula_like=rhs, return_type="dataframe")

        predictions = np.dot(dm, self.fitted_parameters)
        return pd.Series(predictions, index=data.index)

    def run(self):
        """
        Run the model step: calculate predicted values and use them to update a column.

        Predicted values are saved to the class object for interactive use
        (`choices`, with type pd.Series) but are not persisted in the dictionary
        representation of the model step.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        df = get_data(
            tables=self.out_tables,
            fallback_tables=self.tables,
            filters=self.out_filters,
            model_expression=self.model_expression,
            extra_columns=self.out_column,
        )

        predictions = self.predict(df)
        self.choices = predictions

        colname = self._get_out_column()
        tabname = self._get_out_table()

        orca.get_table(tabname).update_col_from_series(colname, predictions, cast=True)

    def run_with_data(self, df):
        """
        Run prediction on a provided DataFrame and return the result without updating
        any Orca tables.

        Parameters
        ----------
        df : pd.DataFrame
            Input data containing the right-hand-side variables of the model expression.

        Returns
        -------
        pd.Series
            Predicted values with the same index as `df`.

        """
        predictions = self.predict(df)
        self.choices = predictions
        return predictions

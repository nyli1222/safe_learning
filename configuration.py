"""General configuration class for dtypes."""

from __future__ import absolute_import, print_function, division

import numpy as np
import torch

class Configuration(object):
    """Configuration class."""

    def __init__(self):
        """Initialization."""
        super(Configuration, self).__init__()

        # Dtype for computations
        self.dtype = torch.float64


    @property
    def np_dtype(self):
        """Return the numpy dtype."""
        return np.float64

    def __repr__(self):
        """Print the parameters."""
        params = ['Configuration parameters:', '']
        for param, value in self.__dict__.items():
            params.append('{}: {}'.format(param, value.__repr__()))

        return '\n'.join(params)

config = Configuration()
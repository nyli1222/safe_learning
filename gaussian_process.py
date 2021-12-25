import gpytorch
import torch
from gpytorch.lazy import RootLazyTensor
from gpytorch.lazy import MatmulLazyTensor
import numpy as np
from configuration import config

class GaussianProcess():
  
  def __init__(self, kernel, mean_function, input_dim):
    """Initialization."""
    super(GaussianProcess, self).__init__()
    self.kernel= kernel
    self.mean_function = mean_function
    self.input_dim = input_dim

    self.likelihood_variance = 0.001 ** 2
  
  def predict(self, Xnew):
    
    if not hasattr(self, "X"):  #predict based on GP prior
      
      fmean = self.mean_function(Xnew)
      fvar = self.kernel(Xnew).diag()
      return fmean.reshape(-1, 1), fvar.reshape(-1, 1)


    else:
      Kx = self.kernel(self.X, Xnew).evaluate()
      v = torch.triangular_solve(Kx, self.L, upper=False)[0]
      a = torch.triangular_solve(self.Y - self.mean_function(self.X).reshape(-1,1), self.L, upper=False)[0]
      fmean = torch.matmul(torch.transpose(a,0,1), v) + self.mean_function(Xnew)
      
      fvar = self.kernel(Xnew).diag() - torch.sum(torch.square(v), 0)
      return fmean.reshape(-1, 1), fvar.reshape(-1, 1)

  def add_data_point(self, X, Y):
        """Add data points to the GP model 
        Parameters
        ----------
        x : ndarray
            A 2d array with the new states to add to the GP model. Each new
            state is on a new row.
        y : ndarray
            A 2d array with the new measurements to add to the GP model.
            Each measurements is on a new row.
        """
        if not hasattr(self, "X"):  
          self.X = X
          self.Y = Y
      
        else:
          self.X = torch.concat((self.X, X))
          self.Y = torch.concat((self.Y, Y))
      
        K = self.kernel(self.X).evaluate() + torch.eye(self.X.shape[0], dtype=config.dtype) * self.likelihood_variance
        #Todo
        self.L = torch.linalg.cholesky(K).detach()



class LinearKernel(gpytorch.kernels.Kernel):
    
    def __init__(self, variances, active_dims=None):
      super(LinearKernel, self).__init__(active_dims=active_dims)
      self.variances = variances
    
    # this is the kernel function
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        
        x1_ = torch.matmul(x1,self.variances.sqrt())
        if last_dim_is_batch:
            x1_ = x1_.transpose(-1, -2).unsqueeze(-1)

        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLazyTensor when x1 == x2 for efficiency when composing
            # with other kernels
            prod = RootLazyTensor(x1_)

        else:
            x2_ = torch.matmul(x2,self.variances.sqrt())
            if last_dim_is_batch:
                x2_ = x2_.transpose(-1, -2).unsqueeze(-1)

            prod = MatmulLazyTensor(x1_, x2_.transpose(-2, -1))

        if diag:
            return prod.diag()
        else:
            return prod

        if diag:
            return prod.diag()
        else:
            return prod

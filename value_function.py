from scipy import sparse, spatial
from itertools import product as cartesian
import numpy as np
import torch
from configuration import config

class V(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, value_function):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.save_for_backward(input)
        ctx.value_function = value_function
        return torch.tensor(value_function.compute_value(input.detach().numpy()))
 
    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        input, = ctx.saved_tensors
        grad = ctx.value_function.tri.gradient(input.detach().numpy())
        grad_input = grad_output.clone()
        grad_input = grad_input*torch.tensor(grad)
        return grad_input, None


class Triangulation():
    def __init__(self, discretization, vertex_values, project=True):
      self.tri = _Triangulation(discretization,project=project)
      self.tri.parameters = vertex_values
    @property
    def project(self):
        """Getter for the project parameter."""
        return self.tri.project

    @project.setter
    def project(self, value):
        """Setter for the project parameter."""
        self.tri.project = value

    @property
    def discretization(self):
        """Getter for the discretization."""
        return self.tri.discretization

    @property
    def nindex(self):
        """Return the number of parameters."""
        return self.tri.nindex


    def _get_hyperplanes(self, points):
        """Return the linear weights associated with points.
        Parameters
        ----------
        points : 2d array
            Each row represents one point
        Returns
        -------
        weights : ndarray
            An array that contains the linear weights for each point.
        hyperplanes : ndarray
            The corresponding hyperplane objects.
        simplices : ndarray
            The indices of the simplices associated with each points
        """
        simplex_ids = self.tri.find_simplex(points)

        simplices = self.tri.simplices(simplex_ids).astype(np.int64)
        origins = self.tri.discretization.index_to_state(simplices[:, 0])

        # Get hyperplane equations
        simplex_ids %= self.tri.triangulation.nsimplex
        hyperplanes = self.tri.hyperplanes[simplex_ids]

        # Pre-multiply each hyperplane by (point - origin)
        return origins, hyperplanes, simplices

    def compute_value(self, points: np.array):
        # Get the appropriate hyperplane
        origins, hyperplanes, simplices = self._get_hyperplanes(points)
        
        # Project points onto the grid of triangles.
        if self.project:
            clip_min = self.tri.limits[:, 0]
            clip_max = self.tri.limits[:, 1]

            #points = tf.minimum(tf.maximum(points, clip_min), clip_max)

        # Compute weights (barycentric coordinates)
        offset = points - origins
        w1 = np.sum(offset[:, :, None] * hyperplanes, axis=1)
        w0 = 1 - np.sum(w1, axis=1, keepdims=True)
        weights = np.concatenate((w0, w1), axis=1)

        # Collect the value on the vertices
        
        #parameter_vector = tf.gather(self.parameters[0],
        #                             indices=simplices,
        #                             validate_indices=False)
        # parameter_vector shape: [num_points, 3, 1]
        parameter_vector = self.tri.parameters[simplices]
        # Compute the values
        return np.sum(weights[:, :, None] * parameter_vector, axis=1)


    def __call__(self, points):
        if isinstance(points, torch.Tensor):
            return V.apply(points, self)
        else:
            return self.compute_value(points)

    def gradient(self, states):
        if isinstance(states, torch.Tensor):
            states = states.detach().numpy()
            grad = self.tri.gradient(states)
            return torch.tensor(grad, dtype=config.dtype)
        else:
            return self.tri.gradient(states)

class _Triangulation():
    """
    Efficient Delaunay triangulation on regular grids.
    This class is a wrapper around scipy.spatial.Delaunay for regular grids. It
    splits the space into regular hyperrectangles and then computes a Delaunay
    triangulation for only one of them. This single triangulation is then
    generalized to other hyperrectangles, without ever maintaining the full
    triangulation for all individual hyperrectangles.
    Parameters
    ----------
    discretization : instance of discretization
        For example, an instance of `GridWorld`.
    vertex_values: arraylike, optional
        A 2D array with the values at the vertices of the grid on each row.
    project: bool, optional
        Whether to project points onto the limits.
    """

    def __init__(self, discretization, vertex_values=None, project=False):
        """Initialization."""
        super(_Triangulation, self).__init__()

        self.discretization = discretization
        self.input_dim = discretization.ndim

        self._parameters = None
        self.parameters = vertex_values

        disc = self.discretization

    
        product = cartesian(*np.diag(disc.unit_maxes))
        hyperrectangle_corners = np.array(list(product),
                                              dtype=config.np_dtype)
        self.triangulation = spatial.Delaunay(hyperrectangle_corners)
        self.unit_simplices = self._triangulation_simplex_indices()

        # Some statistics about the triangulation
        self.nsimplex = self.triangulation.nsimplex * disc.nrectangles

        # Parameters for the hyperplanes of the triangulation
        self.hyperplanes = None
        self._update_hyperplanes()

        self.project = project

    @property
    def output_dim(self):
        """Return the output dimensions of the function."""
        if self.parameters is not None:
            return self.parameters.shape[1]

    @property
    def parameters(self):
        """Return the vertex values."""
        return self._parameters

    @parameters.setter
    def parameters(self, values):
        """Set the vertex values."""
        if values is None:
            self._parameters = values
        else:
            values = np.asarray(values).reshape(self.nindex, -1)
            self._parameters = values

    @property
    def limits(self):
        """Return the discretization limits."""
        return self.discretization.limits

    @property
    def nindex(self):
        """Return the number of discretization indices."""
        return self.discretization.nindex

    def _triangulation_simplex_indices(self):
        """Return the simplex indices in our coordinates.
        Returns
        -------
        simplices: ndarray (int)
            The simplices array in our extended coordinate system.
        Notes
        -----
        This is only used once in the initialization.
        """
        disc = self.discretization
        simplices = self.triangulation.simplices
        new_simplices = np.empty_like(simplices)

        # Convert the points to out indices
        index_mapping = disc.state_to_index(self.triangulation.points +
                                            disc.offset)

        # Replace each index with out new_index in index_mapping
        for i, new_index in enumerate(index_mapping):
            new_simplices[simplices == i] = new_index
        return new_simplices

    def _update_hyperplanes(self):
        """Compute the simplex hyperplane parameters on the triangulation."""
        self.hyperplanes = np.empty((self.triangulation.nsimplex,
                                     self.input_dim, self.input_dim),
                                    dtype=config.np_dtype)

        # Use that the bottom-left rectangle has the index zero, so that the
        # index numbers of scipy correspond to ours.
        for i, simplex in enumerate(self.unit_simplices):
            simplex_points = self.discretization.index_to_state(simplex)
            self.hyperplanes[i] = np.linalg.inv(simplex_points[1:] -
                                                simplex_points[:1])

    def find_simplex(self, points):
        """Find the simplices corresponding to points.
        Parameters
        ----------
        points : 2darray
        Returns
        -------
        simplices : np.array (int)
            The indices of the simplices
        """
        disc = self.discretization
        rectangles = disc.state_to_rectangle(points)

        # Convert to unit coordinates
        points = disc._center_states(points, clip=True)

        # Convert to basic hyperrectangle coordinates and find simplex
        unit_coordinates = points % disc.unit_maxes
        simplex_ids = self.triangulation.find_simplex(unit_coordinates)
        simplex_ids = np.atleast_1d(simplex_ids)

        # Adjust for the hyperrectangle index
        simplex_ids += rectangles * self.triangulation.nsimplex

        return simplex_ids

    def simplices(self, indices):
        """Return the simplices corresponding to the simplex index.
        Parameters
        ----------
        indices : ndarray
            The indices of the simpleces
        Returns
        -------
        simplices : ndarray
            Each row consists of the indices of the simplex corners.
        """
        # Get the indices inside the unit rectangle
        unit_indices = np.remainder(indices, self.triangulation.nsimplex)
        simplices = self.unit_simplices[unit_indices].copy()

        # Shift indices to corresponding rectangle
        rectangles = np.floor_divide(indices, self.triangulation.nsimplex)
        corner_index = self.discretization.rectangle_corner_index(rectangles)

        if simplices.ndim > 1:
            corner_index = corner_index[:, None]

        simplices += corner_index
        return simplices

    def _get_weights(self, points):
        """Return the linear weights associated with points.
        Parameters
        ----------
        points : 2d array
            Each row represents one point
        Returns
        -------
        weights : ndarray
            An array that contains the linear weights for each point.
        simplices : ndarray
            The indices of the simplices associated with each points
        """
        disc = self.discretization
        simplex_ids = self.find_simplex(points)

        simplices = self.simplices(simplex_ids)
        origins = disc.index_to_state(simplices[:, 0])

        # Get hyperplane equations
        simplex_ids %= self.triangulation.nsimplex
        hyperplanes = self.hyperplanes[simplex_ids]

        # Some numbers for convenience
        nsimp = self.input_dim + 1
        npoints = len(points)

        if self.project:
            points = np.clip(points, disc.limits[:, 0], disc.limits[:, 1])

        weights = np.empty((npoints, nsimp), dtype=config.np_dtype)

        # Pre-multiply each hyperplane by (point - origin)
        offset = points - origins
        np.sum(offset[:, :, None] * hyperplanes, axis=1, out=weights[:, 1:])

        # The weights have to add up to one
        weights[:, 0] = 1 - np.sum(weights[:, 1:], axis=1)

        return weights, simplices

    def build_evaluation(self, points):
        """Return the function values.
        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.
        Returns
        -------
        values : ndarray
            The function values at the points.
        """
        points = np.atleast_2d(points)
        weights, simplices = self._get_weights(points)

        # Return function values
        parameter_vector = self.parameters[simplices]

        # Broadcast the weights along output dimensions
        return np.sum(weights[:, :, None] * parameter_vector, axis=1)

    def parameter_derivative(self, points):
        """
        Obtain function values at points from triangulation.
        This function returns a sparse matrix that, when multiplied
        with the vector with all the function values on the vertices,
        returns the function values at points.
        Parameters
        ----------
        points : 2d array
            Each row represents one point.
        Returns
        -------
        values
            A sparse matrix B so that evaluate(points) = B.dot(parameters).
        """
        points = np.atleast_2d(points)
        weights, simplices = self._get_weights(points)
        # Construct sparse matrix for optimization

        nsimp = self.input_dim + 1
        npoints = len(simplices)
        # Indices of constraints (nsimp points per simplex, so we have nsimp
        # values in each row; one for each simplex)
        rows = np.repeat(np.arange(len(points)), nsimp)
        cols = simplices.ravel()

        return sparse.coo_matrix((weights.ravel(), (rows, cols)),
                                 shape=(npoints, self.discretization.nindex))

    def _get_weights_gradient(self, points=None, indices=None):
        """Return the linear gradient weights associated with points.
        Parameters
        ----------
        points : ndarray
            Each row represents one point.
        indices : ndarray
            Each row represents one index. Ignored if points
        Returns
        -------
        weights : ndarray
            An array that contains the linear weights for each point.
        simplices : ndarray
            The indices of the simplices associated with each points
        """
        if points is None:
            simplex_ids = np.atleast_1d(indices)
        elif indices is None:
            simplex_ids = self.find_simplex(points)
        else:
            raise TypeError('Need to provide at least one input argument.')
        simplices = self.simplices(simplex_ids)

        # Get hyperplane equations
        simplex_ids %= self.triangulation.nsimplex

        # Some numbers for convenience
        nsimp = self.input_dim + 1
        npoints = len(simplex_ids)

        # weights
        weights = np.empty((npoints, self.input_dim, nsimp),
                           dtype=config.np_dtype)

        weights[:, :, 1:] = self.hyperplanes[simplex_ids]
        weights[:, :, 0] = -np.sum(weights[:, :, 1:], axis=2)
        return weights, simplices

    def gradient(self, points):
        """Return the gradient.
        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.
        Returns
        -------
        gradient : ndarray
            The function gradient at the points. A 3D array with the gradient
            at the i-th data points for the j-th output with regard to the k-th
            dimension stored at (i, j, k). The j-th dimension is squeezed out
            for 1D functions.
        """
        points = np.atleast_2d(points)
        weights, simplices = self._get_weights_gradient(points)
        # Return function values if desired
        res = np.einsum('ijk,ikl->ilj', weights, self.parameters[simplices, :])
        if res.shape[1] == 1:
            res = res.squeeze(axis=1)
        return res

    def gradient_parameter_derivative(self, points=None, indices=None):
        """
        Return the gradients at the respective points.
        This function returns a sparse matrix that, when multiplied
        with the vector of all the function values on the vertices,
        returns the gradients. Note that after the product you have to call
        ```np.reshape(grad, (ndim, -1))``` in order to obtain a proper
        gradient matrix.
        Parameters
        ----------
        points : ndarray
            Each row contains one state at which to evaluate the gradient.
        indices : ndarray
            The simplex indices. Ignored if points are provided.
        Returns
        -------
        gradient : scipy.sparse.coo_matrix
            A sparse matrix so that
            `grad(points) = B.dot(vertex_val).reshape(ndim, -1)` corresponds
            to the true gradients
        """
        weights, simplices = self._get_weights_gradient(points=points,
                                                        indices=indices)

        # Some numbers for convenience
        nsimp = self.input_dim + 1
        npoints = len(simplices)

        # Construct sparse matrix for optimization

        # Indices of constraints (ndim gradients for each point, which each
        # depend on the nsimp vertices of the simplex.
        rows = np.repeat(np.arange(npoints * self.input_dim), nsimp)
        cols = np.tile(simplices, (1, self.input_dim)).ravel()

        return sparse.coo_matrix((weights.ravel(), (rows, cols)),
                                 shape=(self.input_dim * npoints,
                                        self.discretization.nindex))

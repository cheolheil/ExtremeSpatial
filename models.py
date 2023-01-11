import numpy as np
import scipy.stats as stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel as W
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import mahalanobis
from scipy.optimize import minimize


class StochasticKriging:
    def __init__(self, input_dim, kernel=None, alpha=1e-10, n_restarts_optimizer=0, random_state=None):
        if kernel == None:
            kernel = C() * RBF(np.ones(input_dim)) + W(1e-1, [1e-5, 1e1])
        else:
            pass
        # self.latent_gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer, random_state=random_state)
        self.latent_gp = UniversalKriging(kernel=kernel, alpha=alpha, n_restart=n_restarts_optimizer, random_state=random_state)
        self.nugget_latent_gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer, random_state=random_state)
    
    def fit(self, X, y):
        self.X = X
        self.y = y.copy()
            
        # get row-wise variance of y
        self.y_var = np.var(y, axis=1, ddof=1)
        self.y_mean = np.mean(y, axis=1)
        
        # fit GPs on latent variance
        self.nugget_latent_gp.fit(X, np.log(self.y_var))
        self.latent_gp.fit(X, self.y_mean)
    
    def predict(self, x_new, return_std=False):
        # predict latent variables
        nugget_latent_pred = np.sqrt(np.exp(self.nugget_latent_gp.predict(x_new)))

        # predict y
        y_bar, y_std = self.latent_gp.predict(x_new, return_std=True)
        y_std += nugget_latent_pred
        if return_std:
            return y_bar, y_std
        else:
            return y_bar


class UniversalKriging:
    def __init__(self, fit_intercept=True, kernel=None, alpha=1e-10, n_restart=0, random_state=None):
        self.kernel = kernel
        self.linear_regressor = LinearRegression(fit_intercept=fit_intercept)
        self.gaussian_process = GaussianProcessRegressor(kernel=self.kernel, alpha=alpha, n_restarts_optimizer=n_restart, random_state=random_state)
    
    def fit(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        self.linear_regressor.fit(X, y)
        res = y - self.linear_regressor.predict(X)
        self.gaussian_process.fit(X, res)
    
    def predict(self, X, return_std=False):
        linear_regression_mean = self.linear_regressor.predict(X)
        sigma2_hat = np.sum(np.square(self.y - self.linear_regressor.predict(self.X))) / (self.X.shape[0] - self.X.shape[1] - 1)
        linear_regression_std = np.sqrt(sigma2_hat * (1 + np.diag(X @ np.linalg.inv(self.X.T @ self.X) @ X.T)))
        gp_mean, gp_std = self.gaussian_process.predict(X, return_std=True)
        if return_std:
            return linear_regression_mean + gp_mean, linear_regression_std + gp_std
        else:
            return linear_regression_mean + gp_mean


class LatentExtreme:
    def __init__(self, input_dim, kernel=None, linear_regression=True, **kwargs):
        if kernel is None:
            kernel = C(1.0, (1e-4, 1e2)) * RBF(np.ones(input_dim), [(1e-4, 1e3) for _ in range(input_dim)]) + W(1e-1, (1e-4, 1e2))
        else:
            pass
        
        if linear_regression:
            self.lr = LinearRegression()
        else:
            pass
        self.shape_gp = GaussianProcessRegressor(kernel=kernel, **kwargs)
        self.loc_gp = GaussianProcessRegressor().set_params(**self.shape_gp.get_params())
        self.scale_gp = GaussianProcessRegressor().set_params(**self.shape_gp.get_params())
        
    def fit(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        
        # fit GEV on block maxima
        params_ = np.zeros((self.X.shape[0], 3))
        for i in range(self.X.shape[0]):
            params_[i, :] = stats.genextreme.fit(self.y[i, :])

        # take log on scale parameters
        params_[:, 2] = np.log(params_[:, 2])
        # it can be integrated with UK
        if hasattr(self, 'lr'):
            self.lr.fit(self.X, params_)
            params_ -= self.lr.predict(self.X)
        else:
            pass
        self.shape_gp.fit(self.X, params_[:, 0])
        self.loc_gp.fit(self.X, params_[:, 1])
        self.scale_gp.fit(self.X, params_[:, 2])
    
    def predict(self, x_new, n_sample=1):
        y_pred = np.zeros((len(x_new), n_sample))
        for i, x in enumerate(x_new):
            y_pred[i] = stats.genextreme.rvs(*self.get_params(x), size=n_sample)
        return y_pred
    
    def predict_ppf(self, x_new, q=0.5):
        y_pred = np.zeros((len(x_new), len(q)))
        for i, x in enumerate(x_new):
            y_pred[i] = stats.genextreme.ppf(q, *self.get_params(x))
        return y_pred
    
    def get_params(self, x):
        x = np.atleast_2d(x)
        if hasattr(self, 'lr'):
            params = self.lr.predict(x)
        else:
            params = np.zeros((x.shape[0], 3))
        params[:, 0] += self.shape_gp.predict(x)
        params[:, 1] += self.loc_gp.predict(x)
        params[:, 2] += self.scale_gp.predict(x)
        params[:, 2] = np.exp(params[:, 2])
        return params.T

    def transform_unit_frec(self):
        params = self.get_params(self.X).T
        return (1 - params[:, [0]] / params[:, [2]] * (self.y - params[:, [1]])) ** (-1/params[:, [0]])


class SmithProcess:
    def __init__(self, input_dim, bounds=None, init_theta=None):
        self.input_dim = input_dim
        # check the given nodes are valid
        if bounds == None:
            self.bounds = ([0, 1] for _ in range(input_dim))
        else:
            self.bounds = bounds
        if init_theta == None:
            self.theta = np.ones(input_dim)
        else:
            self.theta = init_theta
    
    def bivariate_measure(self, x1, x2, z1, z2):
        if z1 == 0 or z2 == 0:
            return 0
        else:
            a = mahalanobis(x1, x2, np.diag(1 / self.theta))
            return stats.norm.cdf(a/2 + np.log(z2/z1) / a) / z1 + stats.norm.cdf(a/2 + np.log(z1/z2) / a) / z2
   
    def gen_nodes(self, n):
        self.nodes = np.random.rand(n, self.input_dim)

    def sample(self, x, n=100):
        # generate nodes
        self.gen_nodes(n)
        w = np.zeros((len(x), n))
        for i in range(n):
            w[:, i] = stats.multivariate_normal.pdf(x, self.nodes[i], np.diag(self.theta))
        return w
    
    def update_theta(self, theta):
        self.theta = theta


class SchlatherProcess(GaussianProcessRegressor):
    # sampling from a Schlather process is just sampling from a Gaussian process with a correlation function
    # However, the fitting process is different from the Gaussian process
    def __init__(self, input_dim, bounds=None, init_theta=None, **kwargs):
        if init_theta == None:
            self.theta = np.ones(input_dim)
        else:
            self.theta = init_theta
        kernel = RBF(self.theta)
        super().__init__(kernel=kernel, **kwargs)
        
        # check the given nodes are valid
        if bounds == None:
            self.bounds = ([0, 1] for _ in range(input_dim))
        else:
            self.bounds = bounds
        self.input_dim = input_dim
    
    def bivariate_measure(self, x1, x2, z1, z2):
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        if z1 == 0 or z2 == 0:
            return 0
        else:
            return 0.5 * (1 / z1 + 1 / z2) * (1 + np.sqrt(1 - 2 * (self.kernel(x1, x2).ravel() + 1) * z1 * z2 / np.square(z1 + z2)))

    def sample(self, x, n=100):
        # generate nodes
        return self.sample_y(x, n_samples=n, random_state=None)
    
    def update_theta(self, theta):
        self.theta = theta
        self.kernel.theta = np.log(self.theta)


class MaxStableProcess:
    def __init__(self, input_dim, random_process):
        self.input_dim = input_dim
        self.random_process = random_process
    
    def get_points(self, J):
        s = stats.poisson(1).rvs(J)
        return 1 / np.cumsum(s)[np.nonzero(np.cumsum(s))]
    
    def neg_composite_log_likelihood(self, theta):
        neg_ell = 0
        self.random_process.update_theta(theta)
        for i in range(self.y.shape[1]):            # number of replications
            for j in range(len(self.X)):            # number of design points
                for k in range(j+1, len(self.X)):   # for design points (j, k)
                    # note that bivariate measures are already negative log of distribution functions
                    neg_ell += self.random_process.bivariate_measure(self.X[j], self.X[k], self.y[j, i], self.y[k, i])
        return neg_ell
        
    def fit(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        
        res = minimize(self.neg_composite_log_likelihood, self.theta, method='L-BFGS-B', bounds=[(1e-3, 1e3) for _ in range(len(self.theta))])
        self.neg_ell = res.fun
        self.random_process.theta = res.x

    # simulate method is Algorith 1 in Dombry et al. (2016)
    def simulate(self, X_new):
        X_new = np.atleast_2d(X_new)
        N = len(X_new)
        s = stats.expon.rvs(size=1)
        w = self.random_process.sample(X_new, n=1)
        Z = w / s
        for i in range(1, N):
            s = stats.expon.rvs(size=1)
            while 1 / s > Z[i]:
                w = self.random_process.sample(X_new, n=1)
                Z_new = w / s
                if np.all([Z_new[j] < Z[j] for j in range(i)]):
                    Z = np.maximum(Z, Z_new)
                s += stats.expon.rvs(size=1)
        return Z
           
    @property
    def theta(self):
        return self.random_process.theta
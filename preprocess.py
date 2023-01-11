import numpy as np


class BlockMaxima:
    def __init__(self, block_size=20):
        self.block_size = block_size
    
    def get_maxima(self, y, return_y=True):
        self.y = y.copy()
        r = self.y.shape[1] % self.block_size
        if r != 0:
            y_ = np.pad(self.y, ((0, 0), (0, r)), constant_values=np.nan)
        else:
            y_ = y.copy()
        y_ = y_.reshape(y.shape[0], -1, self.block_size)
        self.y_maxima = np.nanmax(y_, axis=2)
        if return_y:
            return self.y_maxima
        else:
            pass

# more to include
# 1. shuffle (maybe jackknife or bootstrap)
# 2. discard deficient block

class FrechetScaler:    
    # Input: Non-unit Frechet data array, GEV parameters
    # Output: Unit Frechet data array
    def transform(self, y, params):
        self.y = y.copy()



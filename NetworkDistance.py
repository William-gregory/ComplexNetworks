import numpy as np

def Dist(strengthmap1,strengthmap2,seed=1):
  """
  This function computes a normalised network distance
  measure between two complex networks. The two inputs, 
  'strengthmap1' and 'strengthmap2' are assumed to be 
  both 2D arrays where each grid cell contains the 
  network strength value of the node in which that cell
  belongs. The function then returns a float value.
  """
  np.random.seed(seed)
    
  strengthmap1[np.isnan(strengthmap1)] = 0
  strengthmap2[np.isnan(strengthmap2)] = 0
    
  d_metric = np.abs(strengthmap1-strengthmap2).sum()

  shuffle1 = np.random.permutation(strengthmap1.ravel())
  shuffle2 = np.random.permutation(strengthmap2.ravel())
    
  d_metric_random = np.abs(shuffle1-shuffle2).sum()

  return 1 - d_metric/d_metric_random

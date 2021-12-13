from scipy.special import comb
import numpy as np


def ARI(network1,network2,dX=186,dY=186):
  """
  This function computes both the Rand Index (Rand, 1971)
  and the Adjusted Rand Index (Hubert and Arabie, 1985) from
  two different sets of network nodes.
  The two input networks, 'network1' and 'network2' are assumed
  to be dictionaries containing the information on which grid cells
  are contained within each network node (cluster).
  dX and dY are the grid dimensions used to calculate the total 
  number of grid cells.
  """
    def prep(nodes):
        clusters = []
        for area in nodes:
            temp_list = []
            for cell in nodes[area]:
                temp_list.append((cell[0]*dY)+cell[1])
            clusters.append(temp_list)
        
        IDs = [item for sublist in clusters for item in sublist]
        all_cells = np.arange(dX*dY)
        all_cells_bool = np.ones(dX*dY,dtype=bool)
        all_cells_bool[IDs] = False
        cells_not_in_N = all_cells[all_cells_bool==True]
        
        def chunks(l, n):
            for i in range(0, len(l), n):
                yield l[i:i+n]
        
        AC = chunks(cells_not_in_N.tolist(),20)
        for artificial_cluster in AC:
            clusters.append(artificial_cluster)
        
        return clusters
    
    clusters1 = prep(network1)
    clusters2 = prep(network2) 
    
    C = np.zeros((len(clusters1),len(clusters2)))
    ix = 0
    for cluster1 in clusters1:
        jx = 0
        for cluster2 in clusters2:
            for item in cluster1:
                if item in cluster2:
                    C[ix,jx] += 1
            jx += 1
        ix += 1                                                              
    a=comb(C,2).sum()
    b0=comb(C.sum(0),2).sum()
    b=b0 - a
    c0=comb(C.sum(1),2).sum()
    c=c0 - a
    S=comb(C.sum(),2)
    d=S-(a + b + c)
    RI=(a+d)/S
    AdjRI = (a-b0*c0/S)/((b0+c0)/2-b0*c0/S) 
    return RI,AdjRI

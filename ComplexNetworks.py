### Compute Complex Networks of geospatial time series data
### Author: William Gregory
### Last updated: 10/03/2021

import numpy as np
from scipy import stats
import itertools
import operator

class Network:
    def __init__(self,data,dimX=0,dimY=0,dimT=0,nodes={},corrs=[],tau=0,gridcells=[],unavail=[],anomaly={},links={},strength={},strengthmap=[]):
        """
        The input 'data' are expected to be de-trended (zero-mean)
        and in the format x,y,t if an area grid, or lat,lon,t for
        a lat-lon grid.
        """
        self.data = data
        self.dimX,self.dimY,self.dimT = self.data.shape
        self.nodes = nodes
        self.corrs = corrs
        self.tau = tau
        self.gridcells = gridcells
        self.unavail = unavail
        self.anomaly = anomaly
        self.links = links
        self.strength = strength
        self.strengthmap = strengthmap
    
    def get_threshold(self, significance=0.01):
        """
        Compute pairwise correlations between all grid cells.
        The average of all correlations which are positive and
        below a specified significance level will determine the
        threshold which is used to cluster cells to form network
        nodes in the function get_nodes().
        """
        ID = np.where(np.abs(np.nanmax(self.data,2))>0)
        N = np.shape(ID)[1]
        R = np.corrcoef(self.data[ID])
        np.fill_diagonal(R,np.nan)
        self.corrs = np.zeros((N,self.dimX,self.dimY))*np.nan
        self.gridcells = ID[0]*self.dimY + ID[1]
        for n in range(N):
            self.corrs[n,:,:][ID] = R[n,:]
        
        df = self.dimT - 2
        R = R[R>=0]
        T = R*np.sqrt(df/(1 - R**2))
        P = stats.t.sf(T,df)
        R = R[P<significance]

        self.tau = np.mean(R)
    
    def get_nodes(self, latlon=False):
        """
        cluster grid cells together to from nodes of the
        complex network. Clustering is based on a greedy
        algorithm, and the threshold for clustering two 
        grid cells together is defined by self.tau
        """
        ids = np.where(np.isnan(self.data[:,:,:]))
        i_nan = ids[0][0] ; j_nan = ids[1][0]

        def cell_neighbours(i, j, i_nan, j_nan):
            if [i-1,j] not in self.unavail:
                nei_1 = [i-1,j] if 0 <= j <= self.dimY-1 and 0 <= i-1 <= self.dimX-1 else [i_nan,j_nan]
            else:
                nei_1 = [i_nan,j_nan]
            if [i+1,j] not in self.unavail:
                nei_2 = [i+1,j] if 0 <= j <= self.dimY-1 and 0 <= i+1 <= self.dimX-1 else [i_nan,j_nan]
            else:
                nei_2 = [i_nan,j_nan]
            if ([i,j-1] not in self.unavail) & (latlon==False):
                nei_3 = [i,j-1] if 0 <= j-1 <= self.dimY-1 and 0 <= i <= self.dimX-1 else [i_nan,j_nan]
            elif ([i,j-1] not in self.unavail) & (latlon==True):
                nei_3 = [i,j-1] if 0 <= j-1 <= self.dimY-1 and 0 <= i <= self.dimX-1 else [i,self.dimY-1]
            elif [i,j-1] in self.unavail:
                nei_3 = [i_nan,j_nan]
            if ([i,j+1] not in self.unavail) & (latlon==False):
                nei_4 = [i,j+1] if 0 <= j+1 <= self.dimY-1 and 0 <= i <= self.dimX-1 else [i_nan,j_nan]
            elif ([i,j+1] not in self.unavail) & (latlon==True):
                nei_4 = [i,j+1] if 0 <= j+1 <= self.dimY-1 and 0 <= i <= self.dimX-1 else [i,0]
            elif [i,j+1] in self.unavail:
                nei_4 = [i_nan,j_nan]
            return [nei_1, nei_2, nei_3, nei_4]

        def area_neighbours(Area, i_nan, j_nan):
            neighbours = []
            for cell in Area:
                if [cell[0]-1,cell[1]] not in self.unavail:
                    neighbours.append([cell[0]-1,cell[1]] if 0 <= cell[1] <= self.dimY-1 and 0 <= cell[0]-1 <= self.dimX-1 else [i_nan,j_nan])
                else:
                    neighbours.append([i_nan,j_nan])
                if [cell[0]+1,cell[1]] not in self.unavail:
                    neighbours.append([cell[0]+1,cell[1]] if 0 <= cell[1] <= self.dimY-1 and 0 <= cell[0]+1 <= self.dimX-1 else [i_nan,j_nan])
                else:
                    neighbours.append([i_nan,j_nan])
                if ([cell[0],cell[1]-1] not in self.unavail) & (latlon==False):
                    neighbours.append([cell[0],cell[1]-1] if 0 <= cell[1]-1 <= self.dimY-1 and 0 <= cell[0] <= self.dimX-1 else [i_nan,j_nan])
                elif ([cell[0],cell[1]-1] not in self.unavail) & (latlon==True):
                    neighbours.append([cell[0],cell[1]-1] if 0 <= cell[1]-1 <= self.dimY-1 and 0 <= cell[0] <= self.dimX-1 else [cell[0],self.dimY-1])
                elif [cell[0],cell[1]-1] in self.unavail:
                    neighbours.append([i_nan,j_nan])
                if ([cell[0],cell[1]+1] not in self.unavail) & (latlon==False):
                    neighbours.append([cell[0],cell[1]+1] if 0 <= cell[1]+1 <= self.dimY-1 and 0 <= cell[0] <= self.dimX-1 else [i_nan,j_nan])
                elif ([cell[0],cell[1]+1] not in self.unavail) & (latlon==True):
                    neighbours.append([cell[0],cell[1]+1] if 0 <= cell[1]+1 <= self.dimY-1 and 0 <= cell[0] <= self.dimX-1 else [cell[0],0])
                elif [cell[0],cell[1]+1] in self.unavail:
                    neighbours.append([i_nan,j_nan])
            return neighbours

        def area_max_correlation(Area, neighbours):
            Rmean = [] ; X = []
            for cell in neighbours:
                R = []
                new_cell = cell[0]*self.dimY + cell[1]
                if new_cell in self.gridcells:
                    X.append(cell)
                    IDnew = np.where(self.gridcells == new_cell)
                    IDnew = int(IDnew[0])
                    for cells in Area:
                        if ([cells[0],cells[1]] != [cell[0],cell[1]]):
                            R.append(self.corrs[IDnew,cells[0],cells[1]])
                    Rmean.append(np.nanmean(R))
            try:
                Rmax = np.nanmax(Rmean)
            except ValueError:
                Rmax = np.nan
            return X, Rmean, Rmax

        #S T E P   1   (C R E A T E   A R E A S)

        self.nodes = {}
        self.unavail = []
        k = 0
        np.random.seed(2) 
        for i,j in itertools.product(range(self.dimX),range(self.dimY)):
            gcell = i*self.dimY + j
            if gcell in self.gridcells:
                ID = np.where(self.gridcells == gcell)
                ID = int(ID[0])
                if [i,j] not in self.unavail:
                    while True:
                        neighbours = cell_neighbours(i, j, i_nan, j_nan)
                        neighbour_corrs = [self.corrs[ID,neighbours[0][0],neighbours[0][1]],
                                           self.corrs[ID,neighbours[1][0],neighbours[1][1]],
                                           self.corrs[ID,neighbours[2][0],neighbours[2][1]],
                                           self.corrs[ID,neighbours[3][0],neighbours[3][1]]]
                        maxR = np.nanmax(neighbour_corrs)
                        if maxR > self.tau:
                            maxID = np.where(neighbour_corrs==maxR)
                            if np.shape(maxID)[1] == 1:
                                maxID = int(maxID[0])
                            else:
                                maxID = int(maxID[0][np.random.randint(low=0,high=np.shape(maxID)[1])])
                            maxID = neighbours[maxID]
                            if ([i,j] not in self.unavail) and ([maxID[0],maxID[1]] not in self.unavail):
                                self.nodes.setdefault(k, []).append([i,j])
                                self.nodes.setdefault(k, []).append([maxID[0],maxID[1]])
                                self.unavail.append([i,j])
                                self.unavail.append([maxID[0],maxID[1]])

                                while True: #expand
                                    neighbours = area_neighbours(self.nodes[k], i_nan, j_nan)
                                    X, Rmean, Rmax = area_max_correlation(Area=self.nodes[k], neighbours=neighbours)
                                    if Rmax > self.tau:
                                        RmaxID = np.where(Rmean==Rmax)
                                        if np.shape(RmaxID)[1] == 1:
                                            RmaxID = int(RmaxID[0])
                                        else:
                                            RmaxID = int(RmaxID[0][np.random.randint(low=0,high=np.shape(RmaxID)[1])])
                                        m = X[RmaxID]
                                        if m not in self.unavail:
                                            self.nodes.setdefault(k, []).append([m[0],m[1]])
                                            self.unavail.append([m[0],m[1]])
                                        else:
                                            break
                                    else:
                                        break
                                k = k + 1
                            else:
                                break
                        else:
                            break
        
        #S T E P   2   (M I N I M I S E   NO.   O F   A R E A S)
        
        self.unavail = []
        while True:
            Rs = {}
            unavail_neighbours = {}
            num_cells = dict([(area,len(self.nodes[area])) if self.nodes[area] not in self.unavail else (area,0) for area in self.nodes.keys()])
            maxID = max(num_cells.items(), key=operator.itemgetter(1))[0]
            if num_cells[maxID] == 0:
                break
            else:
                neighbours = area_neighbours(self.nodes[maxID], i_nan, j_nan)
                for cell in neighbours:
                    gcell = cell[0]*self.dimY + cell[1]
                    Rmean = []                   
                    if (gcell in self.gridcells) & (cell not in self.nodes[maxID]) & (cell not in [k for k, g in itertools.groupby(sorted(itertools.chain(*unavail_neighbours.values())))]) & (len([area for area, cells in self.nodes.items() if cell in cells]) > 0):
                        nID = [area for area, cells in self.nodes.items() if cell in cells][0]
                        unavail_neighbours[nID] = self.nodes[nID]
                        X, Rmean, Rmax = area_max_correlation(Area=self.nodes[nID]+self.nodes[maxID], neighbours=self.nodes[nID]+self.nodes[maxID])
                        if nID not in Rs: 
                            Rs[nID] = np.nanmean(Rmean)
                try:
                    Rs_maxID = max(Rs.items(), key=operator.itemgetter(1))[0]
                    if Rs[Rs_maxID] > self.tau:
                        for cell in self.nodes.pop(Rs_maxID, None):
                            self.nodes.setdefault(maxID, []).append([cell[0],cell[1]])
                    else:
                        self.unavail.append(self.nodes[maxID])
                except ValueError:
                    self.unavail.append(self.nodes[maxID])
        
        
    def get_links(self, area=None, lat=None):
        """
        compute the anomaly time series associated with
        every node of the network, and subsequently compute
        weighted links (based on covariance) between all of
        these nodes. The strength of each node (also known as
        the weighted degree), is defined as the sum of the
        absolute value of each nodes links. Here the network
        is fully connected, so every node connects to every other
        node
        """
        self.anomaly = {}
        self.links = {}
        self.strength = {}
        self.strengthmap = np.zeros((self.dimX,self.dimY))*np.nan
        if lat is not None:
            scale = np.sqrt(np.cos(np.radians(lat)))
        elif area is not None:
            scale = np.sqrt(area)
        else:
            scale = np.ones((self.dimX,self.dimY))
            
        for A in self.nodes:
            temp_array = np.zeros(self.data.shape)*np.nan
            for cell in self.nodes[A]:
                temp_array[cell[0],cell[1],:] = np.multiply(self.data[cell[0],cell[1],:],scale[cell[0],cell[1]])
            self.anomaly[A] = np.nansum(temp_array, axis=(0,1))
            
        for A in self.anomaly:
            sdA = np.std(self.anomaly[A])
            for A2 in self.anomaly:
                sdA2 = np.std(self.anomaly[A2])
                if A2 != A:
                    self.links.setdefault(A, []).append(stats.pearsonr(self.anomaly[A],self.anomaly[A2])[0]*(sdA*sdA2))
                elif A2 == A:
                    self.links.setdefault(A, []).append(0)
            
        for A in self.links:
            absolute_links = []  
            for link in self.links[A]:
                absolute_links.append(abs(link))
            self.strength[A] = np.nansum(absolute_links)
            for cell in self.nodes[A]:
                self.strengthmap[cell[0],cell[1]] = self.strength[A]

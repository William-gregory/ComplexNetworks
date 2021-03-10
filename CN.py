### Compute Complex Networks of geospatial time series data
### Author: William Gregory
### Last updated: 10/03/2021

import numpy as np
from scipy import stats
import itertools
import operator

class Network:
    def __init__(self,data,dimX=0,dimY=0,dimT=0,V={},corrs=[],tau=0,nodes=[],unavail=[],anomaly={},links={},strength={},strengthmap=[]):
        """
        The input 'data' are expected to be de-trended (zero-mean)
        and in the format x,y,t if an area grid, or lat,lon,t for
        a lat-lon grid.
        """
        self.data = data
        self.dimX,self.dimY,self.dimT = self.data.shape
        self.V = V
        self.corrs = corrs
        self.tau = tau
        self.nodes = nodes
        self.unavail = unavail
        self.anomaly = anomaly
        self.links = links
        self.strength = strength
        self.strengthmap = strengthmap
    
    def tau(self, significance=0.01):
        """
        Compute pairwise correlations between all grid cells.
        The average of all correlations which are positive and
        below a specified significance level will determine the
        threshold which is used to cluster cells to form network
        nodes in the function area_level().
        """
        ID = np.where(np.abs(np.nanmax(self.data,2))>0)
        N = np.shape(ID)[1]
        R = np.corrcoef(self.data[ID])
        self.corrs = np.zeros((N,self.dimX,self.dimY))*np.nan
        self.nodes = ID[0]*self.dimY + ID[1]
        for n in range(N):
            self.corrs[n,:,:][ID] = R[n,:]
        
        df = self.dimT - 2
        R = R[R>=0]
        T = R*np.sqrt(df/(1 - R**2))
        P = stats.t.sf(T,df)
        R = R[P<significance]

        self.tau = np.mean(R)
    
    def area_level(self, data, latlon_grid=False):
        ids = np.where(np.isnan(data[:,:,:]))
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
            if ([i,j-1] not in self.unavail) & (latlon_grid==False):
                nei_3 = [i,j-1] if 0 <= j-1 <= self.dimY-1 and 0 <= i <= self.dimX-1 else [i_nan,j_nan]
            elif ([i,j-1] not in self.unavail) & (latlon_grid==True):
                nei_3 = [i,j-1] if 0 <= j-1 <= self.dimY-1 and 0 <= i <= self.dimX-1 else [i,self.dimY-1]
            elif [i,j-1] in self.unavail:
                nei_3 = [i_nan,j_nan]
            if ([i,j+1] not in self.unavail) & (latlon_grid==False):
                nei_4 = [i,j+1] if 0 <= j+1 <= self.dimY-1 and 0 <= i <= self.dimX-1 else [i_nan,j_nan]
            elif ([i,j+1] not in self.unavail) & (latlon_grid==True):
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
                if ([cell[0],cell[1]-1] not in self.unavail) & (latlon_grid==False):
                    neighbours.append([cell[0],cell[1]-1] if 0 <= cell[1]-1 <= self.dimY-1 and 0 <= cell[0] <= self.dimX-1 else [i_nan,j_nan])
                elif ([cell[0],cell[1]-1] not in self.unavail) & (latlon_grid==True):
                    neighbours.append([cell[0],cell[1]-1] if 0 <= cell[1]-1 <= self.dimY-1 and 0 <= cell[0] <= self.dimX-1 else [cell[0],self.dimY-1])
                elif [cell[0],cell[1]-1] in self.unavail:
                    neighbours.append([i_nan,j_nan])
                if ([cell[0],cell[1]+1] not in self.unavail) & (latlon_grid==False):
                    neighbours.append([cell[0],cell[1]+1] if 0 <= cell[1]+1 <= self.dimY-1 and 0 <= cell[0] <= self.dimX-1 else [i_nan,j_nan])
                elif ([cell[0],cell[1]+1] not in self.unavail) & (latlon_grid==True):
                    neighbours.append([cell[0],cell[1]+1] if 0 <= cell[1]+1 <= self.dimY-1 and 0 <= cell[0] <= self.dimX-1 else [cell[0],0])
                elif [cell[0],cell[1]+1] in self.unavail:
                    neighbours.append([i_nan,j_nan])
            return neighbours

        def area_max_correlation(Area, neighbours):
            Rmean = [] ; X = []
            for cell in neighbours:
                R = []
                new_node = cell[0]*self.dimY + cell[1]
                if new_node in self.nodes:
                    X.append(cell)
                    IDnew = np.where(self.nodes == new_node)
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

        self.V = {}
        self.unavail = []
        k = 0
        np.random.seed(2)
        print('Creating area-level network')  
        print(datetime.datetime.now())
        for i,j in itertools.product(range(self.dimX),range(self.dimY)):
            node = i*self.dimY + j
            if node in self.nodes:
                ID = np.where(self.nodes == node)
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
                                self.V.setdefault(k, []).append([i,j])
                                self.V.setdefault(k, []).append([maxID[0],maxID[1]])
                                self.unavail.append([i,j])
                                self.unavail.append([maxID[0],maxID[1]])

                                while True: #expand
                                    neighbours = area_neighbours(self.V[k], i_nan, j_nan)
                                    X, Rmean, Rmax = area_max_correlation(Area=self.V[k], neighbours=neighbours)
                                    if Rmax > self.tau:
                                        RmaxID = np.where(Rmean==Rmax)
                                        if np.shape(RmaxID)[1] == 1:
                                            RmaxID = int(RmaxID[0])
                                        else:
                                            RmaxID = int(RmaxID[0][np.random.randint(low=0,high=np.shape(RmaxID)[1])])
                                        m = X[RmaxID]
                                        if m not in self.unavail:
                                            self.V.setdefault(k, []).append([m[0],m[1]])
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

        self.unavail = []
        while True:
            num_cells = {}
            Anei_Rs = {}
            unavail_neis = []
            #Identify largest area in terms of number of cells
            for k in self.V:
                if self.V[k][0] not in self.unavail:
                    num_cells.setdefault(k, []).append(np.shape(self.V[k])[0])
                else:
                    num_cells.setdefault(k, []).append(0)
            max_ID = max(num_cells.items(), key=operator.itemgetter(1))[0]
            if num_cells[max_ID][0] == 0:
                break
            else:
                #print('AreaID = ',max_ID, ', # of cells = ',len(self.V[max_ID]))
                for X in self.V[max_ID]: #for each cell in the currently available largest area
                    nei_1, nei_2, nei_3, nei_4 = cell_neighbours(X[0],X[1], i_nan, j_nan) #generate the cell's available neighbours
                    nei_list = [nei_1, nei_2, nei_3, nei_4]
                    for k in self.V: #search through all other areas in the network   
                        for nei in nei_list: #search through each neighbour of the current cell in largest area
                            R_mean = []
                            if (nei not in self.V[max_ID]) & (nei in self.V[k]) & (nei not in unavail_neis): #if the neighbouring cell belongs to a neighbouring AREA, and is available
                                #print('nei = ',nei,'is in Area ',k,'and is not in Area',max_ID)
                                #print('Area',k,' = ',self.V[k])
                                for i in range(np.shape(self.V[k])[0]):
                                    unavail_neis.append(self.V[k][i])
                                #here make a hypothetical area of the largest area (max_ID) and it's available neighbour (k) to check average correlation    
                                hypoth_area = []
                                for cell in self.V[max_ID]:
                                    hypoth_area.append([cell[0],cell[1]])
                                for cell in self.V[k]:
                                    hypoth_area.append([cell[0],cell[1]])
                                NA_list = []
                                for cell in hypoth_area:
                                    #print(cell)
                                    R = []
                                    ID = np.where(self.nodes == (cell[0]*self.dimY)+cell[1])
                                    ID = int(ID[0])
                                    for a in range(np.shape(hypoth_area)[0]):
                                        b = int(hypoth_area[a][0])
                                        c = int(hypoth_area[a][1])
                                        if ([b,c] != [cell[0],cell[1]]) & ([b,c] not in NA_list):
                                            #print('[',b,',',c,']')
                                            R.append(self.corrs[ID,b,c])
                                    NA_list.append([cell[0],cell[1]])
                                    R_mean.append(np.nanmean(R))   
                                if k not in Anei_Rs: 
                                    Anei_Rs.setdefault(k, []).append(np.nanmean(R_mean))
                                #print('Average correlation with Area',max_ID,'and neighbouring Area',k,' = ',Anei_Rs[k])
                try:
                    Anei_Rs_max_ID = max(Anei_Rs.items(), key=operator.itemgetter(1))[0]
                    #print('Maximum correlation with neighbouring area = ',Anei_Rs[Anei_Rs_max_ID][0])
                    if Anei_Rs[Anei_Rs_max_ID][0] > self.tau:
                        #print('ID_pair = ',Anei_Rs_max_ID)
                        temp2 = self.V.pop(Anei_Rs_max_ID, None)
                        for i in temp2:
                            self.V.setdefault(max_ID, []).append([i[0],i[1]])
                    else:
                        for i in range(np.shape(self.V[max_ID])[0]):
                            self.unavail.append(self.V[max_ID][i])
                except ValueError:
                    for i in range(np.shape(self.V[max_ID])[0]):
                        self.unavail.append(self.V[max_ID][i])

        print('Done!')
                
    def intra_links(self, data, area=None, lat=None):
        print('Generating network links')
        self.anomaly = {}
        self.links = {}
        self.strength = {}
        if lat is not None:
            scale = np.sqrt(np.cos(np.radians(lat)))
        elif area is not None:
            scale = np.sqrt(area)
        else:
            scale = np.ones((self.dimX,self.dimY))
            
        for A in self.V:
            temp_array = np.zeros((data.shape))*np.nan
            for cell in self.V[A]:
                temp_array[cell[0],cell[1],:] = np.multiply(data[cell[0],cell[1],:],scale[cell[0],cell[1]])
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
            strength[A] = np.nansum(absolute_links)

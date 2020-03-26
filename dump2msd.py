import numpy as np
import time
from mpi4py import MPI

def d_pbc(vector1,vector2,boxlength):
    l_ref=[boxlength[0]/2.0,boxlength[1]/2.0]
    vector=vector1-vector2

    if vector[0]<(-1)*l_ref[0]:
        vector[0]=vector[0]+boxlength[0]
    elif vector[0]>l_ref[0]:
        vector[0]=vector[0]-boxlength[0]
    else:
        vector[0]=vector[0]

    if vector[1]<(-1)*l_ref[1]:
        vector[1]=vector[1]+boxlength[1]
    elif vector[1]>l_ref[1]:
        vector[1]=vector[1]-boxlength[1]
    else:
        vector[1]=vector[1]

    return vector

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

N = 47520 #number of beads/atoms of interest (from atom 1 to atom N / atom 0 to atom N-1)
subs = 0 #1 if for subsequent simulation
f_ind = np.linspace(0, 990000, 100, dtype=int) #file index
avg_rows_per_process = int(len(f_ind)/size)

if rank == 0:
    t0 = time.time()

    box = []

    infile = 'dump/dump.0'
    f = open(infile, 'r')
    for iind in range(5):
        f.readline()
    for iind in range(3):
        line = f.readline().split()
        box.append([float(line[0]), float(line[1])])
    f.close()

    trj = np.loadtxt(infile, skiprows=9)
    trj = trj[np.lexsort(np.fliplr(trj).T)][:N,:]
 
    morph = np.load('morph.npy', allow_pickle='TRUE').item()
    
    phase0 = np.concatenate([morph['mol'][i][0] for i in morph['mol']])
    #phase1 = np.concatenate([morph['mol'][i][1] for i in morph['mol']])
    mol_of_interest = phase0
    #mol_of_interest = morph['mol'][0][0]

    for jind in range(len(mol_of_interest)):
        if jind == 0:
            logic = trj[:,1] == mol_of_interest[jind]
        else:
            logic = logic | (trj[:,1] == mol_of_interest[jind])
    
    msd_dim = [3,4] # [3,4,5] for msd in 3d space; [3,4] for msd in xy-plane; [5] for msd along z direction
    msd_denom = sum(logic)

else:
    box = None
    logic = None
    msd_dim = None
    msd_denom = None

box = comm.bcast(box, root=0)
logic = comm.bcast(logic, root=0)
msd_dim = comm.bcast(msd_dim, root=0)
msd_denom = comm.bcast(msd_denom, root=0)

dis = {}

start_row = rank * avg_rows_per_process
end_row = start_row + avg_rows_per_process

end_ind = end_row + 1
if rank == size-1:
    end_row = len(f_ind)
    end_ind = end_row

for iind in range(start_row, end_ind):
    infile = 'dump/dump.'+str(f_ind[iind])
    trj = np.loadtxt(infile, skiprows=9)
    trj = trj[np.lexsort(np.fliplr(trj).T)][:N,:]
        
    dis_temp = np.zeros(trj.shape)
    dis_temp[:,:3] = trj[:,:3]

    if (iind == 0 and subs == 1): #subs = 1 if for subsequent simulation
        infile = '../dis_end.txt'
        trj_temp = np.loadtxt(infile)
        dis_temp[:,3:] = trj_temp[:,1:]

    if iind > start_row: #start_row for rank of 0 is equal to 0
        for jind in range(trj.shape[0]):
            #displacement vector over pbc
            dis_temp[jind,3:] = d_pbc(trj[jind,3:],trj_prev[jind,3:],[box[0][1]-box[0][0], box[1][1]-box[1][0]])

        dis_temp[:,3:] += dis[iind-1][:,3:]

    dis.update({iind : dis_temp})
    trj_prev = trj

'calculate displacements'
if rank == 0:
    #even though rank 0 processor processes from dis[0=start_row] to dis[end_row-1],
    #dis[end_row] works as a bridge b/w ranks 0 and 1, and has to be sent to rank 1
    data = dis[end_row] 
    if size > 1:
        req = comm.Isend(data, dest=(rank+1))
        req.Wait()

elif rank == size-1:
    data = np.empty(trj.shape)
    req = comm.Irecv(data, source=(rank-1))
    req.Wait()

    for iind in range(start_row, end_row):
        dis[iind][:,3:] += data[:,3:]

    #dis[end_row-1] is the very last displacement, which is necessary to calculates displacements for the subsequent simulation
    np.savetxt('dis_end.txt',dis[end_row-1][:,[0,3,4,5]]) 

else:
    data = np.empty(trj.shape)
    req = comm.Irecv(data, source=(rank-1))
    req.Wait()

    #even though rank n processor processes from dis[start_row] to dis[end_row-1],
    #dis[end_row] works as a bridge b/w ranks n and n+1, and has to be sent to rank n+1
    for iind in range(start_row, end_row+1): 
        dis[iind][:,3:] += data[:,3:]

    data = dis[end_row] #dis[end_row] works as a bridge b/w adjacent processors
    req = comm.Isend(data, dest=(rank+1))
    req.Wait()

'calculates and collect msd from all procs'
if rank == 0:
    t1 = time.time() - t0
    print (t1)

    msd = np.zeros(end_row - start_row)
    for iind in range(start_row, end_row):
        msd[iind - start_row] = sum(np.linalg.norm(dis[iind][logic][:,msd_dim], axis=1)**2)/msd_denom

    for iind in range(1, size):
        start_row = iind * avg_rows_per_process
        end_row = start_row + avg_rows_per_process
        if iind == size-1:
            end_row = len(f_ind)

        msd_temp = np.empty(end_row - start_row)
        req = comm.Irecv(msd_temp, source=iind)
        req.Wait()

        msd = np.concatenate((msd, msd_temp))

else:
    msd_temp = np.zeros(end_row - start_row)
    for iind in range(start_row, end_row):
        msd_temp[iind - start_row] = sum(np.linalg.norm(dis[iind][logic][:,msd_dim], axis=1)**2)/msd_denom

    req = comm.Isend(msd_temp, dest=0)
    req.Wait()

'save msd'
if rank == 0:
    t2 = time.time() - t0
    print (t2)

    np.savetxt('msd_test.txt', msd)

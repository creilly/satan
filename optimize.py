from saturation import sanitize
from mpi4py import MPI
import numpy as np
from time import perf_counter
from saturation import rabi, gamma, doppler, beam, bloch
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution
import noisyopt
import os

debug = False
# number of monte carlo samples per quantum state probability calculation (qspc)
N = 400

# model parameter computation
beam_diameter = 4.0e-3 # meters
noise_amplitude = 30. # millivolts rms
velocity = 1000. # meters per second

gamma = gamma.get_gamma(noise_amplitude) * 1e-6 # rads / us
tau = beam.get_tau(beam_diameter,velocity) * 1e6 # us
sigmaomega = doppler.get_deltaomega(velocity) * 1e-6 # rads / us

# degeneracy multiplier matrix initialization
jmax = 10
degens = np.array([2 if m else 1 for m in range(jmax + 1)])

# matrix prototypes
M_deltaomega = [
    [   0,     -1,    0   ],
    [  +1,      0,    0   ],
    [   0,      0,    0   ]
]

M_gamma = [
    [  -1,      0,    0   ],
    [   0,     -1,    0   ],
    [   0,      0,    0   ]
]

M_omegabar = [
    [   0,      0,    0   ],
    [   0,      0,   +1   ],
    [   0,     -1,    0   ]
]

M_deltaomega, M_gamma, M_omegabar = map(np.array,(M_deltaomega,M_gamma,M_omegabar))

# MPI initialization
comm = MPI.COMM_WORLD
size = comm.Get_size() # number of cores
rank = comm.Get_rank() # core id
masterrank = size - 1 # core id of master core

md = sanitize.load_metadata() # dataset metadata
cgcd = sanitize.load_cgc() # pre-computed clebsch gordan coefficients

# total number of quantum states to compute
def get_total_length():
    total_length = 0
    for lined in md:
        j1 = lined['j1']
        for mode in (sanitize.FC,sanitize.FS):
            total_length += lined['lengths'][str(mode)] * (j1+1)
    return total_length

END = -1 # marks end of job division transmission
sendrequests = [] # stores asynchronous send requests
recvrequests = [] # stores asynchronous recv requests
data = [] # stores unchanging model data

MEASURED, COMPUTED = 0, 1
arrays = [] # stores job division information
dataarrays = {} # stores computed and measured scientific data
indexbuffer = np.empty(4,dtype=np.int64) # stores job division tranmission data

# synchronous buffer send (thin wrapper)
def send(buffer,dest):
    comm.Send(buffer,dest)
# asynchronous buffer send 
# sendrequests needed to prevent garbage collection of asychronous requests
def isend(buffer,dest):
    sendrequests.append(comm.Isend(buffer,dest))
# synchronous (blocking) recv
def recv(buffer,source):
    comm.Recv(buffer,source)
# asynchronous (nonblocking) recv
def irecv(buffer,source,jobindex):    
    recvrequests[jobindex][source] = comm.Irecv(buffer,source)
# clear send request list
def clear_sends():
    while sendrequests:
        sendrequests.pop(0).Wait()    
# complete all pending recvs
def wait_recvs():
    for jobd in recvrequests:
        for request in jobd.values():
            request.Wait()    

# read data from disk
# compute unchanging model data
# transmit measured data to masterrank
def get_data():
    total_length = get_total_length() # total num of q state prob calcs (qspcs)
    chunk = total_length // size # total num of qspcs for each rank
    rankmarker = 0 # num of qspcs for current chunk 
    _rank = 0 # rank of current chunk
    # loop through datasets, determine job load 
    for lineindex, lined in enumerate(md):        
        j1 = lined['j1']
        n_mu = j1 + 1 # qspcs per point on curve (fc or fs)
        j2 = lined['j2']
        w = lined['wavenumber'][0] # cm-1
        a = lined['einstein coefficient'][0] # s-1
        mubar = rabi.mubar(w,a) # prop. coeff. relating mu to cgc
        # compute mus, pulling cgcs from memory
        mus = [
            cgcd[j1][j2][m] * mubar for m in range(n_mu)
        ]
        for mode in (sanitize.FC,sanitize.FS):
            modelength = lined['lengths'][str(mode)] # num data points in curve
            modemarker = 0 # current position along curve
            while True:                
                modetail = modelength - modemarker # points remaining in curve
                ranktail = chunk - rankmarker # qspcs left in chunk                
                # if qspcs left in curve is greater than qspcs left in chunk:
                #   advance modemarker to fill chunk (rounding up)
                # else:
                #   advance modemarker to end of curve
                tail = (
                    ranktail // n_mu + (1 if ranktail % n_mu else 0)
                ) if modetail * n_mu > ranktail else modetail
                # if current rank is equal to core id
                if rank == _rank:  
                    # pull current segment of curve data from disk                  
                    # deltaomegas: rads / us
                    # powers: watts
                    # zs (detector signals): normed
                    # shape (of each) : (tail,)
                    deltaomegas, powers, zs = sanitize.load_data(lineindex,mode)[modemarker:modemarker+tail].transpose()
                    # compute rabi frequencies (omegabar) for each qspc of curve segment
                    # shape : (n_mu,tail)
                    omegabars = np.outer(mus,1e-6 * 2 * np.pi * rabi.elec_field(powers/(np.pi * (beam_diameter/2)**2)) / rabi.h) # rads / us (n_mu, n_points)
                    # scale omegabar matrix term by all omegabars for each qspc of curve segment
                    # shape : (n_mu,tail,3,3)
                    M_omegabars = np.einsum('ij,kl->ijkl',omegabars,M_omegabar)
                    # save deltaomega (frequency detuning) and omegabar data for this curve segment to local memory
                    data.append(
                        (
                            deltaomegas,M_omegabars
                        )
                    ) 

                    # send curve segment metadata to masterrank
                    start = modemarker
                    stop = modemarker + tail
                    isend(np.array((lineindex,mode,start,stop),dtype=np.int64),masterrank)
                    # send curve segment measured data to masterrank
                    isend(np.ascontiguousarray(zs),masterrank)
                    if debug:
                        print(
                            '*','\t\t'.join(
                                '{}: {:d}'.format(label,num) for label, num in (
                                    ('rank',_rank),
                                    ('line',lineindex),
                                    ('mode',mode),
                                    ('start',start),
                                    ('stop',stop)
                                )
                            )                
                        )  
                # advance chunk marker                  
                rankmarker += tail * n_mu
                # if current chunk is larger than chunk size
                if chunk - rankmarker <  n_mu:
                    # if current rank is core id
                    if rank == _rank:
                        # signal end of job division info transmission
                        isend(np.array([END]*4),masterrank)
                    # reset chunk marker
                    rankmarker = 0
                    # increment rank marker
                    _rank += 1             
                    # if previous rank was core id       
                    if rank < _rank:
                        # exit
                        return
                # advance curve marker
                modemarker += tail
                # if curve marker is at end of curve
                if modemarker == modelength:
                    # go to next curve
                    break   
    # last rank will typically not have full chunk of qspcs
    # signal end of job division info transmission for this edge case
    # (that's why we make it the masterrank)
    isend(np.array([END]*4),masterrank)
# organize masterrank data arrays
# collect transmitted job division information
def finish_get_data():
    if rank == masterrank:  
        # loop through lines (each line has two curves, (FC and FS))
        for lineindex, lined in enumerate(md):
            # create dict to hold data for the FS and FC modes of this line
            linearrayd = {}
            dataarrays[lineindex] = linearrayd
            # loop through curve modes
            for mode in (sanitize.FC,sanitize.FS):
                # pull curve length from metadata
                modelength = lined['lengths'][str(mode)]
                # allocate memory for measured and computed data
                linearrayd[mode] = {
                    key:arr for key, arr in zip(
                        (MEASURED,COMPUTED),
                        np.empty(2*modelength).reshape((2,modelength))
                    )
                }
        # loop through cores
        for _rank in range(size):
            # create job division list for current core
            rankarrays = []            
            arrays.append(rankarrays)
            # loop through each curve segment for current core
            jobindex = 0  
            while True:
                # get curve segment metadata
                recv(indexbuffer,_rank)
                lineindex, mode, start, stop = indexbuffer
                if debug:
                    print(
                        '\t\t'.join(
                            '{}: {:d}'.format(label,num) for label, num in (
                                ('rank',_rank),
                                ('line',lineindex),
                                ('mode',mode),
                                ('start',start),
                                ('stop',stop)
                            )
                        )                
                    )
                # if end of transmission signalled, move on to next core
                if lineindex == END:
                    break            
                # if current core has more curve segments than all previous cores,
                # append a dict to recv request list for this curve segment index
                if jobindex == len(recvrequests):
                    jobd = {}
                    recvrequests.append(jobd)
                # get allocated data for the full curve of this curve segment
                moded = dataarrays[lineindex][mode]
                # initiate data transfer to fill the appropriate segment of the 
                # full measured curve buffer with measured data transmitted by current core                
                irecv(moded[MEASURED][start:stop],_rank,jobindex)
                # put pointer to appropriate segment of full computed curve buffer
                # at appropriate location of job division progression
                rankarrays.append(moded[COMPUTED][start:stop])
                jobindex += 1
        # finish data transfers
        wait_recvs()
    # all cores clear their send requests
    clear_sends()

# model fudge factors (to be fitted)
factors = np.array([1.2,0.25,0.8,0.3])
factors[:] = [0.966, 0.400, 0.613, 0.400]

# compute simulated curves for all curves
def get_outdata():
    # adjust model parameters with fudge factors
    power_factor, gamma_factor, sigmaomega_factor, tau_factor = factors
    sigmaomegap = sigmaomega_factor * sigmaomega
    taup = tau_factor * tau
    gammap = gamma_factor * gamma
    # initiate data transfers of updated computed curves
    if rank == masterrank:        
        for _rank, rankarrays in enumerate(arrays):
            for jobindex, arr in enumerate(rankarrays):
                irecv(arr,_rank,jobindex)
    # loop over curve segments of this core
    for deltaomegas, M_omegabars in data:
        # n_mu : quantum states per curve point
        # n_points : num points in curve segment        
        n_mu, n_points, *_ = M_omegabars.shape
        # compute N random frequency detunings (rfd) for each qspc
        # shape : (N,n_mu,n_points)
        d_deltaomegas = bloch.get_deltaomegas_geometric(
            0.0,sigmaomegap,N*n_mu*n_points
        ).reshape((N,n_mu,n_points))
        # offset rfd by known frequency detuning for each curve point
        # shape : (N,n_mu,n_points)
        deltaomegasp = d_deltaomegas + deltaomegas
        # scale deltaomega matrix term by each offset rfd
        # shape : (N,n_mu,n_points,3,3)
        M_deltaomegasp = np.einsum('ijk,lm->ijklm',deltaomegasp,M_deltaomega)
        # assemble dynamical matrix for each rfd
        # shape : (N,n_mu,n_points,3,3)
        Ms = (gammap * M_gamma + power_factor * M_omegabars) + M_deltaomegasp
        # compute eigenvalues (lambdas) and eigenvectors (Ss) for each rfd
        # shape (lambdas) : (N,n_mu,n_points,3)
        # shape (Ss) : (N,n_mu,n_points,3,3)
        lambdas, Ss = np.linalg.eig(Ms)
        # compute inverse of eigenvector matrix for each rfd
        # shape : (N,n_mu,n_points,3,3)
        Sinvs = np.linalg.inv(Ss)        
        # compute random transit time (rtt) for each rfd
        # shape : (N,n_mu,n_points)
        tausp = bloch.get_taus(taup,N*n_mu*n_points).reshape((N,n_mu,n_points))
        # for each rtt:
        #   take product with each associated eigenvalue
        #   exponentiate
        # shape : (N,n_mu,n_points,3)
        exponents = np.exp(np.einsum('ijk,ijkl->ijkl',tausp,lambdas))
        # take last row of each eigenvector matrix
        # shape : (N,n_mu,n_points,3)
        Szs = Ss[:,:,:,2,:]
        # take last col of each inverse eigenvector matrix
        # shape : (N,n_mu,n_points,3)
        Sinvzs = Sinvs[:,:,:,:,2]        
        #   1. for each rfd
        #       i. take product of:
        #           a) eigenvector matrix row
        #           b) inverse eigenvector matrix col
        #           c) associated eigenvalues
        #           [a), b), c) are all length 3 arrays]
        #       ii. sum the products obtained in step i
        #   2. for each qspc
        #       i.  average sum obtained in 1.ii over random samples
        #       ii. scale and shift averages obtained step i
        # shape : (n_mu,n_points)
        probs = 0.5 + 0.5 * np.real(
            (
                -Szs*Sinvzs*exponents
            ).sum(3)
        ).sum(0) / N
        # get degeneracy factors for this line
        # shape : (n_mu,)
        modedegens = degens[:n_mu]        
        #   1. scale probabilities for each qspc by appropriate degeneracy factor, 
        #   2. for each curve point, sum over quantum states
        #   3. divide probability of each curve point by sum of degeneracy factors
        # shape : (n_points,)
        probs = np.einsum('ij,i->j',probs,modedegens) / (2*n_mu - 1)
        # send computed data to masterrank
        isend(probs,masterrank)    
    if rank == masterrank:
        # collect computed data
        wait_recvs()
        # intialitize sum squared error
        sse = 0.
        # loop over curves
        for lineindex, lined in dataarrays.items():
            for mode, moded in lined.items():
                # get measured curve data
                measured = moded[MEASURED]
                # get freshly computed curve data
                computed = moded[COMPUTED]
                # normalize computed data 
                computed /= computed.sum()
                # compute sum squared error, add to running total
                sse += ((measured - computed)**2).sum()    
    else:
        sse = None
    clear_sends()
    return sse

def optimize():
    if rank == masterrank:
        res = noisyopt.minimizeSPSA(
            _optimize,            
            factors,
            a = 0.01,
            c = 0.01,
            bounds = [(0.4,2.0)] * len(factors),
            paired=False,
            niter = 1000            
        )
        # res = differential_evolution(
        #     _optimize,
        #     bounds=[(0.05,20.0)]*len(factors),
        #     maxiter=100,
        #     popsize=10,
        #     disp=True
        # )
        print('results:',res)        
        stop_factors = -np.ones(len(factors))
        for _rank in range(size):
            if _rank == masterrank:
                continue
            send(stop_factors,_rank)
        return res
    else:
        while True:
            recv(factors,masterrank)
            if factors[0] < 0:
                break        
            get_outdata()        

def _optimize(_factors):        
    start = perf_counter()   
    factors[:] = _factors
    for _rank in range(size):
        if _rank == masterrank:
            continue
        send(factors,_rank)    
    sse = get_outdata()
    stop = perf_counter()    
    print(
        'iter time:','{:.2f}'.format(stop-start),'s',
        '|',
        'factors:',', '.join('{:.3f}'.format(factor) for factor in factors),
        '|',
        'sse:','{:.2e}'.format(sse)
    )
    return sse    

outfile = 'results.txt'
def main():
    get_data()
    finish_get_data()
    results = optimize()
    if rank == masterrank:
        with open(outfile,'w') as f:
            f.write(str(results))

imagefolder = 'images'
def save_data():
    get_data()
    finish_get_data()
    get_outdata()
    if rank == masterrank:
        for lineindex, lined in dataarrays.items():
            for mode, moded in lined.items():
                deltaomegas, powers, _ = sanitize.load_data(lineindex,mode).transpose()
                measureds = moded[MEASURED]
                computeds = moded[COMPUTED]                      
                xs, yms, ycs = zip(
                    *sorted(
                        zip(
                            {sanitize.FC:powers,sanitize.FS:deltaomegas/(2.*np.pi)}[mode],
                            measureds,
                            computeds
                        )
                    )
                )          
                plt.plot(xs,yms,'.')
                plt.plot(xs,ycs)
                plt.title(
                    'line {:d} {}'.format(
                        lineindex,{
                            sanitize.FC:'fluence curve',
                            sanitize.FS:'frequency scan'
                        }[mode]
                    )
                    
                )
                plt.xlabel(
                    {
                        sanitize.FS:'frequency offset (megahertz)',
                        sanitize.FC:'laser power (watts)'
                    }[mode]
                )
                plt.ylabel('bolometer signal (normalized)')           
                plt.savefig(
                    os.path.join(
                        imagefolder,
                        '{:03d}-{:d}.png'.format(lineindex,mode)
                    )
                )
                plt.cla()

if __name__ == '__main__':
    main()
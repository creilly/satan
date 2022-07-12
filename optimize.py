from doctest import master
from saturation import sanitize
from mpi4py import MPI
import numpy as np
from time import perf_counter
from saturation import rabi, gamma, doppler, beam, bloch
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution
import os

debug = False

N = 2000

beam_diameter = 4.0e-3 # meters
noise_amplitude = 30. # millivolts rms
velocity = 1000. # meters per second

gamma = gamma.get_gamma(noise_amplitude) * 1e-6 # rads / us
tau = beam.get_tau(beam_diameter,velocity) * 1e6 # us
sigmaomega = doppler.get_deltaomega(velocity) * 1e-6 # rads / us

jmax = 10

degens = np.array([2 if m else 1 for m in range(jmax + 1)])

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

# get number of processors and processor rank
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
masterrank = size - 1

md = sanitize.load_metadata()
cgcd = sanitize.load_cgc()

def get_total_length():
    total_length = 0
    for lined in md:
        j1 = lined['j1']
        for mode in (sanitize.FC,sanitize.FS):
            total_length += lined['lengths'][str(mode)] * (j1+1)
    return total_length

END = -1
sendrequests = []
recvrequests = []
data = []

MEASURED, COMPUTED = 0, 1
arrays = []    
dataarrays = {}
indexbuffer = np.empty(4,dtype=np.int64)

def send(buffer,dest=masterrank):
    comm.Send(buffer,dest)
def isend(buffer,dest=masterrank):
    sendrequests.append(comm.Isend(buffer,dest))
def recv(buffer,source):
    comm.Recv(buffer,source)
def irecv(buffer,source,jobindex):    
    recvrequests[jobindex][source] = comm.Irecv(buffer,source)
def clear_sends():
    clear_list(sendrequests)
def wait_recvs():
    for jobd in recvrequests:
        for request in jobd.values():
            request.Wait()
def clear_list(l):
    while l:
        l.pop(0).Wait()

def get_data():
    total_length = get_total_length()
    chunk = total_length // size    
    rankmarker = 0        
    _rank = 0    
    for lineindex, lined in enumerate(md):        
        j1 = lined['j1']
        j2 = lined['j2']
        w = lined['wavenumber'][0] # cm-1
        a = lined['einstein coefficient'][0] # s-1
        mubar = rabi.mubar(w,a)
        for mode in (sanitize.FC,sanitize.FS):
            modelength = lined['lengths'][str(mode)]            
            modemarker = 0
            while True:                
                modetail = modelength - modemarker
                ranktail = chunk - rankmarker
                tail = (
                    ranktail // (j1 + 1) + (1 if ranktail % (j1 + 1) else 0)
                ) if modetail * (j1 + 1) > ranktail else modetail
                if rank == _rank:
                    mus = [
                        cgcd[j1][j2][m] * mubar for m in range(j1+1)
                    ]
                    # rads / us, watts, normed
                    deltaomegas, powers, zs = sanitize.load_data(lineindex,mode)[modemarker:modemarker+tail].transpose()                    
                    omegabars = np.outer(mus,1e-6 * 2 * np.pi * rabi.elec_field(powers/(np.pi * (beam_diameter/2)**2)) / rabi.h) # rads / us (n_mu, n_points)
                    M_omegabars = np.einsum('ij,kl->ijkl',omegabars,M_omegabar)
                    start = modemarker
                    stop = modemarker + tail
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
                    isend(np.array((lineindex,mode,start,stop)))                    
                    isend(np.ascontiguousarray(zs))
                    data.append(
                        (
                            deltaomegas,M_omegabars
                        )
                    )                    
                rankmarker += tail * (j1 + 1)
                if chunk - rankmarker <  j1 + 1:
                    if rank == _rank:
                        isend(np.array([END]*4))
                    rankmarker = 0
                    _rank += 1                    
                    if rank < _rank:
                        return
                modemarker += tail
                if modemarker == modelength:
                    break   
    # if last rank falls off the edge    
    isend(np.array([END]*4))
def finish_get_data():
    if rank == masterrank:    
        for lineindex, lined in enumerate(md):
            linearrayd = {}
            dataarrays[lineindex] = linearrayd        
            for mode in (sanitize.FC,sanitize.FS):
                modelength = lined['lengths'][str(mode)]
                linearrayd[mode] = {
                    key:arr for key, arr in zip(
                        (MEASURED,COMPUTED),
                        np.empty(2*modelength).reshape((2,modelength))
                    )
                }    
        for _rank in range(size):
            rankarrays = []
            arrays.append(rankarrays)      
            jobindex = 0  
            while True:            
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
                if lineindex == END:
                    break            
                if jobindex == len(recvrequests):
                    jobd = {}
                    recvrequests.append(jobd)                                    
                moded = dataarrays[lineindex][mode]
                irecv(moded[MEASURED][start:stop],_rank,jobindex)
                rankarrays.append(moded[COMPUTED][start:stop])
                jobindex += 1
        wait_recvs()
    clear_sends()

# factors = np.array([1.2,0.25,0.8,0.3])
# factors = np.array([0.48407309, 0.10796933, 1.33260555, 2.02798956])
# factors = np.array([0.593, 0.281, 0.405, 1.384])
factors = np.array([0.562, 0.200, 0.912, 1.531])

def get_outdata():    
    power_factor, gamma_factor, sigmaomega_factor, tau_factor = factors
    sigmaomegap = sigmaomega_factor * sigmaomega
    taup = tau_factor * tau
    gammap = gamma_factor * gamma   
    if rank == masterrank:        
        for _rank, rankarrays in enumerate(arrays):
            for jobindex, arr in enumerate(rankarrays):
                irecv(arr,_rank,jobindex)
    for deltaomegas, M_omegabars in data:
        n_mu, n_points, *_ = M_omegabars.shape
        d_deltaomegas = bloch.get_deltaomegas_geometric(
            0.0,sigmaomegap,N*n_mu*n_points
        ).reshape((N,n_mu,n_points))
        deltaomegasp = d_deltaomegas + deltaomegas
        M_deltaomegasp = np.einsum('ijk,lm->ijklm',deltaomegasp,M_deltaomega)
        Ms = (gammap * M_gamma + power_factor * M_omegabars) + M_deltaomegasp
        lambdas, Ss = np.linalg.eig(Ms)
        Sinvs = np.linalg.inv(Ss)        
        tausp = bloch.get_taus(taup,N*n_mu*n_points).reshape((N,n_mu,n_points))
        exponents = np.exp(np.einsum('ijk,ijkl->ijkl',tausp,lambdas))
        Szs = Ss[:,:,:,2,:]
        Sinvzs = Sinvs[:,:,:,:,2]
        probs = 0.5 + 0.5 * np.real(
            (
                -Szs*Sinvzs*exponents
            ).sum(3)
        ).sum(0) / N
        modedegens = degens[:n_mu]
        probs = np.einsum('ij,i->j',probs,modedegens) / (2*n_mu - 1)
        isend(probs)    
    if rank == masterrank:
        wait_recvs()
        sse = 0.
        for lineindex, lined in dataarrays.items():
            for mode, moded in lined.items():
                measured = moded[MEASURED]
                computed = moded[COMPUTED]
                computed /= computed.sum()
                sse += ((measured - computed)**2).sum()        
    else:
        sse = None
    clear_sends()
    return sse

def optimize():
    if rank == masterrank:
        res = differential_evolution(
            _optimize,
            bounds=[(0.05,20.0)]*len(factors),
            maxiter=100,
            popsize=10,
            disp=True
        )
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
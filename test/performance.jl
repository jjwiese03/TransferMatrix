using BenchmarkTools
using TransferMatrix

dists = [7.0]*1e-3
tilts = range(-0.01, 0.01, 3)
        
freqs = range(18e9,24e9,10);

M = 1; L = 1

coords = Coordinates(1,0.02; diskR=0.15);
modes = Modes(coords,M,L);

gpm = GrandPropagationMatrix(freqs,range(6e-3,8e-3,3),tilts,modes,coords); 


# transfer_matrix_3d(Dist, dists, 0, 0, gpm, freqs[1])

b = @benchmark transfer_matrix_3d(Dist, dists, 0, 0, gpm, Int64(freqs[1])) samples=20 evals=1



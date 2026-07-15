using Test
using BenchmarkTools
using TransferMatrix

@testset "TransferMatrix Basic Testing" begin
    @testset "Performance" begin
        dists = [7.0]*1e-3
        tilts = zeros(length(dists)-1)
        print(tilts)
                
        freqs = range(18e9,24e9,1_000);

        M = 1; L = 1

        coords = Coordinates(1,0.02; diskR=0.15);
        modes = Modes(coords,M,L);

        # gpm = GrandPropagationMatrix(freqs,collect(range(6e-3,8e-3,3)),tilts,modes,coords); 

        # b = @benchmark transfer_matrix_3d(Dist, dists, 0, 0, gpm, freqs[1]) samples=20 evals=1
        
    end
end
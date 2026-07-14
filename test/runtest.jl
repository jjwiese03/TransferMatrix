using Test
using BenchmarkTools
using TransferMatrix

@testset "TransferMatrix Basic Testing" begin
    @testset "Performance" begin
        dists = [7.0]*1e-3

        b = @benchmark transfer_matrix_3d(M=5, L=5) samples=20 evals=1
        
    end
end
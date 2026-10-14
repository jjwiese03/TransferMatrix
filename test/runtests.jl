#!/usr/bin/env -S julia -t 4
using BenchmarkTools
using TransferMatrix
 

dists = [7.0, 8.0, 9.0, 8.0, 5.3] * 1e-3                 # muss in 0..10 mm liegen
freqs = range(18e9, 24e9, 20)

M = 1; L = 1

coords = Coordinates(1, 0.02; diskR=0.15)
modes  = Modes(coords, M, L)

gpm = GrandPropagationMatrix(freqs, modes, coords)

println("benchmark GrandPropagationMatrix: \n")
display(@benchmark GrandPropagationMatrix(freqs, modes, coords) samples=5 evals=1)

tilts = zeros(length(dists), 2)      # in Grad, |Wert| <= 0.05
ax    = ones(ComplexF64, gpm.ML)


# b1 = @benchmark transfer_matrix_3d($gpm, $dists, $ax, $freqs) samples=5 evals=1
# b2 = @benchmark transfer_matrix_3d($gpm, $dists, $tilts, $ax, $freqs) samples=5 evals=1

# println("\n benchmark without tilts: ")
# display(b1)

# println("\nbenchmark with tilts: ")
# display(b2)

module TransferMatrix

using LinearAlgebra, SpecialFunctions, FunctionZeros, Interpolations
using FFTW: ifftshift, fft!, ifft!

import Base.getindex, Base.setindex!, Base.size, Base.axes

include("types.jl")
include("coordinates.jl")
include("modes.jl")
include("propagation.jl")
include("transfer_matrix_1d.jl")
include("transfer_matrix_3d.jl")

end

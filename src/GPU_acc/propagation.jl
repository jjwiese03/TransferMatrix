using TransferMatrix
using Libdl

struct Cdata
    P::Ptr{ComplexF64}
    Pn0::Cint
    Pn1::Cint
    Pn2::Cint
    Pn3::Cint

    ML::Int64
    modes::Ptr{ComplexF64}
    n0::Cint
    n1::Cint
    n2::Cint
    n3::Cint

    R::Ptr{Cdouble}
    R_rows::Cint
    R_cols::Cint
    kR::Ptr{Cdouble}
    kR_rows::Cint
    kR_cols::Cint
    Phi::Ptr{Cdouble}
    Phi_rows::Cint
    Phi_cols::Cint

    diskmaskin::Ptr{Bool}
    diskmaskout::Ptr{Bool}
    mask_rows::Cint
    mask_cols::Cint
end

function gpuPropagationCoeffs(freqs::AbstractVector{<:Real},
                              distances::AbstractVector{<:Real},
                              tiltx::AbstractVector{<:Real},
                              tilty::AbstractVector{<:Real},
                              eps::Real,
                              modes::Modes,
                              coords::Coordinates;
                              library=joinpath(@__DIR__, "module.so"))
    length(distances) == length(tiltx) == length(tilty) ||
        throw(ArgumentError("distances, tiltx and tilty must have equal lengths"))
    isempty(freqs) && throw(ArgumentError("freqs must not be empty"))
    isempty(distances) && throw(ArgumentError("distances must not be empty"))

    frequencies64 = Float64.(freqs)
    distances64 = Float64.(distances)
    tiltx64 = Float64.(tiltx)
    tilty64 = Float64.(tilty)
    maskin = Matrix{Bool}(coords.diskmaskin)
    maskout = Matrix{Bool}(coords.diskmaskout)
    modeCount = modes.M * (2 * modes.L + 1)
    result = Array{ComplexF64}(undef, length(freqs), length(distances),
                               modeCount, modeCount)

    lib = Libdl.dlopen(library)
    try
        fptr = Libdl.dlsym(lib, :propagationCoeffs)
        GC.@preserve result modes coords maskin maskout frequencies64 distances64 tiltx64 tilty64 begin
            cdata = Cdata(
                pointer(result), size(result, 1), size(result, 2),
                size(result, 3), size(result, 4),
                modeCount, pointer(modes.modes), size(modes.modes, 1),
                size(modes.modes, 2), size(modes.modes, 3),
                size(modes.modes, 4),
                pointer(coords.R), size(coords.R, 1), size(coords.R, 2),
                pointer(coords.kR), size(coords.kR, 1), size(coords.kR, 2),
                pointer(coords.Φ), size(coords.Φ, 1), size(coords.Φ, 2),
                pointer(maskin), pointer(maskout), size(maskin, 1),
                size(maskin, 2),
            )
            status = @ccall $fptr(
                Ref{Cdata}(cdata)::Ref{Cdata},
                pointer(frequencies64)::Ptr{Cdouble},
                length(frequencies64)::Cint,
                pointer(distances64)::Ptr{Cdouble},
                pointer(tiltx64)::Ptr{Cdouble},
                pointer(tilty64)::Ptr{Cdouble},
                length(distances64)::Cint,
                Float64(eps)::Cdouble,
            )::Cint
            status == 0 ||
                error("GPU propagationCoeffs failed with status $status")
        end
    finally
        Libdl.dlclose(lib)
    end
    return result
end
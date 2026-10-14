using TransferMatrix
using Libdl


struct Cdata
    # target array (for result)
    P::Ptr{ComplexF64}; Pn0::Cint; Pn1::Cint; Pn2::Cint; Pn3::Cint; 
    
    # Mode data
    ML::Int64;
    modes::Ptr{Cdouble}; n0::Cint; n1::Cint; n2::Cint; n3::Cint;

    # Coords data
    R::Ptr{Cdouble};    R_rows::Cint;   R_cols::Cint
    kR::Ptr{Cdouble};   kR_rows::Cint;  kR_cols::Cint
    Phi::Ptr{Cdouble};  Phi_rows::Cint; Phi_cols::Cint

    diskmaskin::Ptr{Bool}
    diskmaskout::Ptr{Bool}
    mask_rows::Cint
    mask_cols::Cint
end

pn1 = 10; pn2 = 10; pn3 = 10; pn4 = 10;
P = zeros(ComplexF64, pn1, pn2, pn3, pn4);

M = 2; L = 1

c = Coordinates(1,0.02; diskR=0.15);
m = Modes(c,M,L);

maskin  = Matrix{Bool}(c.diskmaskin)
maskout = Matrix{Bool}(c.diskmaskout)

lib  = Libdl.dlopen(joinpath(@__DIR__, "module.so"))
fptr = Libdl.dlsym(lib, :example)

GC.@preserve P m c maskin maskout begin

    ci = Cdata(
        pointer(P),     size(P, 1), size(P, 2), size(P, 3), size(P, 4),
        m.M*(2*m.L+1), 
        pointer(m.modes), size(m.modes, 1), size(m.modes, 2), size(m.modes, 3), size(m.modes, 4),
        pointer(c.R),   size(c.R, 1),  size(c.R, 2),
        pointer(c.kR),  size(c.kR, 1), size(c.kR, 2),
        pointer(c.Φ),   size(c.Φ, 1),  size(c.Φ, 2),
        pointer(maskin), pointer(maskout),
        size(maskin, 1), size(maskin, 2),
    )
    

    freq = collect(1:0.5:1000)
    dist = fill(0.7, 10)
    tiltx = fill(0.3, 10)
    tilty = fill(0.3, 10)
    eps = 24


    res = @ccall $fptr(ci::Ref{Cdata}, 
                    pointer(freq)::Ptr{Cdouble},
                    length(freq)::Cint,
                    pointer(dist)::Ptr{Cdouble},
                    pointer(tiltx)::Ptr{Cdouble},
                    pointer(tilty)::Ptr{Cdouble},
                    length(dist)::Cint,
                    eps::Cdouble)::ComplexF64
    
    @show res
    return res
end

Libdl.dlclose(lib)

export Coordinates, Modes, GrandPropagationMatrix

struct Coordinates
    X::Vector{Float64}
    kX::Vector{Float64}
    R::Matrix{Float64}
    kR::Matrix{Float64}
    Φ::Matrix{Float64}

    diskR::Float64
    diskmaskin::BitMatrix
    diskmaskout::BitMatrix

    function Coordinates(X::AbstractVector=-0.5:0.01:0.5; diskR::Real=0.15)
        @assert X==-reverse(X) && X[1]*X[end]<0 "Coordinates must be symmetrical around 0."

        kX = kSpace(X)
        R  = [sqrt(x^2+y^2) for x in X, y in X]
        m  = R .<= diskR
        
        new(X,kX,R,
            [kx^2+ky^2 for kx in kX, ky in kX],
            [atan(y,x) for  x in  X,  y in  X],
            diskR,
            m,.!m
        )
    end

    function Coordinates(xsize::Real,dx::Real; diskR::Real=0.15)
        @assert xsize*dx > 0 "Inputs must be larger than 0."
    
        nx = ceil(xsize/2dx);
        X = -nx*dx:dx:nx*dx
        kX = kSpace(X)

        R  = [sqrt(x^2+y^2) for x in X, y in X]
        m  = R .<= diskR
    
        new(X,kX,R,
            [kx^2+ky^2 for kx in kX, ky in kX],
            [atan(y,x) for  x in  X,  y in  X],
            diskR,
            m,.!m
        )
    end
end





mutable struct Modes
    M::Int64
    L::Int64
    modes::Array{ComplexF64,4}
    kt::Vector{ComplexF64}
    id::Matrix{ComplexF64}
    zero::Matrix{ComplexF64}

    function Modes(M,L,modes,kt)                
        @assert M > 0 "m needs to be larger than 0."

        ML = M*(2L+1)
        id = Matrix{ComplexF64}(I,ML,ML)
        z = zeros(ComplexF64,ML,ML)

        new(M,L,modes,kt,id,z)
    end

    function Modes(coords,M,L)
        @assert M > 0 "m needs to be larger than 0."

        ML = M*(2L+1)
        modes = zeros(ComplexF64,length(coords.X),length(coords.X),1,ML)
        kt = zeros(ComplexF64,ML)
        
        for m in 1:M, l in -L:L
            ml = modeidx(m,l,L)
            kt[ml], modes[:,:,:,ml] = mode(coords,m,l)
        end

        return Modes(M,L,modes,kt)
    end
end

fieldDims(modes::Modes) = size(modes.modes,3)

function modeidx(m::Int,l::Int,L::Int)
    return (m-1)*(2L+1)+l+L+1
end

function modeidx(ml::Int,L::Int)
    return div(ml-1,(2L+1))+1, (ml-1)%(2L+1)+-L
end



Base.getindex(m::Modes,inds...) = getindex(m.modes,inds...)
Base.getindex(m::Modes,ind1,ind2) = getindex(m.modes,:,:,:,modeidx(ind1,ind2,m.L))
Base.getindex(m::Modes,ind1,ind2,ind3) = getindex(m.modes,:,:,ind1,modeidx(ind2,ind3,m.L))

Base.setindex!(m::Modes,x,inds...) = setindex!(m.modes,x,inds...)
Base.setindex(m::Modes,ind1,ind2) = setindex(m.modes,:,:,:,modeidx(ind1,ind2,m.L))
Base.setindex(m::Modes,ind1,ind2,ind3) = setindex(m.modes,:,:,ind1,modeidx(ind2,ind3,m.L))

Base.size(m::Modes) = size(m.modes)
Base.size(m::Modes,d::Integer) = size(m.modes,d)
Base.axes(m::Modes,d::Integer) = axes(m.modes,d)









mutable struct GrandPropagationMatrix
    freqs::Union{Real,AbstractVector{Float64}}
    # thickness::Float64
    # nd::ComplexF64

    M::Int
    L::Int
    ML::Int

    ax::ScaledInterpolation
    P::ScaledInterpolation

    function GrandPropagationMatrix(freqs,distances,tilts,modes,coords; 
            eps::Real=24.0,tand::Real=0.0,thickness::Real=1e-3,nm::Real=1e15)

        M = modes.M; L = 2modes.L+1; ML = M*L

        bc = BSpline(Cubic(Natural(OnCell())))
        ni = NoInterp()
        
        ax = axionModes(coords,modes,freqs,tilts)
        itpax = interpolate(ax,(ni,ni,bc,bc))
        sitpax = scale(itpax,1:ML,1:length(freqs),tilts,tilts)

        p = propagationMatrix(freqs,distances,tilts,1.0,modes,coords);
        itpP = interpolate(p,(ni,ni,ni,bc,bc,bc))
        sitpP = scale(itpP,1:ML,1:ML,1:length(freqs),distances,tilts,tilts)

        new(freqs,M,L,ML,sitpax,sitpP)
    end
end

const GPM = GrandPropagationMatrix

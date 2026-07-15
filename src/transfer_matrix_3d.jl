export transfer_matrix_3d

function transfer_matrix_3d(::Type{Dist},distances::AbstractVector{<:Real},
        tiltsx,tiltsy,gpm::GPM,f::Int;
        eps::Real=24.,tand::Real=0.,nm::Real=1e30)
    
    ϵ = eps*(1.0-1.0im*tand); nd = sqrt(ϵ); nm = complex(nm); ϵm = nm^2
    A = 1-1/ϵ; A0 = 1-1/ϵm

    P = gpm.P; ax = gpm.ax; ML = gpm.ML

    Gd = ComplexF64[I(ML)*(1+nd)/2   I(ML)*(1-nd)/2]#;   I(ML)*(1-nd)/2   I(ML)*(1+nd)/2]
    Gv = ComplexF64[I(ML)*(nd+1)/2nd I(ML)*(nd-1)/2nd]#; I(ML)*(nd-1)/2nd I(ML)*(nd+1)/2nd]
    G0 = ComplexF64[I(ML)*(1+nm)/2   I(ML)*(1-nm)/2]#;   I(ML)*(1-nm)/2   I(ML)*(1+nm)/2]

    ϵ  = eps*(1.0-1.0im*tand); nd = sqrt(ϵ); nm = complex(nm); ϵm = nm^2
    A  = 1-1/ϵ; A0 = 1-1/ϵm

    G0 = G(ML,nm,1)
    Gv = G(ML,1,nd)
    Gd = G(ML,nd,1)

    S  =  A/2*I(ML)*diagm(ax)
    S0 = A0/2*I(ML)*diagm(ax)


    S  = A/2*I(ML); S0 = A0/2*I(ML)
    
    M = copy(S)
    
    RB = Array{ComplexF64}(undef,2,ML)
    
    T  = Array{ComplexF64}(undef,2,2,ML)
    MM = zeros(ComplexF64,2,2,ML)

    TW = Array{ComplexF64}(undef,2,2,ML)
    W = Matrix{ComplexF64}(undef,2,2)
    


    pd1 = 1.# cispi(-2*gpm.freqs[f]*nd*(1e-3)/c0)
    pd2 = 1.# cispi(+2*gpm.freqs[f]*nd*(1e-3)/c0)

    for ml in 1:ML
        @views copyto!(T[:,:,ml],Gd)

        s = ax(ml,f,0.,0.)
        MM[1,1,ml] = s*S; MM[2,2,ml] = s*S
    end
    
    # iterate in reverse order to sum up MM in single sweep (thx david)
    for i in Iterators.reverse(eachindex(distances))
        for ml in 1:ML
            @. T[:,1,ml] *= pd1
            @. T[:,2,ml] *= pd2                                             # T = Gd*Pd
            
            @. MM[:,:,ml] -= T[:,:,ml]*S*ax(ml,f,tiltsx[i+1],tiltsy[i+1])   # MM = Gd*Pd*S_-1
            @views mul!(W,T[:,:,ml],Gv); @views copyto!(T[:,:,ml],W)        # T *= Gd*Pd*Gv
        end

        TW .= 0.0im
        
        for ml in 1:ML
            for ml_ in 1:ML
                s = P(ml,ml_,f,distances[i],tiltsx[i+1]-tiltsx[i],tiltsy[i+1]-tiltsy[i])

                @. TW[:,1,ml] += T[:,1,ml_]*s
                @. TW[:,2,ml] += T[:,2,ml_]*conj(s)
            end

            if i > 1
                @. MM[:,:,ml] += TW[:,:,ml]*S*ax(ml,f,tiltsx[i],tiltsy[i])
                @views mul!(W,TW[:,:,ml],Gd); @views copyto!(TW[:,:,ml],W)
            else
                @. MM[:,:,ml] += TW[:,:,ml]*S0*ax(ml,f,tiltsx[i],tiltsy[i])
                @views mul!(W,TW[:,:,ml],G0); @views copyto!(TW[:,:,ml],W)
            end
        end

        copyto!(T,TW)
    end

    return RB
end


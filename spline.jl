
using LinearAlgebra: Tridiagonal

function eval_polynomial(x::Number,coeffs::AbstractVector)
    p = 0

    @inbounds @simd for i in eachindex(coeffs)
        p += coeffs[i]*x^(i-1)
    end

    return p
end

function eval_polynomial(x::Vector{<:Number},coeffs::AbstractVector)
    p = zeros(eltype(coeffs),length(x))

    @inbounds @simd for i in eachindex(coeffs)
        @. p += coeffs[i]*x^(i-1)
    end

    return p
end

struct Spline{T}
    order::Int
    knots::Vector{Float64}
    coeffs::Matrix{T}

    function Spline(order,knots,coeffs)
        @assert order+1 == size(coeffs,2)
        @assert length(knots) == size(coeffs,1)+1

        T = eltype(coeffs)
        new{T}(order,knots,coeffs)
    end
end

function cSpline(x::AbstractArray{<:Real},y::AbstractArray{<:Number}; b=[1,0,0,1],c=[0,0])
    @assert length(x) == length(y) "x and y need same lengths."
    @assert length(x) > 1 "Need at least 2 knots."

    T = eltype(y) <: Complex ? ComplexF64 : Float64

    h = x[2]-x[1]
    n = length(x)

    A = zeros(T,n-1,4)

    du = ones(T,n-1);   du[1] = b[2]
    d  = ones(T,n)*4;    d[1] = b[1]; d[end] = b[4]
    dl = ones(T,n-1); dl[end] = b[3]

    A2 = Tridiagonal(dl, d, du); A2_ = inv(A2)

    D = zeros(T,n);
    D[1] = c[1]; D[end] = c[2]
    for i in 2:n-1
        D[i] = y[i+1]-2y[i]+y[i-1]
    end
    D .*= 3/h^2; a2 = A2_*D

    @. A[:,4] = (a2[2:end]-a2[1:end-1])/3h
    @. A[:,3] = a2[1:end-1]
    @. A[:,2] = (y[2:end]-y[1:end-1])/h - A[:,3]*h - A[:,4]*h^2
    @. A[:,1] =  y[1:end-1]

    return Spline(3,x,A)
end



function cSpline(bounds,len,f; kwargs...)
    x = collect(range(bounds[1],bounds[2],len))
    y = f.(x)

    return cSpline(x,y; kwargs...)
end

function differentiate(spline::Spline)
    if spline.order == 0
        return Spline(0,spline.knots,zeros(size(spline.coeffs)))
    end

    coeffs = Matrix{Float64}(undef,size(spline.coeffs,1),spline.order)

    for i in 1:spline.order
        coeffs[:,i] = i*spline.coeffs[:,i+1]
    end

    return Spline(spline.order-1,spline.knots,coeffs)
end

const diff = differentiate

function differentiate(spline::Spline,order::Int)
    for _ in 1:order; spline = differentiate(spline); end; return spline
end

function spline(spline::Spline,x::Real)
    idx1 = findlast(y->y<=x,spline.knots)
    if isnothing(idx1); idx1 = 1; end
    idx2 = min(idx1,length(spline.knots)-1)

    return eval_polynomial(x-spline.knots[idx1],spline.coeffs[idx2,:])
end

function spline(spline::Spline,x::Vector{<:Real})
    if x[end] <= spline.knots[2]
        return eval_polynomial(x.-spline.knots[1],spline.coeffs[1,:])
    elseif x[1] >= spline.knots[end-1]
        return eval_polynomial(x.-spline.knots[end-1],spline.coeffs[end,:])
    end
    
    s = zeros(eltype(spline.coeffs),length(x))
    
    idx0 = findlast(y->y<=x[1],spline.knots)
    if isnothing(idx0); idx0 = 1; end

    idx1 = 1
    if idx0 < length(spline.knots)
        idx2 = findnext(y->y>=spline.knots[idx0+1],x,idx1+1)
        if isnothing(idx2); idx2 = length(x)+1; end
        idx2 -= 1
    else
        idx2 = length(x)
    end
    
    s[idx1:idx2] .= eval_polynomial(x[idx1:idx2].-spline.knots[idx0],
        spline.coeffs[idx0,:])

    while idx2 < length(x)
        idx0 += 1
        idx1 = idx2+1
        if idx0 < length(spline.knots)-1
            idx2 = findnext(y->y>=spline.knots[idx0+1],x,idx1+1)
            if isnothing(idx2); idx2 = length(x)+1; end
            idx2 -= 1
        else
            idx2 = length(x)
        end
        idx0 = min(idx0,length(spline.knots)-1)
        
        s[idx1:idx2] .= eval_polynomial(x[idx1:idx2].-spline.knots[idx0],
            spline.coeffs[idx0,:])
    end

    return s
end
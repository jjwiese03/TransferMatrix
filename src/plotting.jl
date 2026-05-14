
export showField, showCross

function showField(E::Array{ComplexF64}; kwargs...)
    for i in axes(E,3)
        display(heatmap(abs.(E[:,:,i]); right_margin=4Plots.mm,kwargs...))
    end

    return
end

function showField(E::Array{ComplexF64},coords::Coordinates; kwargs...)
    for i in axes(E,3)
        display(heatmap(coords.X,coords.X,abs.(E[:,:,i]); right_margin=4Plots.mm,kwargs...))
    end

    return
end

function showCross(E::Array{ComplexF64}; kwargs...)
    n = size(E,1); @assert isodd(n) "Grid for E needs odd edge lengths."; n2 = div(n+1,2)
    for i in axes(E,3); display(plot(abs.(E[n2,:,i]); kwargs...)); end

    return
end

function showCross(E::Array{ComplexF64},coords::Coordinates; kwargs...)
    n = size(E,1); @assert isodd(n) "Grid for E needs odd edge lengths."; n2 = div(n+1,2)
    for i in axes(E,3); display(plot(coords.X,abs.(E[n2,:,i]); kwargs...)); end

    return
end

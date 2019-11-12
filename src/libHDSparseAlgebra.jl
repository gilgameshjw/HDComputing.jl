

#==============================================================
        FAST SEARCH AND ENCODING VIA SPARSE BRAIN
==============================================================#


function getSims(sparseM::SparseMatrixCSC{Int64,Int64},
                 hdvec::SparseVector{Int64,Int64})
    sparseM * hdvec
end


function getNearestIds(sparseM::SparseMatrixCSC{Int64,Int64},
                       hdvec::SparseVector{Int64,Int64},
                       nNearest::Int64)
     nz = length(hdvec.nzval)
     getSims(sparseM, hdvec) |>
        enumerate |>
            collect |>
                (Id_S ->
                 partialsort(Id_S, by= x -> x[2], 1:nNearest, rev=true)) |>
                    (I_S -> map(i_s -> Dict(:id => i_s[1],
                                            :sim => i_s[2],
                                            :simN => i_s[2] / nz), I_S))
end


mutable struct HDBrain

    matrix::SparseMatrixCSC{Int64,Int64}
    mapIdNode::Dict{Int64, Any}

end


# overload Aggregators
(Base.push!)(m::SparseMatrixCSC{Int64,Int64}, Is::Array{Int64,1}) = begin
    m = vcat([m, sparse([1 for i in Is], Is, [1 for i in Is], 1, 10000)]...)
end

(Base.push!)(m::SparseMatrixCSC{Int64,Int64}, hdvec::SparseVector{Int64,Int64}) = begin
    m = vcat([m, sparse([1 for i in hdvec.nzind], hdvec.nzind, [1 for i in hdvec.nzind], 1, 10000)]...)
end


function brains2HDMatrices(hdvecs)
    I, J, K = Int64[], Int64[], Int64[]
    @showprogress for (nb, hdvec) in enumerate(hdvecs)
        for i in hdvec.nzind
            if 1 <= i <= dicBrain[:dicParams][:defaultModel][:N]
                push!(I, nb)
                push!(J, i)
                push!(K, 1)
            end
        end
    end
    sparse(I, J, K, length(hdvecs), dicBrain[:dicParams][:defaultModel][:N])
end


function memory2HDBrain(memory)
    collect(memory) |>
        (nodeHD -> (HDBrain(map(b -> b[2], nodeHD) |> brains2HDMatrices,
                            map(b -> b[1], nodeHD) |>
                                (Ns -> map(iNode -> iNode[1] => iNode[2], enumerate(Ns))) |>
                                    Dict)))
end

#=====================
    SDMs model 0.1
======================#

using SparseArrays

module HDSparse

    import SparseArrays
    import SparseArrays.sparsevec

    BITFORMAT = Int64

    function RandomGenerator(n::Int64, w::Int64)
        # generates random vector of form [0 0 0 ... 0 1 0 ... 0 1 0 0 0 ...]
        rand(1:n, w) |> sort |>
            (ri -> sparsevec(ri, [1 for i=1:length(ri)], n))
    end

    function RandomGenerator(dict)
        RandomGenerator(dict[:n], dict[:w])
    end

    function superposition(vectorHD::Vector{SparseArrays.SparseVector{BITFORMAT,
                                                                      BITFORMAT}})
        unique(vcat(map(v -> v.nzind, vectorHD)...)) |>
            (Is -> sparsevec(Is,
                             map(i -> BITFORMAT(1),
                             Is)))
    end

    function superposition(vectorHD::Vector{SparseArrays.SparseVector{BITFORMAT,
                                                                      BITFORMAT}},
                           modelEncoding::Dict{Symbol,Int64})
        idces = unique(vcat(map(v -> v.nzind, vectorHD)...))
        sparsevec(idces,
                  map(i -> BITFORMAT(1), idces),
                  modelEncoding[:N])
    end

    function superposition(vectorHD::Array{Array{BITFORMAT,1},1})
        sort(unique(vcat(vectorHD...))) |>
            (Is -> sparsevec(Is,
                             map(i -> BITFORMAT(1),
                             Is)))
    end

    function superposition(vectorHD::Array{Array{BITFORMAT,1},1}, n)
        sort(unique(vcat(vectorHD...))) |>
            (Is -> sparsevec(Is,
                             map(i -> BITFORMAT(1), Is),
                             n))
    end

    function superposition(vectorHD::Array{Array{BITFORMAT,1},1},
                           modelEncoding::Dict{Symbol,Int64})
        sort(unique(vcat(vectorHD...))) |>
            (Is -> sparsevec(Is,
                             map(i -> BITFORMAT(1), Is),
                             modelEncoding[:N]))
    end

    function cosineSimilarity(ri1::SparseArrays.SparseVector{BITFORMAT,
                                                             BITFORMAT},
                              ri2::SparseArrays.SparseVector{BITFORMAT,
                                                             BITFORMAT})
        sparseVectorMultiplication(ri1, ri2) =
            filter(x -> x in ri1.nzind, ri2.nzind) |>
                    length

        sparseVectorMultiplication(ri1, ri2) /
                    (length(ri1.nzval) * length(ri2.nzval))^.5
    end

    function encodeOnTheFly(dic1DSparseRepres, modelEncoding, wrdOrTag)
        if haskey(dic1DSparseRepres, wrdOrTag)
            dic1DSparseRepres[wrdOrTag].nzind
        else
            randSDM = RandomGenerator(modelEncoding[:N], modelEncoding[:W])
            dic1DSparseRepres[wrdOrTag] = randSDM
            randSDM.nzind
        end
    end

end
export HDSparse

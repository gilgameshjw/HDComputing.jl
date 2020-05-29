

struct LocallyLinearEmbedding
    
    n::Int64
    w::Int64
    data
    nData::Int64
    shifts::Vector{Int64}

    LocallyLinearEmbedding(n::Int64, w::Int64, data) = (nData = if typeof(data) <: Vector
                                                                   length(data)
                                                                elseif typeof(data) <: Matrix
                                                                    size(data)[1]
                                                                elseif typeof(data) <: SparseMatrixCSC
                                                                    size(data)[1]
                                                                else
                                                                    println("error: data type not recognised")
                                                                    nothing
                                                                end;
                                                        shifts = StatsBase.sample(1:n, w, replace = false);
                                                        new(n, w, data, nData, shifts))

end


function encodeData2Idces(dicData, le::LocallyLinearEmbedding, nNeighb::Int64, dMetric::Function)
    
    ids = map(d -> dMetric(d, dicData), le.data) |>
            (D -> sort(1:length(D), by=i-> D[i])[1:nNeighb])
    
    vcat(
         map(id -> map(w -> (id + w) % le.n + 1, le.shifts),
             ids)...)

end


function encodeData2Idces(idx::Integer, le::LocallyLinearEmbedding, nNeighb::Int64)
    
    if typeof(le.data) <: Matrix
        
        col = le.data[idx,:]
        ids = sort(1:le.nData, by=i->col[i],rev=true)[1:nNeighb]
    
    elseif typeof(le.data) <: SparseMatrixCSC
        
        col = le.data[idx,:]
        ids = col.nzind[1:min(end,nNeighb)]
        
    else
        
        println("error: wrong data type detected for locallinearembeddings // graph")
        nothing
 
    end
    
    vcat(
         map(id -> map(w -> (id + w) % le.n + 1, le.shifts),
             ids)...)

end


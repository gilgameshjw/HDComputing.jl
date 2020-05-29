

getFormat(n) = ((if n <= 127
                    return Int8
                 end);
                (if n <= 32767
                    return Int16
                 end);
                (if n <= 2147483647
                    return Int32
                 end);
                return Int64)


function sparse2Dense(hdvec::SparseVector)
    
    n = hdvec.n
    vec = zeros(n)
    map(i -> vec[i] += 1, hdvec.nzind)
    vec
    
end


#===
    Data into sparse vector format:
===#

function encodeData2SparseVec(dicData, data2Idces::Function, encoderModel::Dict{Symbol,Int64})
        
    n = encoderModel[:N]
    DATAFormatn = getFormat(n)
    
    data2Idces(dicData) |>
        (I -> sparsevec(DATAFormatn.(I), Int8.([1 for i in I]), n))
    
end


#===
    Data into sparse matrix format:
===#

function encodeData2SparseMat(dicsData, data2Idces::Function, encoderModel::Dict{Symbol,Int64})
    
    n = encoderModel[:N]
    nData = length(dicsData)
    
    # DATAFormatnData = getFormat(nData)
    # DATAFormatn = getFormat(n)
    
    I, J, K = Int64[], Int64[], Int64[]
    @showprogress for j=1:nData

        for i in data2Idces(dicsData[j])
                
            push!(I, j); push!(J, i); push!(K, 1)
                        
        end

    end
    
    # sparse(DATAFormatnData.(I), DATAFormatn.(J), Int8.(K), nData, n)
    sparse(I, J, K, nData, n)
    
end


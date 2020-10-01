

struct Lattice
   
    gridSizes::Dict{Symbol,Int64}
    nDimensions::Int64
    shifts::Dict{Symbol,Vector{Any}}
    indices::Dict{Symbol,Int64}
    N::Int64
    
end

cp(lat::Lattice) = Lattice(lat.gridSizes,lat.nDimensions,lat.shifts,lat.indices,lat.N)


struct CoarseEncoder
    
    dimensions::Array{Symbol,1}
    nLattices::Int64
    maximas::Dict{Symbol,Any}
    minimas::Dict{Symbol,Any}
    resolution::Dict{Symbol,Any}
    
end

cp(ce::CoarseEncoder) = CoarseEncoder(ce.dimensions,ce.nLattices,ce.maximas,ce.minimas,ce.resolution)


function generateLattice(cE::CoarseEncoder)
    
    # build grid 
    gridSizes = map(d -> d => Int64(round((cE.maximas[d] - cE.minimas[d]) / 
                                            cE.resolution[d])), cE.dimensions) |>
                      Dict

    nDimensions = length(cE.dimensions)

    # Shifts are necessary for the superposition of coarse matrices
    shifts = map(d -> d => [i/cE.nLattices for i=0:cE.nLattices-1], cE.dimensions) |> 
                Dict

    # If X,Y,Z with x,y,z resp. dimensions resp. corresponding data points involved,
    # We allocate univoque indices on the lattice as follows:
    # index(x,y,z) = x + (L(X)+1)*y + (L(X)*L(Y)+1)*z
    indices = [[cE.dimensions[1] => 1];
               [cE.dimensions[i+1] => 
                    reduce(*,  map(k -> gridSizes[k], cE.dimensions[1:i]))+1 for i=1:nDimensions-1]][1:nDimensions] |>
                Dict

    # +3 from 1 for rounding, 1 for shift and one for +1 julia for notations
    N = reduce(*,map(gL -> gL+3, values(gridSizes)))

    # Generate lattices
    Lattice(gridSizes, nDimensions, shifts, indices, N)
    
end


function getPos(x::Number, mi::Number, ma::Number, grid::Int64)
    
    grid * (x - mi) / (ma - mi)
    
end


function getPos(x::TimePeriod, mi::TimePeriod, ma::TimePeriod, grid::Int64)
    
    grid * (x - mi) / (ma - mi)
    
end


function generateIndices(data, ce::CoarseEncoder, lat::Lattice)

    map(i -> (i-1)*lat.N +
            (map(d -> 
                lat.indices[d] * 
                    (1 + 
                        lat.shifts[d][i] + 
                        getPos(data[d], ce.minimas[d], ce.maximas[d], lat.gridSizes[d]) |> 
                      round |> Int64), 
                ce.dimensions) |> sum),
        1:ce.nLattices)
    
end


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


function encodeData2SparseVec(dicData, coarseEncoder::CoarseEncoder, lattice::Lattice)
    
    idces = generateIndices(dicData, coarseEncoder, lattice)
    n = lattice.N*coarseEncoder.nLattices
    # DFormatn = getFormat(n)
    sparsevec(idces, [1 for i in idces], n)

end


function encodeDataset2SparseMat(dicsData, coarseEncoder::CoarseEncoder, lattice::Lattice)
    
    nData = length(dicsData)
    n = lattice.N*coarseEncoder.nLattices
    
    DFormatn = getFormat(n)
    DFormatnData = getFormat(nData)
    
    I, J, K = Int64[], Int64[], Int64[]
    @showprogress for j=1:nData

        for i in generateIndices(dicsData[j], coarseEncoder, lattice)
                
            push!(I, j); push!(J, i); push!(K, 1)
                        
        end

    end
    
    #sparse(DFormatnData.(I), DFormatn.(J), Int8.(K), nData, n)
    sparse(I, J, K, nData, n)
    
end



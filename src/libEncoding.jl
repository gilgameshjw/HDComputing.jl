
using SparseArrays


#==============================================================
        HELPERS FOR ENCODER
==============================================================#


function overestimationOverlapRisk(W, N)
    p = W / N
    stdDev = (W*p*(1-p))^.5
    for i=1:10
        println("stdDev.:", i, " max loss:", W*p + i * 2 * stdDev)
    end
end


function getCapacity(dicBrain)
    sum(map(m -> m[2][:W]*m[2][:maxNbElements],
            collect(dicBrain[:dicModel][:models])))
end


#==============================================================
        ENCODING WITH RANDOM INDICES
==============================================================#


shiftArray(v::Array{Int64,1}, shift::Int64, n::Int64) =
                                        map(i -> (i+shift-1) % n + 1, v)

shiftHD(hdv::SparseVector{Int64,Int64}, shift::Int64) =
                    shiftArray(hdv.nzind, shift, hdv.n) |>
                        (R -> sparsevec(R, [1 for i=1:length(R)], hdv.n))


encodeDict(dicEvent) = sort(filter(e -> typeof(e[2]) <: Array,
                            collect(dicEvent)))


function encoderAtom(data, key, dicBrain)

    if !haskey(dicBrain[:dicMemories], key)
        #=======================================
            Initialise cognitive system
        =======================================#
        dicBrain[:dicMemories][key] =
                        Dict{Any,SparseArrays.SparseVector{Int64,Int64}}()

        model = get(dicBrain[:dicParams],
                    key,
                    dicBrain[:dicParams][:defaultModel])
        # set \Pi_key shift
        model[:Permut] = 11 + 100 * length(dicBrain[:dicMemories][key])
        dicBrain[:dicModel][:models][key] = model
    end

    if typeof(data) <: Array
        vcat([HDSparse.encodeOnTheFly(dicBrain[:dicMemories][key],
                                      dicBrain[:dicModel][:models][key],
                                      d) for d in data]...)  # encoding not ordered
    else
        HDSparse.encodeOnTheFly(dicBrain[:dicMemories][key],
                                dicBrain[:dicModel][:models][key],
                                data)
    end
end


function encodeEvent(event::Array{Pair{String,String},1},
                     dicBrain::Dict{Symbol,Dict},
                     train=true)
    
    if !haskey(dicBrain[:dicMemory], event)
        #================================================
            If node not in memory => build memory
        ================================================#
        hdvec =  map(kv -> (k = kv[1];
                            v = kv[2];
                            encoderAtom(v, k, dicBrain)),
                     event) |>
                (Vs -> vcat(Vs...)) |>
                    (ri ->
                        sparsevec(ri,
                                  [1 for i=1:length(ri)],
                                  dicBrain[:dicParams][:defaultModel][:N]))

        train ?
            dicBrain[:dicMemory][event] = hdvec : ""
        return hdvec
    end

    dicBrain[:dicMemory][event]
end


encodeEvent(dicEvent::Dict{String,String},
            dicBrain::Dict{Symbol,Dict},
            train=true) =
    sort(collect(dicEvent), by=d -> d[1]) |>
        (E -> encodeEvent(E, dicBrain, train))


encodeEvent(dicEvent::Dict{String,Array{String,1}},
            dicBrain::Dict{Symbol,Dict},
            train=true) =
    vcat(
        map(kv -> map(v -> kv[1] => v, kv[2]), collect(dicEvent))
        ...) |> 
            Dict |>
            (E -> encodeEvent(Dict(E), dicBrain, train))


function encodeEvents(events, dicBrain; train=true)
    #================================================
        Encode a sequence of events
    ================================================#        
    idces = 1:length(events)
    map(i -> 
            shiftHD(encodeEvent(events[i], dicBrain, train), i-1), 
        idces) |>
            (Vs -> HDSparse.superposition(Vs, dicBrain[:dicParams][:defaultModel]))
end                    


#==============================================================
        SEARCH NEXT NEIGHBOURS WITH SIMILARITY
==============================================================#


function getNextNodes(hdvec::SparseArrays.SparseVector{Int64,Int64},
                      brain::Dict{Any, SparseArrays.SparseVector{Int64,Int64}},
                      nNext::Int64)

    lstBrains = collect(brain)
    sims = map(b -> HDSparse.cosineSimilarity(hdvec, b[2]),
               lstBrains)
    ids = sort(1:length(sims), by=i->sims[i], rev=true)[1:min(nNext, end)]
    nz = length(hdvec.nzval)
    map(i -> Dict(:sim => sims[i],
                  :simN => sims[i] / nz,
                  :id => Dict(lstBrains[i][1])),
        ids)
end

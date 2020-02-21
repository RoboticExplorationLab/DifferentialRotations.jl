module DifferentialRotations

using LinearAlgebra
using StaticArrays
using ForwardDiff
import Base: +, -, *, /, \, exp, log, ≈, ==, inv, conj
import LinearAlgebra: norm2

export
    Rotation,
    UnitQuaternion,
    MRP,
    RPY,
    RodriguesParam,
    ExponentialMap,
    VectorPart,
    MRPMap,
    CayleyMap,
    IdentityMap,
    ReNorm

export
    differential_rotation,
    map_type,
    scalar,
    vector,
    logm,
    expm,
    kinematics,
    rotmat,
    Lmult,
    Rmult,
    Vmat,
    Hmat,
    Tmat,
    skew,
    vee,
    ⊕,
    ⊖

abstract type Rotation{T} <: StaticMatrix{3,3,T} end

function skew(v::AbstractVector)
    @assert length(v) == 3
    @SMatrix [0   -v[3]  v[2];
              v[3] 0    -v[1];
             -v[2] v[1]  0]
end
skew(r::Rotation) = skew(SVector(r))

function vee(S::AbstractMatrix)
    return @SVector [S[3,2], S[1,3], S[2,1]]
end


include("unitquaternion.jl")
include("MRPs.jl")
include("rodrigues_params.jl")
include("rpy.jl")
include("conversions.jl")


""" "Properly" subtract two rotations, returning a 3-parameter vector
Equivalent to returning the 3-parameter version of `inv(q0)*q`

For Quaternions, the map of the first argument is used to convert to
    a three-parameter error
"""
(⊖)(p::Rotation, p0::Rotation) = SVector(p0\p)

end # module

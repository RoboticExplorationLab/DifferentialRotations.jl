
############################################################################################
#                                CONVERT TO ROTATION MATRICES
############################################################################################
function rotmat(q::UnitQuaternion)
    s = q.s
    v = vector(q)
    (s^2 - v'v)*I + 2*v*v' + 2*s*skew(v)
end

function rotmat(g::RodriguesParam)
    ghat = skew(SVector(g))
    I + 2*ghat*(ghat + I)/(1+norm2(g))
end

function rotmat(p::MRP)
    p = SVector(p)
    P = skew(p)
    R2 = I + 4( (1-p'p)I + 2*P )*P/(1+p'p)^2
end

# Conversion to RPY in rpy.jl

############################################################################################
#                                CONVERT FROM ROTATION MATRICES
############################################################################################

""" Convert from a rotation matrix to a unit quaternion
Uses formula from Markely and Crassidis's book
    "Fundamentals of Spacecraft Attitude Determination and Control" (2014), section 2.9.3
"""
function rotmat_to_quat(A::SMatrix{3,3,T}) where T
    trA = tr(A)
    v,i = findmax(diag(A))
    if trA > v
        i = 1
    else
        i += 1
    end
    if i == 1
        q = UnitQuaternion(1+trA, A[2,3]-A[3,2], A[3,1]-A[1,3], A[1,2]-A[2,1])
    elseif i == 2
        q = UnitQuaternion(A[2,3]-A[3,2], 1 + 2A[1,1] - trA, A[1,2]+A[2,1], A[1,3]+A[3,1])
    elseif i == 3
        q = UnitQuaternion(A[3,1]-A[1,3], A[2,1]+A[1,2], 1+2A[2,2]-trA, A[2,3]+A[3,2])
    elseif i == 4
        q = UnitQuaternion(A[1,2]-A[2,1], A[3,1]+A[1,3], A[3,2]+A[2,3], 1 + 2A[3,3] - trA)
    end
    return normalize(inv(q))
end

function rotmat_to_rp(A::SMatrix{3,3,T}) where T
    RodriguesParam(rotmat_to_quat)
end

function rotmat_to_mrp(A::SMatrix{3,3,T}) where T
    MRP(rotmat_to_quat)
end

# from RPY in rpy.jl


############################################################################################
#                                OTHER CONVERSIONS
############################################################################################

# ~~~~~~~~~~~~~~~~ Quaternion <=> RPY ~~~~~~~~~~~~~~~ #
(::Type{<:UnitQuaternion})(e::RPY) = rotmat_to_quat(rotmat(e))
(::Type{<:RPY})(q::UnitQuaternion) = from_rotmat(rotmat(q))


# ~~~~~~~~~~~~~~~~ Quaternion <=> MRP ~~~~~~~~~~~~~~~~~~ #
function (::Type{Q})(p::MRP) where Q <: UnitQuaternion
    p = SVector(p)
    n2 = p'p
    M = 2/(1+n2)
    q = UnitQuaternion{MRPMap}((1-n2)/(1+n2), M*p[1], M*p[2], M*p[3])
    Q(q)
end

function (::Type{<:MRP})(q::UnitQuaternion)
    M = 1/(1+q.s)
    MRP(q.x*M, q.y*M, q.z*M)
end

# ~~~~~~~~~~~~~~~ Quaternion <=> RP ~~~~~~~~~~~~~~~~~~ #
function (::Type{Q})(g::RodriguesParam{T}) where {T,Q<:UnitQuaternion}
    M = 1/sqrt(1+norm2(g))
    q = UnitQuaternion{T,CayleyMap}(M, M*g.x, M*g.y, M*g.z)
    Q(q)
end

function (::Type{<:RodriguesParam})(q::UnitQuaternion)
    M = 1/q.s
    RodriguesParam(q.x*M, q.y*M, q.z*M)
end

# ~~~~~~~~~~~~~~~ RP <=> MRP ~~~~~~~~~~~~~~~ #
(::Type{<:RodriguesParam})(p::MRP) = RodriguesParam(UnitQuaternion(p))
(::Type{<:MRP})(g::RodriguesParam) = MRP(UnitQuaternion(g))

# ~~~~~~~~~~~~~~ RP <=> RPY ~~~~~~~~~~~~~ #
(::Type{<:RodriguesParam})(e::RPY) = RodriguesParam(UnitQuaternion(e))
(::Type{<:RPY})(g::RodriguesParam) = RPY(rotmat(g))

# ~~~~~~~~~~~~~ MRP <=> RPY ~~~~~~~~~~~~ #
(::Type{<:MRP})(e::RPY) = MRP(UnitQuaternion(e))
(::Type{<:RPY})(p::MRP) = RPY(rotmat(p))

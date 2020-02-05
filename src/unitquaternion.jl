

abstract type QuatMap end

""" $(TYPEDEF)
4-parameter attitute representation that is singularity-free. Quaternions with unit norm
represent a double-cover of SO(3). The `UnitQuaternion` does NOT strictly enforce the unit
norm constraint, but certain methods will assume you have a unit quaternion. The
`UnitQuaternion` type is parameterized by the linearization method, which maps quaternions
to the 3D plane tangent to the 4D unit sphere. Follows the Hamilton convention for quaternions.

There are currently 4 methods supported:
* `VectorPart` - uses the vector (or imaginary) part of the quaternion
* `ExponentialMap` - the most common approach, uses the exponential and logarithmic maps
* `CayleyMap` - or Rodrigues parameters (aka Gibbs vectors).
* `MRPMap` - or Modified Rodrigues Parameter, is a sterographic projection of the 4D unit sphere
onto the plane tangent to either the positive or negative real poles.

# Constructors
```julia
UnitQuaternion(s,x,y,z)  # defaults to `VectorPart`
UnitQuaternion{D}(s,x,y,z)
UnitQuaternion{D}(q::SVector{4})
UnitQuaternion{D}(r::SVector{3})  # quaternion with 0 real part
```
"""
struct UnitQuaternion{T,D<:QuatMap} <: Rotation
    s::T
    x::T
    y::T
    z::T
end

# ~~~~~~~~~~~~~~~ Constructors ~~~~~~~~~~~~~~~ #
# Use default map
UnitQuaternion(s::T,x::T,y::T,z::T) where T = UnitQuaternion{T,DEFAULT_QUATDIFF}(s,x,y,z)
UnitQuaternion(q::SVector{4}) = UnitQuaternion{DEFAULT_QUATDIFF}(q[1],q[2],q[3],q[4])
UnitQuaternion(r::SVector{3}) = UnitQuaternion{DEFAULT_QUATDIFF}(0.0, r[1],r[2],r[3])

# Provide a map
UnitQuaternion{D}(s::T,x::T,y::T,z::T) where {T,D} = UnitQuaternion{T,D}(s,x,y,z)
UnitQuaternion{D}(q::SVector{4}) where D = UnitQuaternion{D}(q[1],q[2],q[3],q[4])
UnitQuaternion{D}(r::SVector{3}) where D = UnitQuaternion{D}(0.0, r[1],r[2],r[3])

# Copy constructors
UnitQuaternion(q::UnitQuaternion) = q
UnitQuaternion{D}(q::UnitQuaternion) where D = UnitQuaternion{D}(q.s, q.x, q.y, q.z)
UnitQuaternion{T,D}(q::R) where {T,D,R <: UnitQuaternion} =
    UnitQuaternion{T,D}(q.s, q.x, q.y, q.z)

(::Type{UnitQuaternion{T,D}})(x::SVector{4,T2}) where {T,T2,D} =
    UnitQuaternion{promote_type(T,T2),D}(x[1], x[2], x[3], x[4])

# ~~~~~~~~~~~~~~~ Getters ~~~~~~~~~~~~~~~ #
map_type(::UnitQuaternion{T,D}) where {T,D} = D
map_type(::Type{UnitQuaternion{T,D}}) where {T,D} = D

scalar(q::UnitQuaternion) = q.s
vector(q::UnitQuaternion{T}) where T = SVector{3,T}(q.x, q.y, q.z)

SVector(q::UnitQuaternion{T}) where T = SVector{4,T}(q.s, q.x, q.y, q.z)

# ~~~~~~~~~~~~~~~ Initializers ~~~~~~~~~~~~~~~ #
Base.rand(::Type{<:UnitQuaternion{T,D}}) where {T,D} =
    normalize(UnitQuaternion{T,D}(randn(T), randn(T), randn(T), randn(T)))
Base.rand(::Type{UnitQuaternion{T}}) where T = Base.rand(UnitQuaternion{T,DEFAULT_QMAP})
Base.rand(::Type{UnitQuaternion}) = Base.rand(UnitQuaternion{Float64,DEFAULT_QMAP})
Base.zero(::Type{Q}) where Q<:UnitQuaternion = Q(I)
Base.zero(q::Q) where Q<:UnitQuaternion = Q(I)


# ~~~~~~~~~~~~~~~ Math Operations ~~~~~~~~~~~~~~~ #
# Inverses
conj(q::UnitQuaternion{T,D}) where {T,D} = UnitQuaternion{T,D}(q.s, -q.x, -q.y, -q.z)
inv(q::UnitQuaternion) = conj(q)
(-)(q::UnitQuaternion{T,D}) where {T,D} = UnitQuaternion{T,D}(-q.s, -q.x, -q.y, -q.z)

# Norms
LinearAlgebra.norm(q::UnitQuaternion) = sqrt(q.s^2 + q.x^2 + q.y^2 + q.z^2)
LinearAlgebra.norm2(q::UnitQuaternion) = q.s^2 + q.x^2 + q.y^2 + q.z^2
vecnorm(q::UnitQuaternion) = sqrt(q.x^2 + q.y^2 + q.z^2)

function LinearAlgebra.normalize(q::UnitQuaternion{T,D}) where {T,D}
    n = 1/norm(q)
    UnitQuaternion{T,D}(q.s*n, q.x*n, q.y*n, q.z*n)
end

# Identity
(::Type{Q}) where Q<:UnitQuaternion = Q(1.0, 0.0, 0.0, 0.0)

# Equality
(≈)(q::UnitQuaternion, u::UnitQuaternion) = q.s ≈ u.s && q.x ≈ u.x && q.y ≈ u.y && q.z ≈ u.z
(==)(q::UnitQuaternion, u::UnitQuaternion) = q.s == u.s && q.x == u.x && q.y == u.y && q.z == u.z

# Angle
function Base.angle(q::UnitQuaternion)
    min(2*atan(vecnorm(q), q.s), 2*atan(vecnorm(q), -q.s))
end

# Exponentials and Logarithms
function exp(q::UnitQuaternion{T,D}) where {T,D}
    θ = vecnorm(q)
    sθ,cθ = sincos(θ)
    es = exp(q.s)
    M = es*sθ/θ
    UnitQuaternion{T,D}(es*cθ, q.x*M, q.y*M, q.z*M)
end

function expm(ϕ::SVector{3,T}) where T
    θ = norm(ϕ)
    sθ,cθ = sincos(θ/2)
    M = 0.5*sinc(θ/2π)
    UnitQuaternion{T,ExponentialMap}(cθ, ϕ[1]*M, ϕ[2]*M, ϕ[3]*M)
end

function log(q::UnitQuaternion{T,D}, eps=1e-6) where {T,D}
    # Assumes unit quaternion
    θ = vecnorm(q)
    if θ > eps
        M = atan(θ, q.s)/θ
    else
        M = (1-(θ^2/(3q.s^2)))/q.s
    end
    UnitQuaternion{T,D}(0.0, q.x*M, q.y*M, q.z*M)
end

function logm(q::UnitQuaternion{T}) where T
    # Assumes unit quaternion
    q = log(q)
    SVector{3,T}(2*q.x, 2*q.y, 2*q.z)
end

# Composition
""" Quternion Composition
Equivalent to `Lmult(q) * SVector(w)` or `Rmult(w) * SVector(q)`
"""
function (*)(q::UnitQuaternion{T1,D1}, w::UnitQuaternion{T2,D2}) where {T1,T2,D1,D2}
    T = promote_type(T1, T2)
    D = D2
    UnitQuaternion{T,D}(q.s * w.s - q.x * w.x - q.y * w.y - q.z * w.z,
                        q.s * w.x + q.x * w.s + q.y * w.z - q.z * w.y,
                        q.s * w.y - q.x * w.z + q.y * w.s + q.z * w.x,
                        q.s * w.z + q.x * w.y - q.y * w.x + q.z * w.s)
end

``` Rotate a vector
Equivalent to `Hmat()' Lmult(q) * Rmult(q)' Hmat() * r`
```
function Base.:*(q::UnitQuaternion{Tq}, r::SVector{3}) where Tq
    qo = (-q.x  * r[1] - q.y * r[2] - q.z * r[3],
           q.s  * r[1] + q.y * r[3] - q.z * r[2],
           q.s  * r[2] - q.x * r[3] + q.z * r[1],
           q.s  * r[3] + q.x * r[2] - q.y * r[1])

   T = promote_type(Tq, eltype(r))

   return similar_type(r, T)(-qo[1] * q.x + qo[2] * q.s - qo[3] * q.z + qo[4] * q.y,
                             -qo[1] * q.y + qo[2] * q.z + qo[3] * q.s - qo[4] * q.x,
                             -qo[1] * q.z - qo[2] * q.y + qo[3] * q.x + qo[4] * q.s)
end

"Scalar multiplication"
function (*)(q::Q, s::Real) where Q<:UnitQuaternion
    return Q(q.s*s, q.x*s, q.y*s, q.z*s)
end
(*)(s::Real, q::Q) = q*s



"Inverted composition. Equivalent to inv(q1)*q2"
(\)(q1::UnitQuaternion, q2::UnitQuaternion) = conj(q1)*q2

"Inverted composition. Equivalent to q1*inv(q2)"
(/)(q1::UnitQuaternion, q2::UnitQuaternion) = q1*conj(q2)

"Inverted rotation. Equivalent to inv(q)*r"
(\)(q::UnitQuaternion, r::SVector{3}) = conj(q)*r


# ~~~~~~~~~~~~~~~ Quaternion Differences ~~~~~~~~~~~~~~~ #
function (⊖)(q::UnitQuaternion{T,D}, q0::UnitQuaternion) where {T,D}
    D(q0\q)
end

function (⊖)(q::UnitQuaternion{T,IdentityMap}, q0::UnitQuaternion) where {T}
    SVector(q) - SVector(q0)
    # return SVector(q0\q)
end


# ~~~~~~~~~~~~~~~ Linear Algebraic Conversions ~~~~~~~~~~~~~~~ #
"Lmult(q2)q1 returns a vector equivalent to q2*q1 (quaternion multiplication)"
function Lmult(q::UnitQuaternion)
    @SMatrix [
        q.s -q.x -q.y -q.z;
        q.x  q.s -q.z  q.y;
        q.y  q.z  q.s -q.x;
        q.z -q.y  q.x  q.s;
    ]
end
Lmult(q::SVector{4}) = Lmult(UnitQuaternion(q))

"Rmult(q1)q2 return a vector equivalent to q2*q1 (quaternion multiplication)"
function Rmult(q::UnitQuaternion)
    @SMatrix [
        q.s -q.x -q.y -q.z;
        q.x  q.s  q.z -q.y;
        q.y -q.z  q.s  q.x;
        q.z  q.y -q.x  q.s;
    ]
end
Rmult(q::SVector{4}) = Rmult(UnitQuaternion(q))

"Tmat()q return a vector equivalent to inv(q)"
function Tmat()
    @SMatrix [
        1  0  0  0;
        0 -1  0  0;
        0  0 -1  0;
        0  0  0 -1;
    ]
end

"Vmat(q) SVector(q) returns the imaginary
    (vector) part of the quaternion q (equivalent to vector(q))"
function Vmat()
    @SMatrix [
        0 1 0 0;
        0 0 1 0;
        0 0 0 1
    ]
end

""" `Hmat()*r` or `Hmat(r)` converts `r` into a pure quaternion, where r is 3-dimensional"""
function Hmat()
    @SMatrix [
        0 0 0;
        1 0 0;
        0 1 0;
        0 0 1.;
    ]
end

function Hmat(r)
    @assert length(r) == 3
    @SVector [0,r[1],r[2],r[3]]
end


# ~~~~~~~~~~~~~~~ Use Jacobians ~~~~~~~~~~~~~~~ #
"""
Jacobian of `Lmult(q) QuatMap(ϕ)`, when ϕ is near zero. Useful for converting Jacobians from R⁴ to R³ and
    correctly account for unit norm constraint. Jacobians for different
    differential quaternion parameterization are the same up to a constant.
"""
function ∇differential(q::UnitQuaternion)
    1.0 * @SMatrix [
        -q.x -q.y -q.z;
         q.s -q.z  q.y;
         q.z  q.s -q.x;
        -q.y  q.x  q.s;
    ]
end

"Jacobian of `(∂/∂ϕ Lmult(q) QuatMap(ϕ))`b, evaluated at ϕ=0"
function ∇²differential(q::UnitQuaternion, b::SVector{4})
    b1 = -SVector(q)'b
    Diagonal(@SVector fill(b1,3))
end

"Jacobian of q*r with respect to the quaternion"
function ∇rotate(q::UnitQuaternion{T,D}, r::SVector{3}) where {T,D}
    rhat = UnitQuaternion{D}(r)
    R = Rmult(q)
    2Vmat()*Rmult(q)'Rmult(rhat)
end

"Jacobian of q2*q1 with respect to q1"
function ∇composition1(q2::UnitQuaternion, q1::UnitQuaternion)
    Lmult(q2)
end

"Jacobian of q2*q1 with respect to q2"
function ∇composition2(q2::UnitQuaternion, q1::UnitQuaternion)
    Rmult(q1)
end


abstract type VectorPart <: QuatMap end
abstract type ExponentialMap <: QuatMap end
abstract type MRPMap <: QuatMap end
abstract type CayleyMap <: QuatMap end
abstract type IdentityMap <: QuatMap end

const DEFAULT_QMAP = CayleyMap

# Scalings
@inline scaling(::Type{ExponentialMap}) = 0.5
@inline scaling(::Type{VectorPart}) = 1.0
@inline scaling(::Type{CayleyMap}) = 1.0
@inline scaling(::Type{MRPMap}) = 2.0

# Quaternion Maps
"""```julia
(::Type{<:QuatMap})(ϕ)
```
Returns a `UnitQuaternion` given a three-parameter quaternion error via the specified map
"""
(::Type{ExponentialMap})(ϕ) = expm(ϕ/scaling(ExponentialMap))

function (::Type{VectorPart})(v)
    μ = 1/scaling(VectorPart)
    UnitQuaternion{VectorPart}(sqrt(1-μ^2*v'v), μ*v[1], μ*v[2], μ*v[3])
end

function (::Type{CayleyMap})(g)
    g /= scaling(CayleyMap)
    M = 1/sqrt(1+g'g)
    UnitQuaternion{CayleyMap}(M, M*g[1], M*g[2], M*g[3])
end

function (::Type{MRPMap})(p)
    p /= scaling(MRPMap)
    n2 = p'p
    M = 2/(1+n2)
    UnitQuaternion{MRPMap}((1-n2)/(1+n2), M*p[1], M*p[2], M*p[3])
end

(::Type{IdentityMap})(q) = UnitQuaternion{IdentityMap}(q[1], q[2], q[3], q[4])


# Quaternion Map Jacobians
"""```julia
jacobian(::Type{<:QuatMap}, ϕ)
```
Jacobian of the quaternion map that takes a three-dimensional vector `\phi` and returns a
    unit quaternion.
Returns a 4x3 Static Matrix

For all the maps (except the `IdentityMap`)
`jacobian(::Type{<:QuatMap}, zeros(3)) = [0; I] = Hmat()'`
"""
function jacobian(::Type{ExponentialMap},ϕ, eps=1e-5)
    μ = 1/scaling(ExponentialMap)
    θ = norm(ϕ)
    cθ = cos(μ*θ/2)
    sincθ = sinc(μ*θ/2π)
    if θ < eps
        0.5*μ*[-0.5*μ*sincθ*ϕ'; sincθ*I + (cθ - sincθ)*ϕ*ϕ']
    else
        0.5*μ*[-0.5*μ*sincθ*ϕ'; sincθ*I + (cθ - sincθ)*ϕ*ϕ'/(ϕ'ϕ)]
    end
end

function jacobian(::Type{VectorPart}, v)
    μ = 1/scaling(VectorPart)
    μ2 = μ*μ
    M = -μ2/sqrt(1-μ2*v'v)
    @SMatrix [v[1]*M v[2]*M v[3]*M;
              μ 0 0;
              0 μ 0;
              0 0 μ]
end

function jacobian(::Type{CayleyMap}, g)
    μ = 1/scaling(CayleyMap)
    μ2 = μ*μ
    n = 1+μ2*g'g
    ni = 1/n
    μ*[-μ*g'; -μ2*g*g' + I*n]*ni*sqrt(ni)
end

function jacobian(::Type{MRPMap}, p)
    μ = 1/scaling(MRPMap)
    μ2 = μ*μ
    n = 1+μ2*p'p
    2*[-2*μ2*p'; I*μ*n - 2*μ*μ2*p*p']/n^2
end

jacobian(::Type{IdentityMap}, q) = I



############################################################################################
#                             INVERSE RETRACTION MAPS
############################################################################################
""" ```julia
(::Type{<:QuatMap})(q::UnitQuaternion)
```
Inverse quaternion map.
Maps a unit quaternion to the 3D quaternion error through the specified map.
"""
(::Type{ExponentialMap})(q::UnitQuaternion) = scaling(ExponentialMap)*logm(q)

(::Type{VectorPart})(q::UnitQuaternion) = scaling(VectorPart)*vector(q)

(::Type{CayleyMap})(q::UnitQuaternion) = scaling(CayleyMap) * vector(q)/q.s

(::Type{MRPMap})(q::UnitQuaternion) = scaling(MRPMap)*vector(q)/(1+q.s)

(::Type{IdentityMap})(q::UnitQuaternion) = SVector(q)


# ~~~~~~~~~~~~~~~ Inverse map Jacobians ~~~~~~~~~~~~~~~ #
"""```julia
jacobian(::Type{<:QuatMap}, q::UnitQuaternion)
```
Jacobian of the inverse quaternion map, returning a 3x4 matrix.
For all maps: `jacobian(::Type{<:QuatMap}, UnitQuaternion(I)) = [0 I] = Hmat()'`
"""
function jacobian(::Type{ExponentialMap}, q::UnitQuaternion, eps=1e-5)
    μ = scaling(ExponentialMap)
    s = scalar(q)
    v = vector(q)
    θ2 = v'v
    θ = sqrt(θ2)
    datan = 1/(θ2 + s^2)
    ds = -datan*v

    if θ < eps
        return 2*μ*[ds (v*v' + I)/s]
    else
        atanθ = atan(θ,s)
        dv = ((s*datan - atanθ/θ)v*v'/θ + atanθ*I )/θ
        d0 = ((s*datan - atanθ/θ)v*v'/θ^2 + atanθ/θ*I )
        d1 = (s*datan - atanθ/θ)
        d2 = v*v'/θ2
        d3 = atanθ/θ * I
        return 2*μ*[ds dv]
    end
end


function jacobian(::Type{VectorPart}, q::UnitQuaternion)
    μ = scaling(VectorPart)
    return @SMatrix [0. μ 0 0;
                     0. 0 μ 0;
                     0. 0 0 μ]
end


function jacobian(::Type{CayleyMap}, q::UnitQuaternion)
    μ = scaling(CayleyMap)
    si = 1/q.s
    return μ*@SMatrix [-si^2*q.x si 0 0;
                       -si^2*q.y 0 si 0;
                       -si^2*q.z 0 0 si]
end


function jacobian(::Type{MRPMap}, q::UnitQuaternion)
    μ = scaling(MRPMap)
    si = 1/(1+q.s)
    return μ*@SMatrix [-si^2*q.x si 0 0;
                       -si^2*q.y 0 si 0;
                       -si^2*q.z 0 0 si]
end


jacobian(::Type{IdentityMap}, q::UnitQuaternion) = I


# ~~~~~~~~~~~~~~~ Inverse map Jacobian derivative ~~~~~~~~~~~~~~~ #
""" ```julia
∇jacobian(::Type{<:QuatMap}, q::UnitQuaternion, b::SVector{3})
```
Jacobian of G(q)'b, where G(q) = jacobian(::Type{<:QuatMap}, q),
    b is a 3-element vector
"""
function ∇jacobian(::Type{ExponentialMap}, q::UnitQuaternion, b::SVector{3}, eps=1e-5)
    μ = scaling(ExponentialMap)
    s = scalar(q)
    v = vector(q)
    θ2 = v'v
    θ = sqrt(θ2)
    datan = 1/(θ2 + s^2)
    ds = -datan*v

    if θ < eps
        # return 2*μ*[b'ds; (v*v'b + b)/s]
        return 2*μ*[b'*(datan^2*2s*v) -b'datan*I;
                    -(v*v'b +b)/s^2 (I*(v'b) + v*b')/s]
    else
        dsds = 2b's*datan^2*v
        dsdv = b'*(-datan*I + 2datan^2*v*v')

        atanθ = atan(θ,s)
        d1 = (s*datan - atanθ/θ)
        d2 = v*v'b/θ2
        d3 = atanθ/θ*b
        d1ds = (datan - 2s^2*datan^2 + datan)
        dvds = d1ds*d2 - datan*b

        d1dv =  (-2s*datan^2*v' - s*datan*v'/θ^2 + atanθ/θ^3*v')
        d2dv = (I*(v'b) + v*b')/θ2 - 2(v*v'b)/θ^4 * v'
        d3dv = b*(s*datan*v'/θ^2 - atanθ/θ^3*v')
        dvdv = d2*d1dv + d1*d2dv + d3dv

        # return 2*μ*[ds'b; dv'b]
        return 2*μ*@SMatrix [
            dsds    dsdv[1] dsdv[2] dsdv[3];
            dvds[1] dvdv[1] dvdv[4] dvdv[7];
            dvds[2] dvdv[2] dvdv[5] dvdv[8];
            dvds[3] dvdv[3] dvdv[6] dvdv[9];
        ]
    end
end

function ∇jacobian(::Type{CayleyMap}, q::UnitQuaternion, b::SVector{3})
    μ = scaling(CayleyMap)
    si = 1/q.s
    v = vector(q)
    μ*@SMatrix [
        2*si^3*(v'b) -si^2*b[1] -si^2*b[2] -si^2*b[3];
       -si^2*b[1] 0 0 0;
       -si^2*b[2] 0 0 0;
       -si^2*b[3] 0 0 0;
    ]
end

function ∇jacobian(::Type{MRPMap}, q::UnitQuaternion, b::SVector{3})
    μ = scaling(MRPMap)
    si = 1/(1+q.s)
    v = vector(q)
    μ * @SMatrix [
        2*si^3*(v'b) -si^2*b[1] -si^2*b[2] -si^2*b[3];
       -si^2*b[1] 0 0 0;
       -si^2*b[2] 0 0 0;
       -si^2*b[3] 0 0 0;
    ]
end

function ∇jacobian(::Type{VectorPart}, q::UnitQuaternion, b::SVector{3})
    μ = scaling(VectorPart)
    @SMatrix zeros(4,4)
end


inverse_map_jacobian(q::R) where R<:Rotation = I
inverse_map_jacobian(q::UnitQuaternion{T,D}) where {T,D} = jacobian(D,q)

inverse_map_∇jacobian(q::R, b::SVector{3}) where R<:Rotation = I*0
inverse_map_∇jacobian(q::UnitQuaternion{T,D}, b::SVector{3}) where {T,D} = ∇jacobian(D, q, b)

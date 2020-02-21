
"""```julia
RodriguesParam{T}
```
Rodrigues parameters are a three-dimensional parameterization of rotations.
They have a singularity at 180° but do not inherit the sign ambiguities of quaternions
or MRPs
"""
struct RodriguesParam{T} <: Rotation{T}
    x::T
    y::T
    z::T
end

# ~~~~~~~~~~~~~~~ Constructors ~~~~~~~~~~~~~~~ #
RodriguesParam(g::SVector{3,T}) where T = RodriguesParam{T}(g[1], g[2], g[3])
(::Type{<:RodriguesParam})(::Type{T},x,y,z) where T = RodriguesParam{T}(T(x),T(y),T(z))
(::Type{<:RodriguesParam})(p::RodriguesParam) = p


# ~~~~~~~~~~~~~~~ Conversions ~~~~~~~~~~~~~~~ #
SVector(g::RodriguesParam{T}) where T = SVector{3,T}(g.x, g.y, g.z)


# ~~~~~~~~~~~~~~~ Initializers ~~~~~~~~~~~~~~~ #
Base.rand(::Type{<:RodriguesParam}) = RodriguesParam(rand(UnitQuaternion))
Base.zero(::Type{<:RodriguesParam}) = RodriguesParam(0.0, 0.0, 0.0)

# ~~~~~~~~~~~~~~~ Math Operations ~~~~~~~~~~~~~~~ #
LinearAlgebra.norm(g::RodriguesParam) = sqrt(g.x^2 + g.y^2 + q.z^2)
LinearAlgebra.norm2(g::RodriguesParam) = g.x^2 + g.y^2 + g.z^2
Base.angle(g::RodriguesParam) = angle(UnitQuaternion(g))

function (≈)(g2::RodriguesParam, g1::RodriguesParam)
    g2.x ≈ g1.x && g2.y ≈ g1.y && g2.z ≈ g1.z
end

function (==)(g2::RodriguesParam, g1::RodriguesParam)
    g2.x ≈ g1.x && g2.y ≈ g1.y && g2.z ≈ g1.z
end

# ~~~~~~~~~~~~~~~ Composition ~~~~~~~~~~~~~~~ #
function (*)(g2::RodriguesParam, g1::RodriguesParam)
    g2 = SVector(g2)
    g1 = SVector(g1)
    RodriguesParam((g2+g1 + g2 × g1)/(1-g2'g1))
end

function (\)(g1::RodriguesParam, g2::RodriguesParam)
    g2 = SVector(g2)
    g1 = SVector(g1)
    RodriguesParam((g2-g1 + g2 × g1)/(1+g1'g2))
end

function (/)(g1::RodriguesParam, g2::RodriguesParam)
    g2 = SVector(g2)
    g1 = SVector(g1)
    RodriguesParam((g1-g2 + g2 × g1)/(1+g1'g2))
end


# ~~~~~~~~~~~~~~~ Rotation ~~~~~~~~~~~~~~~ #
(*)(g::RodriguesParam, r::SVector{3}) = UnitQuaternion(g)*r
(\)(g::RodriguesParam, r::SVector{3}) = inv(UnitQuaternion(g))*r


# ~~~~~~~~~~~~~~~ Kinematics ~~~~~~~~~~~~~~~ #
function kinematics(g::RodriguesParam, ω::SVector{3})
    g = SVector(g)
    0.5*(I + skew(g) + g*g')*ω
end





function ∇rotate(g0::RodriguesParam, r)
    g = SVector(g0)
    ghat = skew(g)
    n1 = 1/(1 + g'g)
    gxr = cross(g,r) + r
    d1 = ghat*gxr * -2*n1^2 * g'
    d2 = -(ghat*skew(r) + skew(gxr))*n1
    return 2d1 + 2d2
end

function ∇composition1(g2::RodriguesParam, g1::RodriguesParam)
    g2 = SVector(g2)
    g1 = SVector(g1)

    N = g2 + g1 + g2 × g1
    D = 1/(1 - g2'g1)
    (I + skew(g2) + D*N*g2')*D
end

function ∇²composition1(g2::RodriguesParam, g1::RodriguesParam, b::SVector{3})
    g2 = SVector(g2)
    g1 = SVector(g1)

    N = g2 + g1 + g2 × g1  # 3x1
    D = 1/(1 - g2'g1)  # scalar
    dN = I + skew(g2)
    dD = D^2*g2'
    return g2*b'*(N*(2*D*dD) + D^2*dN) + (I - skew(g2))*b*dD
end

function ∇composition2(g2::RodriguesParam, g1::RodriguesParam)
    g2 = SVector(g2)
    g1 = SVector(g1)

    N = g2 + g1 + g2 × g1
    D = 1/(1 - g2'g1)
    (I - skew(g1) + D*N*g1')*D
end

function ∇differential(g::RodriguesParam)
    g = SVector(g)
    (I + skew(g) + g*g')
end

function ∇²differential(g::RodriguesParam, b::SVector{3})
    g = SVector(g)
    return g*b'*(2g*g' + I + skew(g)) + (I - skew(g))*b*g'
end


""" ```
MRP{T} <: Rotation
```
Modified Rodrigues Parameter. Is a 3D parameterization of attitude, and is a sterographic
projection of the 4D unit sphere onto the plane tangent to the negative real pole. They
have a singularity at θ = ±180°.

# Constructors
MRP(x, y, z)
MRP(r::SVector{3})
"""
struct MRP{T} <: Rotation
    x::T
    y::T
    z::T
end

# ~~~~~~~~~~~~~~~ Constructors ~~~~~~~~~~~~~~~ #
MRP(r::SVector{3}) = MRP(r[1], r[2], r[3])
(::Type{<:MRP})(::Type{T},x,y,z) where T = MRP{T}(T(x),T(y),T(z))
MRP{T}(r::SVector{3}) where T = MRP(T(r[1]), T(r[2]), T(r[3]))
MRP{T}(p::MRP) where T = MRP(T(p.x), T(p.y), T(p.z))

# ~~~~~~~~~~~~~~~ Conversions ~~~~~~~~~~~~~~~ #

SVector(p::MRP{T}) where T = SVector{3,T}(p.x, p.y, p.z)


# ~~~~~~~~~~~~~~~ Initializers ~~~~~~~~~~~~~~~ #
Base.rand(::Type{MRP{T}}) where T = MRP(rand(UnitQuaternion{T}))
Base.rand(::Type{MRP}) = MRP(rand(UnitQuaternion))
Base.zero(::Type{MRP{T}}) where T = MRP(zero(T), zero(T), zero(T))
Base.zero(::Type{MRP}) = MRP(0.0, 0.0, 0.0)


# ~~~~~~~~~~~~~~~ Math Operations ~~~~~~~~~~~~~~~ #
LinearAlgebra.norm(p::MRP) = sqrt(p.x^2 + p.y^2 + p.z^2)
LinearAlgebra.norm2(p::MRP) = p.x^2 + p.y^2 + p.z^2

(==)(p2::MRP, p1::MRP) = p2.x == p1.x && p2.y == p1.y && p2.z == p1.z
(≈)(p2::MRP, p1::MRP) = p2.x ≈ p1.x && p2.y ≈ p1.y && p2.z ≈ p1.z

Base.angle(p::MRP) = angle(UnitQuaternion(p))


# ~~~~~~~~~~~~~~~ Composition ~~~~~~~~~~~~~~~ #
function (*)(p2::MRP, p1::MRP)
    p2, p1 = SVector(p2), SVector(p1)
    MRP(((1-p2'p2)*p1 + (1-p1'p1)*p2 - cross(2p1, p2) ) / (1+p1'p1*p2'p2 - 2p1'p2))
end

function (\)(p1::MRP, p2::MRP)
    # fun fact: equivalent to p2 - p1 when either is 0
    p1,p2 = SVector(p1), SVector(p2)
    n1,n2 = p1'p1, p2'p2
    θ = 1/((1+n1)*(1+n2))
    s1,s2 = (1-n1), (1-n2)
    v1 = -2p1
    v2 =  2p2
    s = s1*s2 - v1'v2
    v = s1*v2 + s2*v1 + v1 × v2

    M = θ/(1+θ*s)
    return MRP(v*M)
end

function (/)(p1::MRP, p2::MRP)
    n1,n2 = norm2(p1),   norm2(p2)
    θ = 1/((1+n1)*(1+n2))
    s1,s2 = (1-n1), (1-n2)
    p1,p2 = SVector(p1), SVector(p2)
    v1 =  2p1
    v2 = -2p2
    s = s1*s2 - v1'v2
    v = s1*v2 + s2*v1 + v1 × v2

    M = θ/(1+θ*s)
    MRP(v[1]*M, v[2]*M, v[3]*M)

end




# ~~~~~~~~~~~~~~~ Rotation ~~~~~~~~~~~~~~~ #
(*)(p::MRP, r::SVector) = UnitQuaternion(p)*r
(\)(p::MRP, r::SVector) = UnitQuaternion(p)\r


# ~~~~~~~~~~~~~~~ Kinematics ~~~~~~~~~~~~~~~ #
function kinematics(p::MRP, ω)
    p = SVector(p)
    A = @SMatrix [
        1 + p[1]^2 - p[2]^2 - p[3]^2  2(p[1]*p[2] - p[3])      2(p[1]*p[3] + p[2]);
        2(p[2]*p[1] + p[3])            1-p[1]^2+p[2]^2-p[3]^2   2(p[2]*p[3] - p[1]);
        2(p[3]*p[1] - p[2])            2(p[3]*p[2] + p[1])      1-p[1]^2-p[2]^2+p[3]^2]
    0.25*A*ω
end



# ~~~~~~~~~~~~~~~ Use Jacobians ~~~~~~~~~~~~~~~ #
function ∇rotate(p::MRP, r)
    p = SVector(p)
    4( (1-p'p)*skew(r)*(4p*p'/(1+p'p) - I) - (4/(1+p'p)*skew(p) + I)*2*skew(p)*r*p'
      - 2*(skew(p)*skew(r) + skew(skew(p)*r)))/(1+p'p)^2
end

function ∇composition2(p2::MRP, p1::MRP)
    p2,p1 = SVector(p2), SVector(p1)
    n1 = p1'p1
    n2 = p2'p2
    D = 1 / (1+n1*n2 - 2p1'p2)
    d1 = (-2p1*p2' + (1-n1)*I - skew(2p1) ) * D
    d2 = -((1-n2)*p1 + (1-n1)*p2 - cross(2p1, p2) ) * D^2 *
        (2p2*n1 - 2p1)'
    return d1 + d2
end

function ∇composition1(p2::MRP, p1::MRP)
    p2,p1 = SVector(p2), SVector(p1)
    n1 = p1'p1
    n2 = p2'p2
    D = 1 / (1+n1*n2 - 2p1'p2)
    d1 = ((1-n2)*I + -2p2*p1' + 2skew(p2) ) * D
    d2 = -((1-n2)*p1 + (1-n1)*p2 - cross(2p1, p2) ) * D^2 *
        (2p1*n2 - 2p2)'
    d1 + d2
end

function ∇²composition1(p2::MRP, p1::MRP, b::SVector{3})
    p2,p1 = SVector(p2), SVector(p1)
    n1 = p1'p1
    n2 = p2'p2
    D = 1 / (1+n1*n2 - 2p1'p2)  # scalar
    dD = -D^2 * (n2*2p1 - 2p2)  # 3x1
    A = -((1-n2)*p1 + (1-n1)*p2 - cross(2p1, p2) )  # 3x1
    dA = -I*(1-n2) + 2p2*p1' - 2skew(p2)  # 3x3
    B = 2(p1*n2 -  p2)'b  # scalar
    dB = 2n2*b  # 3x1
    d1 = -2p2*b' * D + ((1-n2)*b + -2p2*p1'b + 2skew(p2)*b )*dD'
    d2 = dA * D^2 * B +
         A * 2D * dD' * B +
         A * D^2 * dB'
    return d1 + d2
end

function ∇differential(p::MRP)
    p = SVector(p)
    n = p'p
    # p = SVector(p)
    # (1-n)I + 2(skew(p) + p*p')
    # @SMatrix [n + 2p.x^2      2(p.x*p.y-p.z)  2(p.x*p.z+p.y);
    #           2(p.y*p.x+p.z)  n + 2p.y^2      2(p.y*p.z-p.x);
    #           2(p.z*p.x-p.y)  2(p.z*p.y+p.x)  n + 2p.z^2]
    #
    # p2 = SVector(p)
    # n2 = p2'p2
    return (1-n)*I + 2(skew(p) + p*p')
end


function ∇²differential(p2::MRP, b::SVector{3})
    p2 = SVector(p2)
    n2 = p2'p2
    A = -p2  # 3x1
    B = -2p2  # 3x1
    D = 1
    dD = 2p2

    dA = -I*(1-n2) - 2skew(p2)  # 3x3
    dB = 2n2*I  # 3x3

    d1 = (-2p2'b*I*D) - (dA'b * dD')
    d2 = dB * A'b * D^2 +
        (-(1-n2)*b + (skew(2p2))*b)*B' * D^2 +
        B*A'b * 2D * dD'
    d1 + d2'
end

""" Jacobian of `p1\\p2` wrt `p2`
"""
function ∇err(p1::MRP, p2::MRP)
    n1,n2 = norm2(p1),   norm2(p2)
    θ = 1/((1+n1)*(1+n2))
    s1,s2 = (1-n1), (1-n2)
    p1,p2 = SVector(p1), SVector(p2)
    v1 = -2p1
    v2 =  2p2
    s = s1*s2 - v1'v2
    v = s1*v2 + s2*v1 + v1 × v2
    dsdp = -2s1*p2 - 2v1
    dvdp = 2s1*I + -2v1*p2' + 2skew(v1)
    dθdp = -θ^2*(1+n1)*2p2

    M = θ/(1+θ*s)
    dMdp = 1/(1+θ*s)*dθdp - θ/(1+θ*s)^2*(dθdp*s + θ*dsdp)
    return dvdp*M + v*dMdp'
end

""" Jacobian of `(∂/∂p p1\\p2)'b` wrt `p2`
"""
function ∇²err(p1::MRP, p2::MRP, b::SVector{3})
    n1,n2 = norm2(p1),   norm2(p2)
    θ = 1/((1+n1)*(1+n2))
    s1,s2 = (1-n1), (1-n2)
    p1,p2 = SVector(p1), SVector(p2)
    v1 = -2p1
    v2 =  2p2
    s = s1*s2 - v1'v2
    v = s1*v2 + s2*v1 + v1 × v2

    dsdp = -2s1*p2 - 2v1  # 3x1
    dsdp2 = -2s1*I  # 3x3

    dvdp = 2s1*b + -2p2*v1'b - 2skew(v1)*b
    dvdp2 = -I*2v1'b

    dθdp = -θ^2*(1+n1)*2p2  # 3x1
    dθdp2 = -2θ*(1+n1)*2p2*dθdp' - θ^2*(1+n1)*2I # 3x3

    M = θ/(1+θ*s)  # scalar
    dMdp = 1/(1+θ*s)*dθdp - θ/(1+θ*s)^2*(dθdp*s + θ*dsdp) # 3x1
    dM2 = θ/(1+θ*s)^2  # scalar
    dM3 = dθdp*s + θ*dsdp  # 3x1
    dM2dp = dθdp'/(1+θ*s)^2 - 2θ/(1+θ*s)^3 * (dθdp*s + θ*dsdp)'
    dM3dp = dθdp2*s + dθdp*dsdp' + dsdp*dθdp' + θ*dsdp2

    dMdp2 = -1/(1+θ*s)^2*dθdp*(dθdp*s + dsdp*θ)' + 1/(1+θ*s)*dθdp2  # good
    dMdp2 -= dM3*dM2dp + dM2*dM3dp

    vb = s1*v2'b + s2*v1'b + b'skew(v1)*v2  # scalar
    vpdp = s1*2b' - 2p2' * (v1'b)  + b'skew(v1)*2 # good
    # vpdp = s1*2b' - 2p2 * (v1'b)

    d1 = M*dvdp2 + dvdp*dMdp'
    d2 = dMdp2*vb + dMdp*vpdp
    return d1 + d2
end

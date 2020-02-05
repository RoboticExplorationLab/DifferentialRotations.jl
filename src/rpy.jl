
"""
Roll-pitch-yaw Euler angles.
    Follows convetion from "The GRASP multiple micro-UAV testbed" [Michael et al, 2010]
"""
struct RPY{T} <: Rotation
    ϕ::T  # roll
    θ::T  # pitch
    ψ::T  # yaw
end

roll(e::RPY) = e.ϕ
pitch(e::RPY) = e.θ
yaw(e::RPY) = e.ψ

# ~~~~~~~~~~~~~~~ Constructors ~~~~~~~~~~~~~~~ #
RPY(e::SVector{3,T}) where T = RPY{T}(e[1], e[2], e[3])
RPY(R::SMatrix{3,3,T}) where T =  RPY(rotmat_to_rpy(R))
RPY(q::UnitQuaternion) = RPY(rotmat(q))
RPY(p::MRP) = RPY(rotmat(p))
(::Type{<:RPY})(e::RPY) = e
(::Type{<:RPY})(::Type{T},x,y,z) where T = RPY{T}(T(x),T(y),T(z))
function RPY(ϕ::T1,θ::T2,ψ::T3) where {T1,T2,T3}
    T = promote_type(T1,T2)
    T = promote_type(T,T3)
    RPY(T(ϕ), T(θ), T(ψ))
end

# ~~~~~~~~~~~~~~~ Conversion ~~~~~~~~~~~~~~~ #
SVector(e::RPY{T}) where T = SVector{3,T}(e.ϕ, e.θ, e.ψ)
@inline rotmat(e::RPY) = rotmat(e.ϕ, e.θ, e.ψ)

function rotmat(ϕ, θ, ψ)
    # Equivalent to RotX(e[1])*RotY(e[2])*RotZ(e[3])
    sϕ,cϕ = sincos(ϕ)
    sθ,cθ = sincos(θ)
    sψ,cψ = sincos(ψ)
    # A = @SMatrix [
    #     cθ*cψ          -cθ*sψ              sθ;
    #     sϕ*sθ*cψ+cϕ*sψ -sϕ*sθ*sψ + cϕ*cψ  -cθ*sϕ;
    #    -cϕ*sθ*cψ+sϕ*sψ  cϕ*sθ*sψ + sϕ*cψ   cθ*cϕ
    # ]
    A = @SMatrix [
        cψ*cθ - sϕ*sψ*sθ   -cϕ*sψ  cψ*sθ + cθ*sϕ*sψ;
        cθ*sψ + cψ*sϕ*sθ    cϕ*cψ  sψ*sθ - cψ*cθ*sϕ;
        -cϕ*sθ              sϕ          cϕ*cθ;
    ]
end

function rotmat_to_rpy(R::SMatrix{3,3,T}) where T
    # ψ = atan(-R[1,2], R[1,1])
    # ϕ = atan(-R[2,3], R[3,3])
    # θ = asin(R[1,3])
    θ = atan(-R[3,1], R[3,3])
    ψ = atan(-R[1,2], R[2,2])
    ϕ = asin(clamp(R[3,2],-1,1))
    return SVector{3,T}(ϕ, θ, ψ)
end

function from_rotmat(R::SMatrix{3,3,T}) where T
    ϕ,θ,ψ = rotmat_to_rpy(R)
    return RPY(ϕ, θ, ψ)
end

@inline RPY(R::SMatrix{3,3}) = from_rotmat(R)


# ~~~~~~~~~~~~~~~ Initializers ~~~~~~~~~~~~~~~ #
Base.rand(::Type{RPY{T}}) where T = RPY(rand(UnitQuaternion{T}))
Base.rand(::Type{RPY}) = RPY(rand(UnitQuaternion))
Base.zero(::Type{RPY{T}}) where T = RPY(zero(T), zero(T), zero(T))
Base.zero(::Type{RPY}) = RPY(0.0, 0.0, 0.0)


# ~~~~~~~~~~~~~~~ Math Operation ~~~~~~~~~~~~~~~ #
Base.angle(e::RPY) = angle(UnitQuaternion(e))

(≈)(e1::RPY, e2::RPY) = rotmat(e1) ≈ rotmat(e2)
(==)(e1::RPY, e2::RPY) = rotmat(e1) == rotmat(e2)


# ~~~~~~~~~~~~~~~ Rotation ~~~~~~~~~~~~~~~ #
(*)(e::RPY, r::SVector{3}) = rotmat(e)*r
(\)(e::RPY, r::SVector{3}) = rotmat(e)'r


# ~~~~~~~~~~~~~~~ Composition ~~~~~~~~~~~~~~~ #
function (*)(e2::RPY, e1::RPY)
    from_rotmat(rotmat(e2)*rotmat(e1))
end

function (\)(e1::RPY, e2::RPY)
    from_rotmat(rotmat(e1)'rotmat(e2))
end

function (/)(e1::RPY, e2::RPY)
    from_rotmat(rotmat(e1)*rotmat(e2)')
end


# ~~~~~~~~~~~~~~~ Kinematics ~~~~~~~~~~~~~~~ #
function kinematics(e::RPY, ω::SVector{3})
    sθ,cθ = sincos(e.θ)
    sϕ,cϕ = sincos(e.ϕ)
    A = @SMatrix [
        cθ 0 -cϕ*sθ;
        0  1  sϕ;
        sθ 0  cϕ*cθ
    ]
    A\ω
end

# ~~~~~~~~~~~~~~~ Useful Jacobians ~~~~~~~~~~~~~~~ #
function ∇rotate(e::RPY, r::SVector{3})
    rotate(e) = RPY(e)*r
    ForwardDiff.jacobian(rotate, SVector(e))
end

function ∇composition1(e2::RPY, e1::RPY)
    R2 = rotmat(e2)
    rotate(e) = rotmat_to_rpy(R2*rotmat(e[1],e[2],e[3]))
    ForwardDiff.jacobian(rotate,SVector(e1))
end

function ∇composition2(e2::RPY, e1::RPY)
    R1 = rotmat(e1)
    rotate(e) = rotmat_to_rpy(rotmat(e[1],e[2],e[3])*R1)
    ForwardDiff.jacobian(rotate,SVector(e2))
end

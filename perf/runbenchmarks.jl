using DifferentialRotations
using BenchmarkTools
using Rotations
using Quaternions
using StaticArrays
using Statistics
using Random
using Test

# ~~~~~~~~~~~~~~~~~ Set up Comparators ~~~~~~~~~~~~~~~~~~~ #
import Base: ==, ≈
import DifferentialRotations.vector

# Create constructors between different reps
@inline StaticArrays.SVector(q::Quaternion) = SVector{4}(q.s, q.v1, q.v2, q.v3)
@inline vector(q::Quaternion) = SVector{3}(q.v1, q.v2, q.v3)
Rotations.Quat(q::Quaternion) = Rotations.Quat(q.s, q.v1, q.v2, q.v3)
DifferentialRotations.UnitQuaternion(q::Quaternion) = UnitQuaternion(q.s, q.v1, q.v2, q.v3)
StaticArrays.SVector(q::Quaternion) = SVector{4}(q.s, q.v1, q.v2, q.v3)

# Create multiply method for Quaternions.jl type
function Base.:*(q::Quaternion{Tq}, r::SVector{3}) where Tq
    qo = (-q.v1 * r[1] - q.v2 * r[2] - q.v3 * r[3],
           q.s  * r[1] + q.v2 * r[3] - q.v3 * r[2],
           q.s  * r[2] - q.v1 * r[3] + q.v3 * r[1],
           q.s  * r[3] + q.v1 * r[2] - q.v2 * r[1])

   T = promote_type(Tq, eltype(r))

   return similar_type(r, T)(-qo[1] * q.v1 + qo[2] * q.s  - qo[3] * q.v3 + qo[4] * q.v2,
                             -qo[1] * q.v2 + qo[2] * q.v3 + qo[3] * q.s  - qo[4] * q.v1,
                             -qo[1] * q.v3 - qo[2] * q.v2 + qo[3] * q.v1 + qo[4] * q.s)
end

# Set up comparators
(==)(q::Quaternion, u::UnitQuaternion) = q.s == u.s && q.v1 == u.x && q.v2 == u.y && q.v3 == u.z
(==)(u::UnitQuaternion, q::Quaternion) = q == u
(==)(q::Quat, u::UnitQuaternion) = q.w == u.s && q.x == u.x && q.y == u.y && q.z == u.z
(==)(u::UnitQuaternion, q::Quat) = q == u
(==)(q::Quaternion, u::Quat) = q.s == u.w && q.v1 == u.x && q.v2 == u.y && q.v3 == u.z
(==)(u::Quat, q::Quaternion) = q == u

(≈)(q::Quaternion, u::UnitQuaternion) = q.s ≈ u.s && q.v1 ≈ u.x && q.v2 ≈ u.y && q.v3 ≈ u.z
(≈)(u::UnitQuaternion, q::Quaternion) = q ≈ u
(≈)(q::Quat, u::UnitQuaternion) = q.w ≈ u.s && q.x ≈ u.x && q.y ≈ u.y && q.z ≈ u.z
(≈)(u::UnitQuaternion, q::Quat) = q ≈ u
(≈)(q::Quaternion, u::Quat) = q.s ≈ u.w && q.v1 ≈ u.x && q.v2 ≈ u.y && q.v3 ≈ u.z
(≈)(u::Quat, q::Quaternion) = q ≈ u


# ~~~~~~~~~~~~~~~~~~~~ Create two of each type of Quaternion ~~~~~~~~~~~~~~~~~~~~ #
Random.seed!(1)
q1 = nquatrand()
q2 = nquatrand()

r1 = Rotations.Quat(q1)
r2 = Rotations.Quat(q2)

u1 = UnitQuaternion(q1)
u2 = UnitQuaternion(q2)

# Test Equality
@test q1 == u1
@test u1 == q1
@test r1 == u1
@test u1 == r1
@test q1 == r1
@test r1 == q1

# Test composition
@test q1*q2 ≈ r1*r2
@test q1*q2 ≈ u1*u2
b1 = @benchmark for i = 1:10000; $q1 = $q1*$q2; end
b2 = @benchmark for i = 1:10000; $r1 = $r1*$r2; end
b3 = @benchmark for i = 1:10000; $u1 = $u1*$u2; end
judge(median(b3), median(b1))
judge(median(b3), median(b2))


# Test rotation
r = @SVector rand(3)
@test q1*r ≈ r1*r
@test q1*r ≈ u1*r
b1 = @benchmark for i = 1:10000; $r = $q1*$r; end
b2 = @benchmark for i = 1:10000; $r = $r1*$r; end
b3 = @benchmark for i = 1:10000; $r = $u1*$r; end
judge(median(b3), median(b1))
judge(median(b3), median(b2))

@test exp(q1) ≈ exp(u1)
b1 = @benchmark exp($q1)
b2 = @benchmark exp($u1)
judge(median(b2),median(b1))

@test log(q1) ≈ log(u1)
b1 = @benchmark log($q1)
b2 = @benchmark log($u1)
judge(median(b2),median(b1))

ω = @SVector rand(3)
DifferentialRotations.kinematics(q::Quaternion, ω) =
        SVector(0.5*q*Quaternion(0, ω[1], ω[2], ω[3]))
@test kinematics(u1, ω) ≈ kinematics(q1, ω)
b1 = @benchmark kinematics($q1,$ω)
b2 = @benchmark kinematics($u1,$ω)
judge(median(b2),median(b1))

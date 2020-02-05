import DifferentialRotations: jacobian, ∇rotate, ∇composition1, ∇composition2

q1 = rand(UnitQuaternion)
q2 = rand(UnitQuaternion)
r = @SVector rand(3)
ω = @SVector rand(3)


############################################################################################
#                                 QUATERNIONS
############################################################################################

ϕ = 0.1*@SVector [1,0,0]
@test logm(expm(ϕ)) ≈ ϕ
@test expm(logm(q1)) ≈ q1
@test angle(expm(ϕ)) ≈ 0.1

@test norm(q1 * ExponentialMap(ϕ)) ≈ 1
@test q1 ⊖ q2 isa SVector{3}
@test (q1 * ExponentialMap(ϕ)) ⊖ q1 ≈ ϕ

# Test inverses
q3 = q2*q1
@test q2\q3 ≈ q1
@test q3/q1 ≈ q2
@test inv(q1)*r ≈ q1\r
@test r ≈ q3\(q2*q1*r)

q = q1
rhat = UnitQuaternion(r)
@test q*r ≈ Vmat()*Lmult(q)*Rmult(q)'*Vmat()'r
@test q*r ≈ Vmat()*Lmult(q)*Lmult(rhat)*Tmat()*SVector(q)
@test q*r ≈ Vmat()*Rmult(q)'*Rmult(rhat)*SVector(q)

@test ForwardDiff.jacobian(q->UnitQuaternion{VectorPart}(q)*r,SVector(q)) ≈ ∇rotate(q,r)
# @btime ForwardDiff.jacobian(q->UnitQuaternion(q)*$r,SVector($q))
# @btime ∇rotate($q,$r)

@test ForwardDiff.jacobian(q->SVector(q2*UnitQuaternion{VectorPart}(q)),SVector(q1)) ≈
    ∇composition1(q2,q1)
@test ForwardDiff.jacobian(q->SVector(UnitQuaternion{VectorPart}(q)*q1),SVector(q2)) ≈
    ∇composition2(q2,q1)
# @btime ForwardDiff.jacobian(q->SVector($u2*UnitQuaternion(q)),SVector($u1))
# @btime ∇composition1($u2,$u1)
# @btime ForwardDiff.jacobian(q->SVector(UnitQuaternion(q)*$u1),SVector($u2))
# @btime ∇composition2($u2,$u1)

@test Lmult(q) ≈ ∇composition1(q,q2)

ϕ = @SVector zeros(3)
@test DifferentialRotations.∇differential(q) ≈ Lmult(q)*jacobian(VectorPart,ϕ)
@test DifferentialRotations.∇differential(q) ≈ Lmult(q)*jacobian(ExponentialMap,ϕ)
@test DifferentialRotations.∇differential(q) ≈ Lmult(q)*jacobian(CayleyMap,ϕ)
@test DifferentialRotations.∇differential(q) ≈ Lmult(q)*jacobian(MRPMap,ϕ)


############################################################################################
#                          ROLL, PITCH, YAW EULER ANGLES
############################################################################################

e0 = @SVector [deg2rad(45), deg2rad(60), deg2rad(20)]
e1 = RPY(e0...)

R = rotmat(e1)
e1_ = DifferentialRotations.from_rotmat(rotmat(e1))
@test e1_.θ ≈ e1.θ
@test e1_.ψ ≈ e1.ψ
@test e1_.ϕ ≈ e1.ϕ

# @test rotmat(e1*e1) ≈ RotXYZ(e2*e2)

e1 = RPY(rand(3)...)
e2 = RPY(rand(3)...)
R = rotmat(e2*e1)

# Test inverses
e1,e2 = rand(RPY), rand(RPY)
e3 = e2*e1
@test e2\e3 ≈ e1
@test e3/e1 ≈ e2
@test r ≈ e3\(e2*e1*r)

@test ∇rotate(e1,r) isa SMatrix{3,3}
@test ∇composition1(e2,e1) isa SMatrix{3,3}
@test ∇composition2(e2,e1) isa SMatrix{3,3}


############################################################################################
#                              MODIFIED RODRIGUES PARAMETERS
############################################################################################
p1 = MRP(q1)
p2 = MRP(q2)
# @btime $p2*$p1
# @btime $p2*$r

# Test rotations and composition
@test p2*r ≈ q2*r
@test p2*p1*r ≈ q2*q1*r
@test rotmat(p2)*r ≈ p2*r
p3 = p2*p1
@test p2\p3 ≈ p1
@test p3/p1 ≈ p2

@test r ≈ p3\(p2*p1*r)

r = @SVector rand(3)
R = rotmat(q1)
@test R*r ≈ q1*r

p = MRP(q1)
R1 = rotmat(p)
@test R1*r ≈ R*r
@test p*r ≈ R*r

@test UnitQuaternion(p) ≈ q1


# Test composition jacobians
@test ForwardDiff.jacobian(x->SVector(p2*MRP(x)),SVector(p1)) ≈ ∇composition1(p2,p1)
@test ForwardDiff.jacobian(x->SVector(MRP(x)*p1),SVector(p2)) ≈ ∇composition2(p2,p1)
p0 = MRP(0,0,0)
@test DifferentialRotations.∇differential(p2) ≈ ∇composition1(p2,p0)

############################################################################################
#                              RODRIGUES PARAMETERS
############################################################################################

g1 = RodriguesParam(q1)
g2 = RodriguesParam(q2)
@test g2 ≈ RodriguesParam(-q2)  # test double-cover
@test q2 ≈ UnitQuaternion(g2)
@test g2 ≈ RodriguesParam(UnitQuaternion(g2))

# Test compostion and rotation
@test g1*r ≈ q1*r
@test g2*r ≈ q2*r
@test g2*g1*r ≈ q2*q1*r
@test rotmat(g2)*r ≈ g2*r

g3 = g2*g1
@test g2\g3 ≈ g1
@test g3/g1 ≈ g2
@test r ≈ g3\(g2*g1*r)
@test r ≈ (g3\g2*g1)*r

# Test Jacobians
@test ForwardDiff.jacobian(g->RodriguesParam(g)*r, SVector(g1)) ≈ ∇rotate(g1, r)

function compose(g2,g1)
    N = (g2+g1 + g2 × g1)
    D = 1/(1-g2'g1)
    return D*N
end
@test ForwardDiff.jacobian(g->compose(SVector(g2),g), SVector(g1)) ≈ ∇composition1(g2,g1)
@test ForwardDiff.jacobian(g->compose(g,SVector(g1)), SVector(g2)) ≈ ∇composition2(g2,g1)

g0 = RodriguesParam{Float64}(0,0,0)
@test ∇composition1(g2, g0) ≈ DifferentialRotations.∇differential(g2)


# Conversions
import DifferentialRotations: rotmat_to_quat
Random.seed!(1) # i = 3
q = rand(UnitQuaternion)
@test rotmat_to_quat(rotmat(q)) ≈ q

Random.seed!(2) # i = 4
q = rand(UnitQuaternion)
@test rotmat_to_quat(rotmat(q)) ≈ q

Random.seed!(3) # i = 2
q = rand(UnitQuaternion)
@test rotmat_to_quat(rotmat(q)) ≈ q

Random.seed!(5) # i = 1
q = rand(UnitQuaternion)
@test rotmat_to_quat(rotmat(q)) ≈ q

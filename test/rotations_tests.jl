import DifferentialRotations: jacobian, ∇rotate, ∇composition1, ∇composition2

q1 = rand(UnitQuaternion)
q2 = rand(UnitQuaternion)
r = @SVector rand(3)
ω = @SVector rand(3)


############################################################################################
#                                 QUATERNIONS
############################################################################################

# Test constructors
@test UnitQuaternion(@SVector [1,3,4,5.]) == UnitQuaternion(1.,3.,4.,5.)
@test UnitQuaternion(rand(UnitQuaternion)) isa
    UnitQuaternion{Float64,DifferentialRotations.DEFAULT_QMAP}

# Initializers
@test zero(UnitQuaternion) == UnitQuaternion(1.0, 0.0, 0.0, 0.0)
@test zero(UnitQuaternion{Float64,MRPMap}) isa UnitQuaternion{Float64,MRPMap}
@test zero(UnitQuaternion) == UnitQuaternion(I)
@test zero(rand(UnitQuaternion)) == UnitQuaternion(I)

# Test math
@test UnitQuaternion(I) isa UnitQuaternion{Float64,DifferentialRotations.DEFAULT_QMAP}
@test UnitQuaternion{Float64,MRPMap}(I) isa UnitQuaternion{Float64,MRPMap}
@test LinearAlgebra.norm2(q1) ≈ norm(q1)^2

ϕ = ExponentialMap(q1)
@test expm(ϕ*2) ≈ q1
q = UnitQuaternion(ϕ)
@test exp(q) ≈ q1

q = UnitQuaternion(@SVector [1,2,3,4.])
@test 2*q == UnitQuaternion(@SVector [2,4,6,8.])
@test q*2 == UnitQuaternion(@SVector [2,4,6,8.])

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
@test q3 ⊖ q2 ≈ CayleyMap(q1)
q3 = UnitQuaternion{IdentityMap}(q3)
@test q3 ⊖ q2 ≈ SVector(q3) - SVector(q2)

q = q1
rhat = UnitQuaternion(r)
@test q*r ≈ Vmat()*Lmult(q)*Rmult(q)'*Vmat()'r
@test q*r ≈ Vmat()*Lmult(q)*Rmult(q)'*Hmat(r)
@test q*r ≈ Vmat()*Lmult(q)*Lmult(rhat)*Tmat()*SVector(q)
@test q*r ≈ Vmat()*Rmult(q)'*Rmult(rhat)*SVector(q)
@test q*r ≈ Hmat()'Rmult(q)'*Rmult(rhat)*SVector(q)
@test Rmult(SVector(q)) == Rmult(q)
@test Lmult(SVector(q)) == Lmult(q)
@test Hmat(r) == SVector(UnitQuaternion(r))

@test kinematics(q1,ω) isa SVector{4}

@test ForwardDiff.jacobian(q->UnitQuaternion{VectorPart}(q)*r,SVector(q)) ≈ ∇rotate(q,r)
# @btime ForwardDiff.jacobian(q->UnitQuaternion(q)*$r,SVector($q))
# @btime ∇rotate($q,$r)

@test ForwardDiff.jacobian(q->SVector(q2*UnitQuaternion{VectorPart}(q)),SVector(q1)) ≈
    ∇composition1(q2,q1)
@test ForwardDiff.jacobian(q->SVector(UnitQuaternion{VectorPart}(q)*q1),SVector(q2)) ≈
    ∇composition2(q2,q1)

b = @SVector rand(4)
qval = SVector(q1)
ForwardDiff.jacobian(q->∇composition1(q2,UnitQuaternion(q))'b, @SVector [1,0,0,0.])
diffcomp =  ϕ->SVector(q2*CayleyMap(ϕ))
∇diffcomp(ϕ) = ForwardDiff.jacobian(diffcomp, ϕ)
@test ∇diffcomp(@SVector zeros(3)) ≈ DifferentialRotations.∇differential(q2)
@test ForwardDiff.jacobian(ϕ->∇diffcomp(ϕ)'b, @SVector zeros(3)) ≈
    DifferentialRotations.∇²differential(q2, b)

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

# Test constructors
e = RPY(0,0,pi/2)
@test e isa RPY{Float64}
@test RPY(0,0,0) isa RPY{Int}
@test RPY(e) isa RPY{Float64}
@test RPY(Float32, 0,0,0) isa RPY{Float32}

# Test initializers
@test rand(RPY) isa RPY{Float64}
@test rand(RPY{Float32}) isa RPY{Float32}
@test zero(RPY{Float32}) == RPY(0,0,0)
@test zero(RPY{Float32}) isa RPY{Float32}
@test zero(RPY) isa RPY{Float64}

e0 = @SVector [deg2rad(45), deg2rad(60), deg2rad(20)]
e1 = RPY(e0...)
@test roll(e1) == deg2rad(45)
@test pitch(e1) == deg2rad(60)
@test yaw(e1) == deg2rad(20)

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

# Test kinematics
@test kinematics(e1, ω) isa SVector{3}


############################################################################################
#                              MODIFIED RODRIGUES PARAMETERS
############################################################################################

# Test constructors
@test MRP(Float32,1,1,1) isa MRP{Float32}
@test MRP{Float32}(@SVector rand(3)) isa MRP{Float32}
p = rand(MRP{Float64})
@test p isa MRP{Float64}
@test MRP{Float32}(p) isa MRP{Float32}

# Test initializers
@test zero(MRP) == MRP(0.0, 0.0, 0.0)
@test zero(MRP{Float32}) == MRP(0f0, 0f0, 0f0)

# Test math
@test norm(p) == norm(SVector(p))

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

b = @SVector rand(3)
pval = SVector(p1)
@test ForwardDiff.jacobian(p->∇composition1(p2,MRP(p))'b, pval) ≈
    DifferentialRotations.∇²composition1(p2,p1,b)
@test DifferentialRotations.∇²differential(p2, b) ≈
    DifferentialRotations.∇²composition1(p2, p0, b)

@test ForwardDiff.jacobian(p->SVector(p2\MRP(p)), pval) ≈ DifferentialRotations.∇err(p2, p1)
@test ForwardDiff.jacobian(p->DifferentialRotations.∇err(p2, MRP(p))'b, pval) ≈
    DifferentialRotations.∇²err(p2, p1, b)

# Test rotation jacobian
pval = SVector(p1)
@test ForwardDiff.jacobian(x->SVector(MRP(x)*r), pval) ≈ ∇rotate(p1,r)

# Test kinematics
@test kinematics(p1, ω) isa SVector{3}

############################################################################################
#                              RODRIGUES PARAMETERS
############################################################################################

# Test constructors
g = rand(RodriguesParam)
@test RodriguesParam(Float32,1,1,1) isa RodriguesParam{Float32}
@test RodriguesParam{Float32}(g) isa RodriguesParam{Float64}  # QUESTION: is this desired?

# Test initializers
@test zero(RodriguesParam) == RodriguesParam(0.,0.,0.)

g1 = RodriguesParam(q1)
g2 = RodriguesParam(q2)
@test g2 ≈ RodriguesParam(-q2)  # test double-cover
@test (q2 ≈ UnitQuaternion(g2)) || (q2 ≈ -UnitQuaternion(g2))
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

gval = SVector(g1)
@test ForwardDiff.jacobian(g->∇composition1(g2,RodriguesParam(g))'b, gval) ≈
    DifferentialRotations.∇²composition1(g2,g1,b)
@test DifferentialRotations.∇²differential(g2, b) ≈
    DifferentialRotations.∇²composition1(g2, g0, b)

# Test kinematics
@test kinematics(g1, ω) isa SVector{3}

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

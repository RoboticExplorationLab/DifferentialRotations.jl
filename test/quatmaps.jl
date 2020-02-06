import DifferentialRotations.jacobian
# Exponential
ϕ = @SVector rand(3)
@test ForwardDiff.jacobian(x->SVector(ExponentialMap(x)),ϕ) ≈ jacobian(ExponentialMap,ϕ)
@test map_type(ExponentialMap(ϕ)) == ExponentialMap

ϕ = 1e-6*@SVector rand(3)
@test ForwardDiff.jacobian(x->SVector(ExponentialMap(x)),ϕ) ≈ jacobian(ExponentialMap,ϕ)
@test map_type(ExponentialMap(ϕ)) == ExponentialMap

# Vector Part
v = 0.1*@SVector rand(3)
@test ForwardDiff.jacobian(x->SVector(VectorPart(x)),v) ≈
    jacobian(VectorPart, v)
@test map_type(VectorPart(v)) == VectorPart

# Gibbs Vectors
g = @SVector rand(3)
@test ForwardDiff.jacobian(x->SVector(CayleyMap(x)),g) ≈ jacobian(CayleyMap, g)
@test map_type(CayleyMap(g)) == CayleyMap

# MRPs
p = SVector(rand(MRP))
@test ForwardDiff.jacobian(x->SVector(MRPMap(x)),p) ≈
    jacobian(MRPMap, p)
@test map_type(MRPMap(p)) == MRPMap


μ0 = 1/DifferentialRotations.scaling(VectorPart)
jac_eye = [@SMatrix zeros(1,3); μ0*Diagonal(@SVector ones(3))];
@test jacobian(ExponentialMap, p*1e-10) ≈ jac_eye
@test jacobian(MRPMap, p*1e-10) ≈ jac_eye
@test jacobian(CayleyMap, p*1e-10) ≈ jac_eye
@test jacobian(VectorPart, p*1e-10) ≈ jac_eye


############################################################################################
#                                 INVERSE RETRACTION MAPS
############################################################################################

# Exponential Map
Random.seed!(1);
q = rand(UnitQuaternion)
q = UnitQuaternion{ExponentialMap}(q)
qval = SVector(q)
@test ExponentialMap(q) == DifferentialRotations.scaling(ExponentialMap)*logm(q)
@test ExponentialMap(ExponentialMap(q)) ≈ q
@test ExponentialMap(ExponentialMap(ϕ)) ≈ ϕ

function invmap(q)
    μ = DifferentialRotations.scaling(ExponentialMap)
    v = @SVector [q[2], q[3], q[4]]
    s = q[1]
    θ = norm(v)
    M = μ*2atan(θ, s)/θ
    return M*v
end
@test invmap(qval) ≈ DifferentialRotations.scaling(ExponentialMap)*logm(q)

qI = VectorPart(v*1e-5)
@test ForwardDiff.jacobian(invmap, qval) ≈ jacobian(ExponentialMap, q)
@test ForwardDiff.jacobian(invmap, SVector(qI)) ≈ jacobian(ExponentialMap, qI)

b = @SVector rand(3)
@test ForwardDiff.jacobian(q->jacobian(ExponentialMap,
    UnitQuaternion{ExponentialMap}(q))'b, qval) ≈
    DifferentialRotations.∇jacobian(ExponentialMap, q, b)

# Vector Part
invmap(q) = @SVector [q[2], q[3], q[4]]
@test VectorPart(q) ≈ DifferentialRotations.scaling(VectorPart)*invmap(qval)
@test ForwardDiff.jacobian(invmap, qval) ≈ jacobian(VectorPart, q)
@test VectorPart(VectorPart(q)) ≈ q
@test VectorPart(VectorPart(v)) ≈ v

@test ForwardDiff.jacobian(q->jacobian(VectorPart,
    UnitQuaternion{VectorPart}(q))'b, qval) ≈
    DifferentialRotations.∇jacobian(VectorPart, q, b)

# Cayley
invmap(q) = 1/q[1] * @SVector [q[2], q[3], q[4]]
@test CayleyMap(q) ≈ DifferentialRotations.scaling(CayleyMap)*invmap(qval)
@test ForwardDiff.jacobian(invmap, qval) ≈ jacobian(CayleyMap, q)
@test CayleyMap(CayleyMap(q)) ≈ q
@test CayleyMap(CayleyMap(g)) ≈ g

@test ForwardDiff.jacobian(q->jacobian(CayleyMap,
    UnitQuaternion{CayleyMap}(q))'b, qval) ≈
    DifferentialRotations.∇jacobian(CayleyMap, q, b)

# MRP
invmap(q) = DifferentialRotations.scaling(MRPMap)/(1+q[1]) * @SVector [q[2], q[3], q[4]]
@test MRPMap(q) ≈ invmap(qval)
@test ForwardDiff.jacobian(invmap, qval) ≈ jacobian(MRPMap, q)
@test MRPMap(MRPMap(q)) ≈ q
@test MRPMap(MRPMap(p)) ≈ p

@test ForwardDiff.jacobian(q->jacobian(MRPMap,
    UnitQuaternion{MRPMap}(q))'b, qval) ≈
    DifferentialRotations.∇jacobian(MRPMap, q, b)

# Test near origin
μ0 = DifferentialRotations.scaling(VectorPart)
jacT_eye = [@SMatrix zeros(1,3); μ0*Diagonal(@SVector ones(3))]';
@test isapprox(jacobian(ExponentialMap,qI), jacT_eye, atol=1e-5)
@test isapprox(jacobian(VectorPart,qI), jacT_eye, atol=1e-5)
@test isapprox(jacobian(CayleyMap,qI), jacT_eye, atol=1e-5)
@test isapprox(jacobian(MRPMap,qI), jacT_eye, atol=1e-5)


# Test synonyms
@test DifferentialRotations.inverse_map_jacobian(q) == jacobian(map_type(q), q)
q = UnitQuaternion{CayleyMap}(q)
@test DifferentialRotations.inverse_map_jacobian(q) == jacobian(map_type(q), q)
@test DifferentialRotations.inverse_map_∇jacobian(q, b) ==
    DifferentialRotations.∇jacobian(map_type(q), q, b)
@test DifferentialRotations.inverse_map_jacobian(rand(MRP)) == I
@test DifferentialRotations.inverse_map_∇jacobian(rand(MRP), b) == I*0

# IdentityMap
q = rand(UnitQuaternion{Float64,IdentityMap})
@test IdentityMap(q) == SVector(q)
@test jacobian(IdentityMap,q) == I
@test IdentityMap(SVector(q)) == q
@test map_type(IdentityMap(SVector(q))) == IdentityMap

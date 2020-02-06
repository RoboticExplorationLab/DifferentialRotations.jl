rots = [UnitQuaternion{Float64,CayleyMap}, RodriguesParam{Float64},
        MRP{Float64}, RPY{Float64}]

for (r1,r2) in permutations(rots,2)
    r = rand(r1)
    @test r2(r) isa r2
end

@test UnitQuaternion{Float64,CayleyMap}(rand(UnitQuaternion)) isa rots[1]
@test map_type(rots[1]) == CayleyMap
@test map_type(typeof(rand(UnitQuaternion))) == DifferentialRotations.DEFAULT_QMAP

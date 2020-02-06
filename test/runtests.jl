using DifferentialRotations
using Test
using StaticArrays
using ForwardDiff
using LinearAlgebra
using Random
using Combinatorics

@testset "Quaternions" begin
    include("quatmaps.jl")
end

@testset "All Rotations" begin
    include("rotations_tests.jl")
    include("other_tests.jl")
end

@testset "Conversions" begin
    include("conversions_test.jl")
end

using DifferentialRotations
using Test
using StaticArrays
using ForwardDiff
using LinearAlgebra
using Random

@testset "Quaternions" begin
    include("quatmaps.jl")
end

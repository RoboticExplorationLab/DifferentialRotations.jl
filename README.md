# DifferentialRotations.jl
This package provides several useful types for representing 3D rotations,
as well as many derivatives with respect to those rotations.

In particular, DifferentialRotations includes the `UnitQuaternion` type
that allows for several different "quaternion maps" that map the 4D
unit quaternion onto the 3D tangent plane, necessary for correctly taking
derivatives with respect to quaternion-valued functions.

We treat all rotations as active rotations that rotate from a local frame to a
global frame.

## Implemented Types
The following types for parameterizing 3D rotations are currently implemented
* `UnitQuaternion{T,D}`: described in detail below
* `RodriguesPara{T}`: a three-parameter representation that is singular at
180° but has no sign ambiguities
* `MRP{T}`: Modified Rodrigues Parameter, is a stereographic projection
of the quaternion. Goes singular at 360°.
* `RPY{T}`: Roll-pitch-yaw Euler angles.

## Interface
The following operations are supported by all of the types listed in the
previous section.

### Composition
Any two rotations of the same type can be composed with simple multiplication:
```julia
q3 = q2*q1
```
Rotations can be composed with the opposite (or inverse) rotation with the appropriate
division operation
```julia
q1 = q2\q3
q2 = q3/q1
```

### Rotation
Any rotation can operate on a 3D vector (represented as a `SVector{3}`), again through
simple multiplication:
```julia
r2 = q*r
```
which also supports multiplication by the opposite rotation
```julia
r = q\r2
```

### Kinematics
The kinematics, or time derivative as a function of the angular velocity, can be evaluated
using
`kinematics(q,ω)`
where ω is the 3D angular velocity

## Jacobians
The following Jacobians are also implemented for all types
* `∇rotate(q,r)`: Jacobian of the `q*r` with respect to the rotation
* `∇composition1(q2,q1)`: Jacobian of `q2*q1` with respect to q1
* `∇composition2(q2,q1,b)`: Jacobian of `q2*q1` with respect to q2
* `∇²composition1(q2,q1)`: Jacobian of `∇composition1(q2,q2)'b` where b is an arbitrary vector
* `∇differential(q)`: Jacobian of composing the rotation with an infinitesimal rotation, with
respect to the infinitesimal rotation. For unit quaternions, this is a 4x3 matrix.
* `∇²differential(q,b)`: Jacobian of `∇differential(q)'b` for some vector b.


## Unit Quaternion
The `UnitQuaternion{T,D}` type is parameterized by its internal data type
`T` (typically `Float64`) and its quaternion map `D <: QuatMap`.
The quaternion map is responsible for converting to and from the space of
differential unit quaternions, which necessarily live in the 3D plane tangent
to the quaternion. There are several different quaternion maps currently
implemented:
* `ExponentialMap`: A very common mapping that uses the quaternion
exponential and the quaternion logarithm. The quaternion logarithm
converts a 3D rotation vector (i.e. axis-angle vector) to a unit quaternion.
It tends to be the most computationally expensive mapping.

* `CayleyMap`: Represents the differential quaternion using Rodrigues
parameters. This parameterization goes singular at 180° but does not
inherit the sign ambiguities of the unit quaternion. It offers an
excellent combination of cheap computation and good behavior.

* `MRPMap`: Uses Modified Rodrigues Parameters (MRPs) to represent the
differential unit quaternion. This mapping goes singular at 360°.

* `VectorPart`: Uses the vector part of the unit quaternion as the
differential unit quaternion. This mapping also goes singular at 180° but is
the computationally cheapest map and often performs well.

The following quaternion-specific functions are currently implemented
* `scalar`: return the scalar (or real) part of the quaternion
* `vector`: return the vector (or imaginary) part of the quaternion
* `conj`
* `inv`
* `vecnorm`: returns the norm of the vector part of the unit quaternion
* `normalize`: re-normalizes the unit quaternion
* `exp`: quaternion exponential
* `log`: quaternion logarithm
* `expm`: exponential map
* `logm`: logarithmic map

Additionally, we provide the following functions that are useful for
doing linear-algebraic operations with quaternions
* `Lmult`: `q2*q1` is the same as `Lmult(q2)*SVector(q1)`
* `Rmult`: `q2*q1` is the same as `Rmult(q1)*SVector(q2)`
* `Tmat`: `conj(q)` is the same as `Tmat()*SVector(q)`
* `Hmat`: `vector(q)` is the same as `Hmat()'SVector(q)`



## Conversions
The following conversions are implemented. An "x" indicates a direct conversion,
a "q" represents a conversion by converting first to a quaternion,
an "R" represents a conversion by converting first to a rotation matrix,
and "R->q" represents a conversion by converting to a rotation matrix, then unit quaternion

| Row -> Col | Quat | RP | MRP | RPY | RotMat |
|------------|------|----|-----|-----|--------|
| Quat       |  x   | x  |  x  |  R  |   x    |
| RP         |  x   | x  |  q  |  R  |   x    |
| MRP        |  x   | q  |  x  |  R  |   x    |
| RPY        |  R   |R->q|R->q |  x  |   x    |
| RotMat     |  x   | q  |  q  |  x  |   x    |


## Releated Packages
This package is similar to `Rotations.jl` but more focused on implementing the methods
needed to take derivatives with respect to rotations, and provides functionality to the
quaternion type that `Rotations.jl` and `Quaternions.jl` do not, namely the parameterization
on the quaternion mapping to differential quaternions.

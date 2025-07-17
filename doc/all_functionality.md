# Manif Library Functionality Reference

This document provides a comprehensive list of all functions and operators with their input and output arguments that need to be implemented across all manifolds in the manif library.

## Core LieGroup Base Operations

### Construction and Assignment
- `setIdentity()` → `_Derived&`
- `setRandom()` → `_Derived&`
- `Identity()` (static) → `LieGroup`
- `Random()` (static) → `LieGroup`

### Data Access
- `coeffs()` → `DataType&` / `const DataType&`
- `data()` → `Scalar*` / `const Scalar*`
- `cast<NewScalar>()` → `LieGroupTemplate<NewScalar>`
- `size()` → `unsigned int`
- `operator[](int index)` → `auto&` / `const auto&`

### Core Lie Group Operations
- `inverse(OptJacobianRef J_minv_m = {})` → `LieGroup`
- `log(OptJacobianRef J_t_m = {})` → `Tangent`
- `lift(OptJacobianRef J_t_m = {})` (deprecated) → `Tangent`
- `compose(const LieGroupBase& m, OptJacobianRef J_mc_ma = {}, OptJacobianRef J_mc_mb = {})` → `LieGroup`
- `act(const Vector& v, OptJacobianRef J_vout_m = {}, OptJacobianRef J_vout_v = {})` → `Vector`
- `adj()` → `Jacobian`
- `between(const LieGroupBase& m, OptJacobianRef J_mc_ma = {}, OptJacobianRef J_mc_mb = {})` → `LieGroup`

### Manifold Operations (Plus/Minus)
- `rplus(const TangentBase& tangent, OptJacobianRef J_mout_m = {}, OptJacobianRef J_mout_t = {})` → `LieGroup`
- `lplus(const TangentBase& tangent, OptJacobianRef J_mout_m = {}, OptJacobianRef J_mout_t = {})` → `LieGroup`
- `plus(const TangentBase& tangent, OptJacobianRef J_mout_m = {}, OptJacobianRef J_mout_t = {})` → `LieGroup`
- `rminus(const LieGroupBase& m, OptJacobianRef J_t_ma = {}, OptJacobianRef J_t_mb = {})` → `Tangent`
- `lminus(const LieGroupBase& m, OptJacobianRef J_t_ma = {}, OptJacobianRef J_t_mb = {})` → `Tangent`
- `minus(const LieGroupBase& m, OptJacobianRef J_t_ma = {}, OptJacobianRef J_t_mb = {})` → `Tangent`

### Comparison Operations
- `isApprox(const LieGroupBase& m, const Scalar& prec = Constants<Scalar>::eps)` → `bool`
- `operator==(const LieGroupBase& m)` → `bool`
- `operator!=(const LieGroupBase& m)` → `bool`

### Arithmetic Operators
- `operator+(const TangentBase& tangent)` → `LieGroup`
- `operator+=(const TangentBase& tangent)` → `_Derived&`
- `operator-(const LieGroupBase& m)` → `Tangent`
- `operator*(const LieGroupBase& m)` → `LieGroup`
- `operator*=(const LieGroupBase& m)` → `_Derived&`

## Core Tangent Base Operations

### Construction and Assignment
- `setZero()` → `_Derived&`
- `setRandom()` → `_Derived&`
- `setVee(const LieAlg& v)` → `_Derived&`
- `Zero()` (static) → `Tangent`
- `Random()` (static) → `Tangent`

### Data Access
- `coeffs()` → `DataType&` / `const DataType&`
- `data()` → `Scalar*` / `const Scalar*`
- `cast<NewScalar>()` → `TangentTemplate<NewScalar>`
- `size()` → `unsigned int`
- `operator[](int index)` → `auto&` / `const auto&`

### Core Tangent Operations
- `exp(OptJacobianRef J_e_t = {})` → `LieGroup`
- `retract(OptJacobianRef J_r_t = {})` (deprecated) → `LieGroup`
- `hat()` → `LieAlg`
- `generator(int i)` → `LieAlg`
- `innerWeights()` → `InnerWeightsMatrix`
- `inner(const TangentBase& other)` → `Scalar`
- `weightedNorm()` → `Scalar`
- `squaredWeightedNorm()` → `Scalar`

### Manifold Operations (Plus/Minus)
- `rplus(const LieGroupBase& m, OptJacobianRef J_mout_m = {}, OptJacobianRef J_mout_t = {})` → `LieGroup`
- `lplus(const LieGroupBase& m, OptJacobianRef J_mout_m = {}, OptJacobianRef J_mout_t = {})` → `LieGroup`
- `plus(const LieGroupBase& m, OptJacobianRef J_mout_m = {}, OptJacobianRef J_mout_t = {})` → `LieGroup`
- `plus(const TangentBase& tangent)` → `Tangent`
- `minus(const TangentBase& tangent)` → `Tangent`

### Jacobian Operations
- `rjac()` → `Jacobian`
- `ljac()` → `Jacobian`
- `rjacinv()` → `Jacobian`
- `ljacinv()` → `Jacobian`
- `smallAdj()` → `Jacobian`

### Tangent-Specific Operations
- `bracket(const TangentBase& other)` → `Tangent`
- `Bracket(const TangentBase& a, const TangentBase& b)` (static) → `Tangent`
- `Vee(const LieAlg& v)` (static) → `Tangent`
- `Generator(int i)` (static) → `LieAlg`
- `InnerWeights()` (static) → `InnerWeightsMatrix`

### Comparison Operations
- `isApprox(const TangentBase& other, const Scalar& prec = Constants<Scalar>::eps)` → `bool`
- `isApprox(const LieAlg& other, const Scalar& prec = Constants<Scalar>::eps)` → `bool`

### Arithmetic Operators
- `operator-()` (unary) → `Tangent`
- `operator+(const LieGroupBase& m)` → `LieGroup`
- `operator+(const TangentBase& tangent)` → `Tangent`
- `operator-(const TangentBase& tangent)` → `Tangent`
- `operator+=(const TangentBase& tangent)` → `_Derived&`
- `operator-=(const TangentBase& tangent)` → `_Derived&`
- `operator*(const Scalar& s)` → `Tangent`
- `operator/(const Scalar& s)` → `Tangent`
- `operator*=(const Scalar& s)` → `Tangent`
- `operator/=(const Scalar& s)` → `Tangent`
- `operator==(const TangentBase& other)` → `bool`
- `operator!=(const TangentBase& other)` → `bool`

## Global Utility Functions

### Data Access Functions
- `coeffs(const LieGroupBase& lie_group)` → `const DataType&`
- `coeffs(const TangentBase& tangent)` → `const DataType&`
- `data(const LieGroupBase& lie_group)` → `const Scalar*` / `Scalar*`
- `data(const TangentBase& tangent)` → `const Scalar*` / `Scalar*`

### Construction Functions
- `identity(LieGroupBase& lie_group)` → `void`
- `Identity<LieGroup>()` → `LieGroup`
- `zero(TangentBase& tangent)` → `void`
- `Zero<Tangent>()` → `Tangent`
- `random(LieGroupBase& lie_group)` → `void`
- `random(TangentBase& tangent)` → `void`
- `Random<Type>()` → `Type`

### Core Operations
- `inverse(const LieGroupBase& lie_group, OptJacobianRef J_minv_m = {})` → `LieGroup`
- `log(const LieGroupBase& lie_group, OptJacobianRef J_l_m = {})` → `Tangent`
- `lift(const LieGroupBase& lie_group, OptJacobianRef J_l_m = {})` (deprecated) → `Tangent`
- `exp(const TangentBase& tangent, OptJacobianRef J_e_t = {})` → `LieGroup`
- `retract(const TangentBase& tangent, OptJacobianRef J_r_t = {})` (deprecated) → `LieGroup`
- `compose(const LieGroupBase& lhs, const LieGroupBase& rhs, OptJacobianRef J_mc_ma = {}, OptJacobianRef J_mc_mb = {})` → `LieGroup`
- `between(const LieGroupBase& lhs, const LieGroupBase& rhs, OptJacobianRef J_mc_ma = {}, OptJacobianRef J_mc_mb = {})` → `LieGroup`
- `act(const LieGroupBase& lie_group, const Vector& v, OptJacobianRef J_vout_m = {}, OptJacobianRef J_vout_v = {})` → `Vector`

### Manifold Operations
- `rplus(const LieGroupBase& lie_group, const TangentBase& tangent, OptJacobianRef J_mout_m = {}, OptJacobianRef J_mout_t = {})` → `LieGroup`
- `lplus(const LieGroupBase& lie_group, const TangentBase& tangent, OptJacobianRef J_mout_m = {}, OptJacobianRef J_mout_t = {})` → `LieGroup`
- `plus(const LieGroupBase& lie_group, const TangentBase& tangent, OptJacobianRef J_mout_m = {}, OptJacobianRef J_mout_t = {})` → `LieGroup`
- `rminus(const LieGroupBase& lhs, const LieGroupBase& rhs, OptJacobianRef J_t_ma = {}, OptJacobianRef J_t_mb = {})` → `Tangent`
- `lminus(const LieGroupBase& lhs, const LieGroupBase& rhs, OptJacobianRef J_t_ma = {}, OptJacobianRef J_t_mb = {})` → `Tangent`
- `minus(const LieGroupBase& lhs, const LieGroupBase& rhs, OptJacobianRef J_t_ma = {}, OptJacobianRef J_t_mb = {})` → `Tangent`

## Algorithm Functions

### Averaging Algorithms
- `average_biinvariant(const Container<LieGroup>& points, Scalar eps = Constants<Scalar>::eps, int max_iterations = 20)` → `LieGroup`
- `average(const Container<LieGroup>& points, Scalar eps = Constants<Scalar>::eps, int max_iterations = 20)` → `LieGroup`
- `average_frechet_left(const Container<LieGroup>& points, Scalar eps = Constants<Scalar>::eps, int max_iterations = 20)` → `LieGroup`
- `average_frechet_right(const Container<LieGroup>& points, Scalar eps = Constants<Scalar>::eps, int max_iterations = 20)` → `LieGroup`

### Interpolation Functions
- `interpolate_slerp(const LieGroupBase& ma, const LieGroupBase& mb, const Scalar& t)` → `LieGroup`
- `interpolate_cubic(const LieGroupBase& ma, const LieGroupBase& mb, const Scalar& t, const Tangent& ta = Tangent::Zero(), const Tangent& tb = Tangent::Zero())` → `LieGroup`
- `interpolate_smooth(const LieGroupBase& ma, const LieGroupBase& mb, const Scalar& t, const unsigned int m, const Tangent& ta = Tangent::Zero(), const Tangent& tb = Tangent::Zero())` → `LieGroup`
- `interpolate(const LieGroupBase& ma, const LieGroupBase& mb, const Scalar& t, const INTERP_METHOD method = INTERP_METHOD::SLERP, const Tangent& ta = Tangent::Zero(), const Tangent& tb = Tangent::Zero())` → `LieGroup`

### Curve Fitting
- `decasteljau(const std::vector<LieGroup>& trajectory, const unsigned int degree, const unsigned int k_interp, const bool closed_curve = false)` → `std::vector<LieGroup>`
- `computeBezierCurve(const std::vector<LieGroup>& control_points, const unsigned int degree, const unsigned int k_interp)` → `std::vector<LieGroup>`

### Utility Functions
- `binomial_coefficient(const T n, const T k)` → `T`
- `ipow(const T base, const int exp, T carry = 1)` → `T`
- `polynomialBernstein(const T n, const T i, const T t)` → `T`
- `smoothing_phi(const T t, const std::size_t degree)` → `T`

## Manifold-Specific Functions

### SO3 Specific
- `transform()` → `Transformation`
- `rotation()` → `Rotation`
- `x()` → `Scalar`
- `y()` → `Scalar`
- `z()` → `Scalar`
- `w()` → `Scalar`
- `quat()` → `QuaternionDataType`
- `quat(const QuaternionDataType& quaternion)` → `void`
- `quat(const Eigen::MatrixBase& quaternion)` → `void`
- `normalize()` → `void`

### SE3 Specific
- `transform()` → `Transformation`
- `rotation()` → `Rotation`
- `translation()` → `Translation`
- `x()` → `Scalar`
- `y()` → `Scalar`
- `z()` → `Scalar`
- `quat()` → `QuaternionDataType`
- `quat(const QuaternionDataType& quaternion)` → `void`
- `normalize()` → `void`

### SO2 Specific
- `transform()` → `Transformation`
- `rotation()` → `Rotation`
- `angle()` → `Scalar`
- `real()` → `Scalar`
- `imag()` → `Scalar`

### SE2 Specific
- `transform()` → `Transformation`
- `rotation()` → `Rotation`
- `translation()` → `Translation`
- `x()` → `Scalar`
- `y()` → `Scalar`
- `angle()` → `Scalar`

### Rn Specific
- `transform()` → `Transformation`

## Common Operators Across All Manifolds

### Assignment Operators
- `operator=(const _Derived& other)` → `_Derived&`
- `operator=(const LieGroupBase& other)` → `_Derived&`
- `operator=(const DataType& coeffs)` → `_Derived&`

### Arithmetic Operators
- `operator+(const TangentBase& tangent)` → `LieGroup`
- `operator+(const LieGroupBase& other)` → `LieGroup`
- `operator-(const LieGroupBase& other)` → `Tangent`
- `operator-(const TangentBase& tangent)` → `Tangent`
- `operator-()` (unary) → `Tangent`
- `operator*(const LieGroupBase& other)` → `LieGroup`
- `operator*(const Scalar& s)` → `Tangent`
- `operator/(const Scalar& s)` → `Tangent`
- `operator+=(const TangentBase& tangent)` → `_Derived&`
- `operator-=(const TangentBase& tangent)` → `_Derived&`
- `operator*=(const LieGroupBase& other)` → `_Derived&`
- `operator*=(const Scalar& s)` → `Tangent`
- `operator/=(const Scalar& s)` → `Tangent`

### Comparison Operators
- `operator==(const LieGroupBase& other)` → `bool`
- `operator==(const TangentBase& other)` → `bool`
- `operator!=(const LieGroupBase& other)` → `bool`
- `operator!=(const TangentBase& other)` → `bool`

### Stream Operators
- `operator<<(std::ostream& os, const LieGroupBase& m)` → `std::ostream&`
- `operator<<(std::ostream& os, const TangentBase& t)` → `std::ostream&`

### Access Operators
- `operator[](int index)` → `auto&` / `const auto&`

## Bundle Operations

### Bundle-Specific Functions
- `element<Index>()` → `Element&` / `const Element&`
- `cast<NewScalar>()` → `BundleTemplate<NewScalar>`

## Constants and Enums

### Interpolation Methods
- `INTERP_METHOD::SLERP`
- `INTERP_METHOD::CUBIC`
- `INTERP_METHOD::CNSMOOTH`

### Mathematical Constants
- `Constants<Scalar>::eps` → `Scalar`
- `Constants<Scalar>::pi` → `Scalar`

## Implementation Notes

1. **Optional Jacobians**: Most functions accept optional Jacobian references (`OptJacobianRef`) for computing derivatives
2. **Scalar Template**: Functions work with different scalar types (float, double, etc.)
3. **Container Templates**: Algorithm functions work with various container types (std::vector, etc.)
4. **Error Handling**: Functions may throw exceptions or use assertions for validation
5. **Deprecation**: Some functions (`lift()`, `retract()`) are deprecated in favor of `log()` and `exp()`
6. **Static Functions**: Many functions have both member and static versions
7. **Const Correctness**: Functions properly handle const and non-const versions
8. **Reference Types**: Functions return references where appropriate to avoid unnecessary copying

This reference covers all the functionality with input/output arguments that need to be implemented when creating a new manifold or when working with existing manifolds in the manif library.

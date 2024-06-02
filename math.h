//=============================== nostalgia.solutions ====================================//
//
// Purpose: to make math fun and easy to understand... simplify our load on CPU
// for the compromise of tiny accuracy loss. This can be useful for many things that doesn't
// require 100% precision. We can select from different algorithm strengths for our needs.
//
// This library has helpful functions for triggernometry and 3D math. Requires GLM.
// 
//=====================================================================================//

#pragma once
#include <xmmintrin.h> // Header for SSE intrinsics
#include <math.h>

#include "Math.h"

#include <glm/gtx/compatibility.hpp> // lerp etc
#include <glm/gtx/rotate_vector.hpp>
//#include "glm/gtx/euler_angles.hpp"

// GLM math
#define GLM_FORCE_PRECISION_LOWP_INT
#define GLM_FORCE_PRECISION_LOWP_FLOAT
#include "glm/ext.hpp"
#include "glm/gtx/norm.hpp" // glm::length2 (length sqr)
#include "glm/gtx/easing.hpp" // easing functions for smooth stuff

// Define one of these normalize methods
// #define NORMALIZE_REMAINDER
// #define NORMALIZE_ANGLE_SOURCE
// #define NORMALIZE_ANGLE_SOURCE_2
// #define NORMALIZE_ANGLE_SIMPLE_SOURCE_2
// #define NORMALIZE_ANGLE_GLM
#define NORMALIZE_ANGLE_BASIC

// Use ASM sqrt algorithm or Quake (Default SSE)
// #define ASM_SQRT
// #define QUAKE_SQRT

#if defined (QUAKE_SQRT)
	#include <bit>
	#include <limits>
#endif

// SQRT_APPROXIMATION: Enables an approximate algorithm for square root calculations.
// This approximation improves performance at the cost of precision. Useful for applications where exact accuracy is less critical.
#define SQRT_APPROXIMATION

// TRIG_APPROXIMATION: Enables approximate calculations for trigonometric functions (sin, cos, etc.).
// These approximations are faster but less precise, suitable for scenarios where speed is prioritized over exact accuracy.
#define TRIG_APPROXIMATION

// Use more simple formula for each trig function
#ifdef TRIG_APPROXIMATION
	 // If faster approxmiations are needed
	 // Disable this to use a medium strength approximation
	 #define TRIG_APPROXIMATION_STRONG
#endif

// If you want to rotate a 3D Vector without glm::rotate
// #define ALTERNATIVE_ROTATIONS

// If we want GetAngle normalized to 0-360 instead of -180 180
// #define GET_ANGLE_NORMALIZE_360

// Calculate Roll function(s) if needed
// #define GET_ROLL_FUNCTIONS

namespace math {
	// Some helpful constants
	const glm::vec3 ZERO_VEC(0.0f, 0.0f, 0.0f);
	const glm::vec3 NAN_VEC(NAN, NAN, NAN);

#if defined(NORMALIZE_ANGLE_SOURCE_2_SIMPLE) || defined(NORMALIZE_ANGLE_SOURCE_2)
	constexpr float INV_360 = 1.0f / 360.0f;
#endif

#ifdef X86_ASM_SQRT
	// Define FASTSQRT for 32-bit (x86) build inline ASM edition
#if defined(_M_IX86) || defined(__i386__)
#ifdef SQRT_APPROXIMATION
	float inline __declspec (naked) __fastcall FASTSQRT(float n)
	{
		_asm {
			movss xmm0, dword ptr[esp + 4] // Load n into xmm0
			rsqrtss xmm0, xmm0             // xmm0 = 1/sqrt(n)
			rcpss xmm0, xmm0               // xmm0 = 1/xmm0 = sqrt(n)
			ret 4
		}
	}
#else
	float inline __declspec (naked) __fastcall FASTSQRT(float n)
	{
		_asm fld dword ptr[esp + 4]
			_asm fsqrt
		_asm ret 4
	}
#endif
#define FastSqrt FASTSQRT
#else
	// Use std::sqrt for other architectures (like x64)
#define FastSqrt std::sqrt
#endif
#else
	// Only use these implementations if X86_ASM_SQRT is not defined
#ifdef QUAKE_SQRT
	inline constexpr float FastSqrt(float number) noexcept
	{
		static_assert(std::numeric_limits<float>::is_iec559); // (enable only on IEEE 754)

		float y = std::bit_cast<float>(
			0x5F200000 - (std::bit_cast<std::uint32_t>(number) >> 1));

		// Refine the inverse square root approximation
		y *= 1.68191391f - 0.703952009f * number * y * y;

		// Additional refinement step if SQRT_APPROXIMATION is not defined
#ifndef SQRT_APPROXIMATION
		y *= 1.50000036f - 0.500000053f * number * y * y;
#endif

		return number * y;
	}
#else
#ifdef SQRT_APPROXIMATION
	inline float fast_sqrt(float x) {
		return x > 0.0f ? _mm_cvtss_f32(_mm_mul_ss(_mm_rsqrt_ss(_mm_set_ss(x)), _mm_set_ss(x))) : 0.0f;
	}
#else
	inline float fast_sqrt(float x) {
		return x > 0.0f ? _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x))) : 0.0f;
	}
#endif
#endif
#endif

// Fast trigonometry functions using polynomial approximations
#ifdef TRIG_APPROXIMATION
#ifndef TRIG_APPROXIMATION_STRONG
	inline float fast_acos(const float& x) {
		return (-0.69813170079773212f * x * x - 0.87266462599716477f) * x + glm::half_pi<float>();
	}

	inline float fast_asin(const float& x) {
		return glm::half_pi<float>() - ((-0.69813170079773212f * x * x - 0.87266462599716477f) * x + glm::half_pi<float>());
	}

	// If you need something more simplified... here is an example.
	// Feel free to uncomment this and comment the below function if needed.
	// Max Error: 0.00084323
	/*
	inline float fast_atan(const float& x) {
		constexpr float A = 0.0776509570923569f;
		constexpr float B = -0.287434475393028f;
		constexpr float C = glm::quarter_pi<float>() - A - B;

		// 3-degree polynomial approximation
		const float xx = x * x;
		return ((A * xx + B) * xx + C) * x;
	}
	*/

	// Max Error: 0.00002324
	inline float fast_atan(const float& x) {
		constexpr float a1 = 0.99988f;
		constexpr float a3 = -0.330534f;
		constexpr float a5 = 0.181156f;
		constexpr float a7 = -0.0866971f;
		constexpr float a9 = 0.0216165f;

		// 9-degree polynomial approximation
		const float y = x * x;
		return x * (a1 + y * (a3 + y * (a5 + y * (a7 + a9 * y))));
	}

	inline float fast_atan2(const float& y, const float& x) {
		if (x == 0.0f) {
			if (y >= 0.0f) return glm::half_pi<float>();
			if (y <= 0.0f) return -glm::half_pi<float>();
		}
		else if (x > 0.0f) {
			if (-x <= y && y <= x) return fast_atan(y / x);
			if (y > x) return glm::half_pi<float>() - fast_atan(x / y);
			// y < -x
			else return -glm::half_pi<float>() - fast_atan(x / y);
		}
		else { // x < 0.0f
			if (0.0f <= y && y <= -x) return glm::pi<float>() + fast_atan(y / x);
			if (-x < y) return glm::half_pi<float>() - fast_atan(x / y);
			if (x <= y && y <= 0.0f) return fast_atan(y / x) - glm::pi<float>();
			// y < x
			else return -glm::half_pi<float>() - fast_atan(x / y);
		}

		return 0.0f;
	}
#else // MORE ACCURATE APPROXIMATIONS
	// Absolute error <= 6.7e-5
	inline float fast_acos(float x) {
		const float negate = float(x < 0);
		x = glm::abs(x);
		float ret = -0.0187293f;
		ret = ret * x;
		ret = ret + 0.0742610f;
		ret = ret * x;
		ret = ret - 0.2121144f;
		ret = ret * x;
		ret = ret + 1.5707288f;
		ret = ret * fast_sqrt(1.0 - x);
		ret = ret - 2 * negate * ret;
		return negate * glm::pi<float>() + ret;
	}

	inline float fast_asin(float x) {
		const float negate = float(x < 0);
		x = glm::abs(x);
		float ret = -0.0187293f;
		ret *= x;
		ret += 0.0742610f;
		ret *= x;
		ret -= 0.2121144f;
		ret *= x;
		ret += 1.5707288f;
		ret = glm::half_pi<float>() - fast_sqrt(1.0 - x) * ret;
		return ret - 2 * negate * ret;
	}

	inline float fast_atan2(const float& y, const float& x) {
		float t3 = glm::abs(x);
		float t1 = glm::abs(y);
		float t0 = glm::max(t3, t1);
		t1 = glm::min(t3, t1);
		t3 = 1.0f / t0;
		t3 = t1 * t3;

		const float t4 = t3 * t3;
		t0 = -0.013480470f;
		t0 = t0 * t4 + 0.057477314f;
		t0 = t0 * t4 - 0.121239071f;
		t0 = t0 * t4 + 0.195635925f;
		t0 = t0 * t4 - 0.332994597f;
		t0 = t0 * t4 + 0.999995630f;
		t3 = t0 * t3;

		t3 = (glm::abs(y) > glm::abs(x)) ? glm::half_pi<float>() - t3 : t3;
		t3 = (x < 0) ? glm::pi<float>() - t3 : t3;
		t3 = (y < 0) ? -t3 : t3;

		return t3;
	}

	inline float fast_atan(const float& x) {
		return fast_atan2(x, 1.0f);
	}
#endif // GLOBAL APPROX FUNCTIONS
	inline float fast_cos(const float& x) {
		// Coefficients
		constexpr float c1 = 0.9999999999999999999999914771f;
		constexpr float c2 = -0.4999999999999999999991637437f;
		constexpr float c3 = 0.04166666666666666665319411988f;
		constexpr float c4 = -0.00138888888888888880310186415f;
		constexpr float c5 = 0.00002480158730158702330045157f;
		constexpr float c6 = -0.000000275573192239332256421489f;
		constexpr float c7 = 0.000000002087675698165412591559f;
		constexpr float c8 = -0.0000000000114707451267755432394f;
		constexpr float c9 = 0.0000000000000477945439406649917f;
		constexpr float c10 = -0.00000000000000015612263428827781f;
		constexpr float c11 = 0.00000000000000000039912654507924f;

		const float x2 = x * x;
		return c1 + x2 * (c2 + x2 * (c3 + x2 * (c4 + x2 * (c5 + x2 * (c6 + x2 * (c7 + x2 * (c8 + x2 * (c9 + x2 * (c10 + x2 * c11)))))))));
	}

	// Fast sine function using fast_cos
	inline float fast_sin(const float& x) {
		return fast_cos(glm::half_pi<float>() - x);
	}

	inline float fast_tan(float x) {
		// Coefficients for the numerator
		constexpr float num_coeff[] = { 135135.0f, -17325.0f, 378.0f, -1.0f };
		// Coefficients for the denominator
		constexpr float den_coeff[] = { 135135.0f, -62370.0f, 3150.0f, -28.0f };

		// Calculate x^2, x^4, and x^6
		const float x2 = x * x;
		const float x4 = x2 * x2;
		const float x6 = x4 * x2;

		// Numerator: x * (c0 + c1*x^2 + c2*x^4 + c3*x^6)
		const float numerator = x * (num_coeff[0] + num_coeff[1] * x2 + num_coeff[2] * x4 + num_coeff[3] * x6);

		// Denominator: d0 + d1*x^2 + d2*x^4 + d3*x^6
		const float denominator = den_coeff[0] + den_coeff[1] * x2 + den_coeff[2] * x4 + den_coeff[3] * x6;

		// Return the approximation
		return numerator / denominator;
	}
#endif

	// Verify the matrix contains sane values (not NaN, Infinite)
	inline bool IsMatrixValid(const glm::fmat4x4& matrix) {
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j)
				if (!glm::isfinite(matrix[i][j])) return false;
		}
		return true; // All values are finite numbers.
	}

	// Transforms a 3D vector by a 3x4 matrix.
	inline void VectorTransform(glm::vec3& in_out, const glm::mat3x4& matrix) {
		in_out = {
			in_out.x * matrix[0][0] + in_out.y * matrix[0][1] + in_out.z * matrix[0][2] + matrix[0][3],
			in_out.x * matrix[1][0] + in_out.y * matrix[1][1] + in_out.z * matrix[1][2] + matrix[1][3],
			in_out.x * matrix[2][0] + in_out.y * matrix[2][1] + in_out.z * matrix[2][2] + matrix[2][3]
		};
	}

	// Converts Euler angles (pitch, yaw, roll) to directional vectors
	inline void AngleVectors(const glm::vec3& angles, glm::vec3* forward = nullptr, glm::vec3* right = nullptr, glm::vec3* up = nullptr) {
		// Convert angles to quaternion
		const glm::quat rotation = glm::quat(angles);

		// Default direction vectors
		const glm::vec3 defaultForward = glm::vec3(0, 0, -1);
		const glm::vec3 defaultRight = glm::vec3(1, 0, 0);
		const glm::vec3 defaultUp = glm::vec3(0, 1, 0);

		if (forward) *forward = rotation * defaultForward;
		if (right) *right = rotation * defaultRight;
		if (up) *up = rotation * defaultUp;
	}

	// Calculates the angle in degrees between a point and a line in 3D space.
	inline float DistancePointToLine(const glm::vec3& Point, const glm::vec3& LineOrigin, const glm::vec3& Dir) {
		// Assuming Dir is normalized. If not, normalize it first:
		// Dir = glm::normalize(Dir);

		// Vector from the line origin to the point
		const glm::vec3 PointDir = Point - LineOrigin;

		// Projected length of PointDir onto Dir (scalar value)
		const float projectedLength = glm::dot(PointDir, Dir);

		// Closest point on the line to the given point
		const glm::vec3 ClosestPointOnLine = LineOrigin + Dir * projectedLength;

		// Perpendicular distance from the point to the line
		const float distance = glm::length(Point - ClosestPointOnLine);

		// Distance from LineOrigin to Point
		const float lineLength = glm::length(LineOrigin - Point);

		// Compute the angle in degrees
	#ifdef TRIG_APPROXIMATION
		const float angle = glm::degrees(fast_atan2(distance, lineLength));
	#else
		const float angle = glm::degrees(glm::atan(distance, lineLength));
	#endif
		return angle;
	}

	// Extracts the position component from a 3x4 matrix.
	inline void MatrixPosition(glm::mat3x4& matrix, glm::vec3& out) {
		out = { matrix[0][3], matrix[1][3], matrix[2][3] };
	}

	// Applies a rotation transformation to a 3D vector using a 3x4 matrix.
	inline void VectorRotate(const glm::mat3x4& mat, glm::vec3& out) {
		out = mat * glm::vec4(out, 1.0f);
	}

#ifdef ALTERNATIVE_ROTATIONS
	inline void AngleMatrix(const glm::vec3& angles, glm::mat3x4& matrix) {
		const glm::vec3 rad = glm::radians(angles);
		const glm::vec3 sinValues = glm::sin(rad);
		const glm::vec3 cosValues = glm::cos(rad);

		// Pre-compute common term
		const float sinX_cosY = sinValues.x * cosValues.y;

		matrix[0] = glm::vec4(glm::vec3(cosValues.x * cosValues.y,
			sinValues.z * sinX_cosY + cosValues.z * -sinValues.y,
			cosValues.z * sinX_cosY - sinValues.z * -sinValues.y), 0.0f);

		// Pre-compute common term
		const float sinX_sinY = sinValues.x * sinValues.y;

		matrix[1] = glm::vec4(glm::vec3(cosValues.x * sinValues.y,
			sinValues.z * sinX_sinY + cosValues.z * cosValues.y,
			cosValues.z * sinX_sinY - sinValues.z * cosValues.y), 0.0f);

		matrix[2] = glm::vec4(glm::vec3(-sinValues.x,
			sinValues.z * cosValues.x,
			cosValues.z * cosValues.x), 0.0f);
	}

	inline void VectorRotate2(glm::vec3& in, glm::vec3& out) {
		glm::mat3x4 matRotate;
		AngleMatrix(in, matRotate);
		VectorRotate(matRotate, out);
	}
#endif // ALTERNATIVE_ROTATIONS

	// Checks if a 3D vector is approximately zero within a specified tolerance.
	inline bool IsZero(glm::vec3 vec, float tolerance = 0.01f)
	{
		return (vec.x > -tolerance && vec.x < tolerance &&
			vec.y > -tolerance && vec.y < tolerance &&
			vec.z > -tolerance && vec.z < tolerance);
	}

	// These functions will normalize an angle between -180 / 180
#ifdef NORMALIZE_REMAINDER
	// Warning. This will work however it is not how the Engine does it.
	// fmodf(370, 360) will return 10, because 370 is 10 degrees more than a full circle(360 degrees).
	// remainderf(370, 360) will return -10, because it calculates the shortest distance to the nearest multiple of 360, which in this case is 360 degrees(a full circle), and -10 is the shortest way to get back to this multiple.
	// therefore... you may use this but just know it CAN produce different results than fmodf.
	inline float NormalizeAngle(const float& angle)
	{
		return std::remainderf(angle, 360.f);
	}
#elif defined(NORMALIZE_ANGLE_SOURCE)
	// Source SDK Edition 'AngleNormalize' line 3498
	// https://github.com/ValveSoftware/source-sdk-2013/blob/master/sp/src/mathlib/mathlib_base.cpp
	inline float NormalizeAngle(float angle) {
		angle = std::fmodf(angle, 360.0f);

		if (angle > 180.0f) angle -= 360.0f;
		else if (angle < -180.0f) angle += 360.0f;

		return angle;
	};
#elif defined(NORMALIZE_ANGLE_SOURCE_2)
	// Source 2 Gucci Mane Edition
	inline float AngleNormalize(float deg)
	{
		float f = deg * INV_360 + 0.5f;

		const int i = (int)f;
		const float fi = (float)i;

		if (f < 0.f && f != fi) f = fi - 1.f;
		else f = fi;

		deg -= f * 360.f;

		return deg;
	}
#elif defined(NORMALIZE_ANGLE_SIMPLE_SOURCE_2)
	// Simplified Source 2 Version
	inline float NormalizeAngle(float angle) {
		return angle - glm::floor(angle * INV_360 + 0.5f) * 360.0f;
	};
#elif defined(NORMALIZE_ANGLE_GLM)
	// Does the same thing as the above function but glm version.
	inline float NormalizeAngle(const float& angle)
	{
		// Remove non-finite values (NaN and Infinity)
		//if (!std::isfinite(angle)) angle = 0.0f;

		return glm::mod(angle + 180.f, 360.f) - 180.f;
	}
#elif defined(NORMALIZE_ANGLE_BASIC)
	// Alternative Normalize Method
	// Only use this method if you are sure angle only needs one iteration of correction
	inline float NormalizeAngle(float angle) {
		if (angle > 180.0f) angle -= 360.0f;
		else if (angle < -180.0f) angle += 360.0f;
		return angle;
	};
#endif

	// Override above function for convenience
	inline glm::vec3 NormalizeAngles(glm::vec3 angle) {
		return { glm::clamp(NormalizeAngle(angle.x), -89.0f, 89.0f), glm::clamp(NormalizeAngle(angle.y), -180.0f, 180.0f), 0.0f };
	};

	// Computes the forward vector from a given Euler angle.
	inline glm::vec3 Forward(const glm::vec3& angle) {
		// Convert angles from degrees to radians
		const glm::vec3 rad = glm::radians(angle);

#ifdef TRIG_APPROXIMATION
		// Compute sine and cosine values for pitch and yaw
		const float sy = fast_sin(rad.y);
		const float cy = fast_cos(rad.y);
		const float sp = fast_sin(rad.x);
		const float cp = fast_cos(rad.x);
#else
		// Compute sine and cosine values for pitch and yaw
		const float sy = glm::sin(rad.y);
		const float cy = glm::cos(rad.y);
		const float sp = glm::sin(rad.x);
		const float cp = glm::cos(rad.x);
#endif
		// Calculate and return the forward vector
		return { cp * cy, cp * sy, -sp };
	}

	// Function to calculate the roll angle
#ifdef GET_ROLL_FUNCTIONS
	inline float GetRollAngle(const glm::vec3& vec, const glm::vec3& up) {
		// Compute a vector orthogonal to both 'up' and 'vec' to determine the roll.
		const glm::vec3 left = glm::cross(up, vec);
		return glm::degrees(glm::atan(left.z, (left.y * vec.x) - (left.x * vec.y))); // roll
	}
#endif

	// Converts a 3D vector into Euler angles (pitch, yaw)
	inline glm::vec3 GetAngle(const glm::vec3& vec) {
		// Check for zero vector to prevent division by zero and other undefined behavior.
		// if (glm::length2(vec) < 1e-6f) return ZERO_VEC; // Uncomment if using asin for pitch to prevent division by 0

		// Compute the hypotenuse using the Pythagorean theorem for the 2D plane formed by vec.x and vec.y.
		const float hypotenuse = fast_sqrt(vec.x * vec.x + vec.y * vec.y);

#ifdef TRIG_APPROXIMATION
		glm::vec3 angle = {
			glm::degrees(fast_atan2(-vec.z, hypotenuse)), // pitch | alternative : glm::degrees(-glm::asin(vec.z / hypotenuse)) | second alt: glm::degrees(glm::acos(hypotenuse / glm::length(vec))) | third alt: glm::degrees(-glm::asin(glm::normalize(vec).z))
			glm::degrees(fast_atan2(vec.y, vec.x)),       // yaw
			0.0f                                          // default roll
		};
#else
		glm::vec3 angle = {
			glm::degrees(glm::atan2(-vec.z, hypotenuse)), // pitch | alternative : glm::degrees(-glm::asin(vec.z / hypotenuse)) | second alt: glm::degrees(glm::acos(hypotenuse / glm::length(vec))) | third alt: glm::degrees(-glm::asin(glm::normalize(vec).z))
			glm::degrees(glm::atan2(vec.y, vec.x)),       // yaw
			0.0f                                          // default roll
		};
#endif
		// Normalize angles from -180 180 to 0-360
#ifdef GET_ANGLE_NORMALIZE_360
		if (angle.x < 0) angle.x += 360;
		if (angle.y < 0) angle.y += 360;
#endif
		return angle;
	}
};

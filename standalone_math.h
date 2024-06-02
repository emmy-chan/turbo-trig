//=============================== nostalgia.solutions ====================================//
//
// Purpose: to make math fun and easy to understand... simplify our load on CPU
// for the compromise of tiny accuracy loss. This can be useful for many things that doesn't
// require 100% precision. We can select from different algorithm strengths for our needs.
//
// This library has helpful functions for trigonometry and 3D math.
// 
//=====================================================================================//

#pragma once
#include <xmmintrin.h> // Header for SSE intrinsics
#include <math.h>
#include <limits>

// Define one of these normalize methods
// #define NORMALIZE_REMAINDER
// #define NORMALIZE_ANGLE_SOURCE
// #define NORMALIZE_ANGLE_SOURCE_2
// #define NORMALIZE_ANGLE_SIMPLE_SOURCE_2
// #define NORMALIZE_ANGLE_BASIC

// Use ASM sqrt algorithm or Quake (Default SSE)
// #define ASM_SQRT
// #define QUAKE_SQRT

#if defined (QUAKE_SQRT)
	#include <bit>
#endif

// SQRT_APPROXIMATION: Enables an approximate algorithm for square root calculations.
// This approximation improves performance at the cost of precision. Useful for applications where exact accuracy is less critical.
#define SQRT_APPROXIMATION

// TRIG_APPROXIMATION: Enables approximate calculations for trigonometric functions (sin, cos, etc.).
// These approximations are faster but less precise, suitable for scenarios where speed is prioritized over exact accuracy.
#define TRIG_APPROXIMATION

// Use more simple formula for each trig function
#ifdef TRIG_APPROXIMATION
	 // If faster approximations are needed
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
	const float ZERO_VEC[3] = {0.0f, 0.0f, 0.0f};
	const float NAN_VEC[3] = {NAN, NAN, NAN};

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
		return (-0.69813170079773212f * x * x - 0.87266462599716477f) * x + 1.57079632679f;
	}

	inline float fast_asin(const float& x) {
		return 1.57079632679f - ((-0.69813170079773212f * x * x - 0.87266462599716477f) * x + 1.57079632679f);
	}

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
			if (y >= 0.0f) return 1.57079632679f;
			if (y <= 0.0f) return -1.57079632679f;
		}
		else if (x > 0.0f) {
			if (-x <= y && y <= x) return fast_atan(y / x);
			if (y > x) return 1.57079632679f - fast_atan(x / y);
			else return -1.57079632679f - fast_atan(x / y);
		}
		else {
			if (0.0f <= y && y <= -x) return 3.14159265359f + fast_atan(y / x);
			if (-x < y) return 1.57079632679f - fast_atan(x / y);
			if (x <= y && y <= 0.0f) return fast_atan(y / x) - 3.14159265359f;
			else return -1.57079632679f - fast_atan(x / y);
		}

		return 0.0f;
	}
#else // MORE ACCURATE APPROXIMATIONS
	// Absolute error <= 6.7e-5
	inline float fast_acos(float x) {
		const float negate = float(x < 0);
		x = fabs(x);
		float ret = -0.0187293f;
		ret = ret * x;
		ret = ret + 0.0742610f;
		ret = ret * x;
		ret = ret - 0.2121144f;
		ret = ret * x;
		ret = ret + 1.5707288f;
		ret = ret * fast_sqrt(1.0 - x);
		ret = ret - 2 * negate * ret;
		return negate * 3.14159265359f + ret;
	}

	inline float fast_asin(float x) {
		const float negate = float(x < 0);
		x = fabs(x);
		float ret = -0.0187293f;
		ret *= x;
		ret += 0.0742610f;
		ret *= x;
		ret -= 0.2121144f;
		ret *= x;
		ret += 1.5707288f;
		ret = 1.57079632679f - fast_sqrt(1.0 - x) * ret;
		return ret - 2 * negate * ret;
	}

	inline float fast_atan2(const float& y, const float& x) {
		float t3 = fabs(x);
		float t1 = fabs(y);
		float t0 = fmax(t3, t1);
		t1 = fmin(t3, t1);
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

		t3 = (fabs(y) > fabs(x)) ? 1.57079632679f - t3 : t3;
		t3 = (x < 0) ? 3.14159265359f - t3 : t3;
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
		return fast_cos(1.57079632679f - x);
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
	inline bool IsMatrixValid(const float matrix[4][4]) {
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j)
				if (!std::isfinite(matrix[i][j])) return false;
		}
		return true; // All values are finite numbers.
	}

	// Transforms a 3D vector by a 3x4 matrix.
	inline void VectorTransform(float in_out[3], const float matrix[3][4]) {
		float x = in_out[0] * matrix[0][0] + in_out[1] * matrix[0][1] + in_out[2] * matrix[0][2] + matrix[0][3];
		float y = in_out[0] * matrix[1][0] + in_out[1] * matrix[1][1] + in_out[2] * matrix[1][2] + matrix[1][3];
		float z = in_out[0] * matrix[2][0] + in_out[1] * matrix[2][1] + in_out[2] * matrix[2][2] + matrix[2][3];
		in_out[0] = x;
		in_out[1] = y;
		in_out[2] = z;
	}

	// Converts Euler angles (pitch, yaw, roll) to directional vectors
	inline void AngleVectors(const float angles[3], float forward[3] = nullptr, float right[3] = nullptr, float up[3] = nullptr) {
		// Convert angles to radians
		const float pitch = angles[0] * 0.0174532925f; // degrees to radians
		const float yaw = angles[1] * 0.0174532925f;   // degrees to radians
		const float roll = angles[2] * 0.0174532925f;  // degrees to radians

		const float cp = cosf(pitch);
		const float sp = sinf(pitch);
		const float cy = cosf(yaw);
		const float sy = sinf(yaw);
		const float cr = cosf(roll);
		const float sr = sinf(roll);

		if (forward) {
			forward[0] = cp * cy;
			forward[1] = cp * sy;
			forward[2] = -sp;
		}

		if (right) {
			right[0] = -1 * sr * sp * cy + -1 * cr * -sy;
			right[1] = -1 * sr * sp * sy + -1 * cr * cy;
			right[2] = -1 * sr * cp;
		}

		if (up) {
			up[0] = cr * sp * cy + -sr * -sy;
			up[1] = cr * sp * sy + -sr * cy;
			up[2] = cr * cp;
		}
	}

	// Calculates the angle in degrees between a point and a line in 3D space.
	inline float DistancePointToLine(const float Point[3], const float LineOrigin[3], const float Dir[3]) {
		// Vector from the line origin to the point
		const float PointDir[3] = { Point[0] - LineOrigin[0], Point[1] - LineOrigin[1], Point[2] - LineOrigin[2] };

		// Projected length of PointDir onto Dir (scalar value)
		const float projectedLength = PointDir[0] * Dir[0] + PointDir[1] * Dir[1] + PointDir[2] * Dir[2];

		// Closest point on the line to the given point
		const float ClosestPointOnLine[3] = { LineOrigin[0] + Dir[0] * projectedLength, LineOrigin[1] + Dir[1] * projectedLength, LineOrigin[2] + Dir[2] * projectedLength };

		// Perpendicular distance from the point to the line
		const float distance = fast_sqrt((Point[0] - ClosestPointOnLine[0]) * (Point[0] - ClosestPointOnLine[0]) +
			(Point[1] - ClosestPointOnLine[1]) * (Point[1] - ClosestPointOnLine[1]) +
			(Point[2] - ClosestPointOnLine[2]) * (Point[2] - ClosestPointOnLine[2]));

		// Distance from LineOrigin to Point
		const float lineLength = fast_sqrt((LineOrigin[0] - Point[0]) * (LineOrigin[0] - Point[0]) +
			(LineOrigin[1] - Point[1]) * (LineOrigin[1] - Point[1]) +
			(LineOrigin[2] - Point[2]) * (LineOrigin[2] - Point[2]));

		// Compute the angle in degrees
	#ifdef TRIG_APPROXIMATION
		const float angle = fast_atan2(distance, lineLength) * 57.2957795131f; // radians to degrees
	#else
		const float angle = atan2(distance, lineLength) * 57.2957795131f; // radians to degrees
	#endif
		return angle;
	}

	// Extracts the position component from a 3x4 matrix.
	inline void MatrixPosition(float matrix[3][4], float out[3]) {
		out[0] = matrix[0][3];
		out[1] = matrix[1][3];
		out[2] = matrix[2][3];
	}

	// Applies a rotation transformation to a 3D vector using a 3x4 matrix.
	inline void VectorRotate(const float mat[3][4], float out[3]) {
		out[0] = mat[0][0] * out[0] + mat[0][1] * out[1] + mat[0][2] * out[2];
		out[1] = mat[1][0] * out[0] + mat[1][1] * out[1] + mat[1][2] * out[2];
		out[2] = mat[2][0] * out[0] + mat[2][1] * out[1] + mat[2][2] * out[2];
	}

#ifdef ALTERNATIVE_ROTATIONS
	inline void AngleMatrix(const float angles[3], float matrix[3][4]) {
		const float rad[3] = { angles[0] * 0.0174532925f, angles[1] * 0.0174532925f, angles[2] * 0.0174532925f }; // degrees to radians
		const float sinValues[3] = { sinf(rad[0]), sinf(rad[1]), sinf(rad[2]) };
		const float cosValues[3] = { cosf(rad[0]), cosf(rad[1]), cosf(rad[2]) };

		// Pre-compute common term
		const float sinX_cosY = sinValues[0] * cosValues[1];

		matrix[0][0] = cosValues[0] * cosValues[1];
		matrix[0][1] = sinValues[2] * sinX_cosY + cosValues[2] * -sinValues[1];
		matrix[0][2] = cosValues[2] * sinX_cosY - sinValues[2] * -sinValues[1];

		// Pre-compute common term
		const float sinX_sinY = sinValues[0] * sinValues[1];

		matrix[1][0] = cosValues[0] * sinValues[1];
		matrix[1][1] = sinValues[2] * sinX_sinY + cosValues[2] * cosValues[1];
		matrix[1][2] = cosValues[2] * sinX_sinY - sinValues[2] * cosValues[1];

		matrix[2][0] = -sinValues[0];
		matrix[2][1] = sinValues[2] * cosValues[0];
		matrix[2][2] = cosValues[2] * cosValues[0];

		matrix[0][3] = 0.0f;
		matrix[1][3] = 0.0f;
		matrix[2][3] = 0.0f;
	}

	inline void VectorRotate2(float in[3], float out[3]) {
		float matRotate[3][4];
		AngleMatrix(in, matRotate);
		VectorRotate(matRotate, out);
	}
#endif // ALTERNATIVE_ROTATIONS

	// Checks if a 3D vector is approximately zero within a specified tolerance.
	inline bool IsZero(float vec[3], float tolerance = 0.01f)
	{
		return (vec[0] > -tolerance && vec[0] < tolerance &&
			vec[1] > -tolerance && vec[1] < tolerance &&
			vec[2] > -tolerance && vec[2] < tolerance);
	}

	// These functions will normalize an angle between -180 / 180
#ifdef NORMALIZE_REMAINDER
	inline float NormalizeAngle(const float& angle)
	{
		return std::remainderf(angle, 360.f);
	}
#elif defined(NORMALIZE_ANGLE_SOURCE)
	inline float NormalizeAngle(float angle) {
		angle = std::fmodf(angle, 360.0f);

		if (angle > 180.0f) angle -= 360.0f;
		else if (angle < -180.0f) angle += 360.0f;

		return angle;
	};
#elif defined(NORMALIZE_ANGLE_SOURCE_2)
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
	inline float NormalizeAngle(float angle) {
		return angle - floorf(angle * INV_360 + 0.5f) * 360.0f;
	};
#elif defined(NORMALIZE_ANGLE_BASIC)
	inline float NormalizeAngle(float angle) {
		if (angle > 180.0f) angle -= 360.0f;
		else if (angle < -180.0f) angle += 360.0f;
		return angle;
	};
#endif

	// Override above function for convenience
	inline void NormalizeAngles(float angle[3]) {
		angle[0] = fmaxf(-89.0f, fminf(89.0f, NormalizeAngle(angle[0])));
		angle[1] = fmaxf(-180.0f, fminf(180.0f, NormalizeAngle(angle[1])));
		angle[2] = 0.0f;
	};

	// Computes the forward vector from a given Euler angle.
	inline void Forward(const float angle[3], float forward[3]) {
		// Convert angles from degrees to radians
		const float rad[3] = { angle[0] * 0.0174532925f, angle[1] * 0.0174532925f, angle[2] * 0.0174532925f }; // degrees to radians

	#ifdef TRIG_APPROXIMATION
		// Compute sine and cosine values for pitch and yaw
		const float sy = fast_sin(rad[1]);
		const float cy = fast_cos(rad[1]);
		const float sp = fast_sin(rad[0]);
		const float cp = fast_cos(rad[0]);
	#else
		// Compute sine and cosine values for pitch and yaw
		const float sy = sinf(rad[1]);
		const float cy = cosf(rad[1]);
		const float sp = sinf(rad[0]);
		const float cp = cosf(rad[0]);
	#endif
		// Calculate and return the forward vector
		forward[0] = cp * cy;
		forward[1] = cp * sy;
		forward[2] = -sp;
	}

	// Function to calculate the roll angle
#ifdef GET_ROLL_FUNCTIONS
	inline float GetRollAngle(const float vec[3], const float up[3]) {
		// Compute a vector orthogonal to both 'up' and 'vec' to determine the roll.
		const float left[3] = { up[1] * vec[2] - up[2] * vec[1], up[2] * vec[0] - up[0] * vec[2], up[0] * vec[1] - up[1] * vec[0] };
		return atan2f(left[2], (left[1] * vec[0]) - (left[0] * vec[1])) * 57.2957795131f; // roll
	}
#endif

	// Converts a 3D vector into Euler angles (pitch, yaw)
	inline void GetAngle(const float vec[3], float angle[3]) {
		// Compute the hypotenuse using the Pythagorean theorem for the 2D plane formed by vec.x and vec.y.
		const float hypotenuse = fast_sqrt(vec[0] * vec[0] + vec[1] * vec[1]);

	#ifdef TRIG_APPROXIMATION
		angle[0] = fast_atan2(-vec[2], hypotenuse) * 57.2957795131f; // pitch
		angle[1] = fast_atan2(vec[1], vec[0]) * 57.2957795131f;       // yaw
	#else
		angle[0] = atan2f(-vec[2], hypotenuse) * 57.2957795131f; // pitch
		angle[1] = atan2f(vec[1], vec[0]) * 57.2957795131f;       // yaw
	#endif
		angle[2] = 0.0f;                                          // default roll

		// Normalize angles from -180 180 to 0-360
	#ifdef GET_ANGLE_NORMALIZE_360
		if (angle[0] < 0) angle[0] += 360;
		if (angle[1] < 0) angle[1] += 360;
	#endif
	}
};

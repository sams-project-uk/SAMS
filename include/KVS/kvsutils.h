#ifndef KVS_UTILS_H
#define KVS_UTILS_H

#include <sstream>
#include <iomanip>
#include <limits>

namespace SAMS{

	namespace KVSConversion
	{
		inline std::string floatToString(float value)
		{
			//If between 1e-4 and 1e4 used fixed notation, otherwise use scientific notation.
			//Keep the scientific notation to maximum precision for a float
			if (std::abs(value) >= 1e-4 && std::abs(value) < 1e4)
			{
				return std::to_string(value);
			}
			else
			{
				std::ostringstream oss;
				oss.precision(std::numeric_limits<float>::max_digits10);
				oss << std::scientific << value;
				return oss.str();
			}
		}

		inline std::string doubleToString(double value)
		{
			//If between 1e-4 and 1e4 used fixed notation, otherwise use scientific notation.
			//Keep the scientific notation to maximum precision for a double
			if (std::abs(value) >= 1e-4 && std::abs(value) < 1e4)
			{
				return std::to_string(value);
			}
			else
			{
				std::ostringstream oss;
				oss.precision(std::numeric_limits<double>::max_digits10);
				oss << std::scientific << value;
				return oss.str();
			}
		}

		inline std::string boolToString(bool value)
		{
			//true and false are read as booleans by both the JSON and EPOCH readers
			return value ? "true" : "false";
		}
	}
}

#endif

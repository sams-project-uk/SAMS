/*
* This is a two-fluid routine for all the multi-fluid physics modules
*/

#include <cstdint>
#include <cassert>
#include <string>
#include "constants.h"
#include "pp/parallelWrapper.h"
#include "remapData.h"
#include "typedefs.h"
#include "mpiManager.h"
#include "variableDef.h"
#include "harness.h"
#include "runner.h"
#include "io/writerProto.h"

class TwoFluid {
public:
    static constexpr std::string_view name = "TwoFluid";
    // other member variables and functions
};

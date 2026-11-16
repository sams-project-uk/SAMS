#ifndef SAMS_COMMONNAMES_H
#define SAMS_COMMONNAMES_H

#include <cstdint>

namespace SAMS
{

    /**
     * Values of runnerOrder for packages whose hooks must run in a fixed order. Lower values run first
     */
    namespace fixedOrders
    {
        //Solvers, which read the values that initial conditions can override
        inline constexpr uint8_t solver = 128;
        //Initial conditions, which run after the solvers
        inline constexpr uint8_t initialConditions = 192;
    }

    /**
     * Names of packages that other code refers to
     */
    namespace packages
    {
        inline constexpr char outputControl[] = "OutputControl";
        inline constexpr char simulationTimeTerminate[] = "SimulationTimeTerminate";
    }

    /**
     * Deck keys that more than one package or the runner use
     */
    namespace deckKeys
    {
        //End time of the simulation
        inline constexpr char endTime[] = "/time/t_end";
        //Maximum number of steps. Negative for no limit
        inline constexpr char stepLimit[] = "/time/nsteps";
        //Writer of the output
        inline constexpr char diagnosticsFormat[] = "/diagnostics/format";
        //Dump that the run restarts from
        inline constexpr char restart[] = "/SAMS/restart";
        //Number of cells in each direction
        inline constexpr char nx[] = "/domain/nx";
        inline constexpr char ny[] = "/domain/ny";
        inline constexpr char nz[] = "/domain/nz";
        //Limits of the domain in each direction
        inline constexpr char xMin[] = "/domain/x_min";
        inline constexpr char xMax[] = "/domain/x_max";
        inline constexpr char yMin[] = "/domain/y_min";
        inline constexpr char yMax[] = "/domain/y_max";
        inline constexpr char zMin[] = "/domain/z_min";
        inline constexpr char zMax[] = "/domain/z_max";

        //Description of the run
        inline constexpr char description[] = "/SAMS/description";
    }

}
#endif //SAMS_COMMONNAMES_H

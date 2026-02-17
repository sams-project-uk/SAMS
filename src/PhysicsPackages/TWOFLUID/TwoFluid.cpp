/*
* This is a two-fluid routine for all the multi-fluid physics modules
*/

//////////////////////No idea which ones of these are needed
#include <iostream>
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

#include "twofluid.h"

#include "shared_data.h"
#include "variableRegistry.h"
#include "axisRegistry.h"

namespace TWOFLUID
{
    namespace pw = portableWrapper;
    
    struct two_fluid_properties
    {
        bool collisions=true;
        bool ion_rec_empirical=false;
        bool ion_rec_nlevel=false;
    };

    struct data_two_fluid_source
    {
        LARE::volumeArray source_mass; // mass source term
        LARE::volumeArray source_v_x; // velocity source term
        LARE::volumeArray source_v_y; // velocity source term
        LARE::volumeArray source_v_z; // velocity source term
        LARE::volumeArray source_energy; // energy source term
        LARE::volumeArray source_electron_energy; // energy source term
        LARE::volumeArray ac; //coupling coeficient
    };
    void get_ac(LARE::simulationData &data, LARE::simulationData &dataNeutral, data_two_fluid_source &plasma_source);
    void two_fluid_source(LARE::simulationData &data,LARE::simulationData &dataNeutral);
    void ion_rec_rates_empirical(LARE::simulationData &data, LARE::simulationData &dataNeutral);
    void get_collisional_source_terms(LARE::simulationData &data, LARE::simulationData &dataNeutral, data_two_fluid_source &plasma_source, data_two_fluid_source &neutral_source);
    void get_ion_rec_source_terms(LARE::simulationData &data, LARE::simulationData &dataNeutral, data_two_fluid_source &plasma_source, data_two_fluid_source &neutral_source);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //static constexpr std::string_view name = "TwoFluid";
    // other member variables and functions
    void PIP::initialize(){
    //Just set up internal state
    //SAMS::debugAll3 << "Initialising twofluid" 
    //                << std::endl;
    SAMS::cout << "Initialising twofluid" << std::endl;
    }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //void registerAxes(SAMS::harness &harnessRef);
    
    /*
    * Register variables for the neutral fluid
    * Magnetic field is an odd one. It needs to be registered for the LARE core code to work but is identically zero always
    * Is there a memory-efficient way of doing this?
    */
    void PIP::registerVariables(SAMS::harness& harness){
        SAMS::cout << "Registering twofluid variables" << std::endl;
        auto &varRegistry = harness.variableRegistry;   // Take care to remember the reference marker & in this idiom!
        const int ghosts = 2; // 2 Ghost cells at top and bottom of each dimension
        //Register density (cell centred on all axes)
        varRegistry.registerVariable<LARE::T_dataType>("rho_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        varRegistry.registerVariable<LARE::T_dataType>("energy_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));

        //Register x velocity (face centred on all axes)
        varRegistry.registerVariable<LARE::T_dataType>("vx_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
        varRegistry.registerVariable<LARE::T_dataType>("vy_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
        varRegistry.registerVariable<LARE::T_dataType>("vz_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));

        varRegistry.registerVariable<LARE::T_dataType>("LARE/vx1_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));

        varRegistry.registerVariable<LARE::T_dataType>("LARE/vy1_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));

        varRegistry.registerVariable<LARE::T_dataType>("LARE/vz1_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));

        //Register By (face centred on Y, cell centred on X and Z)
        varRegistry.registerVariable<LARE::T_dataType>("bx_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts));
        varRegistry.registerVariable<LARE::T_dataType>("by_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts));
        varRegistry.registerVariable<LARE::T_dataType>("bz_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts));
    }
            
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
    * Default values
    */
    void PIP::defaultValues(LARE::simulationData &data,LARE::simulationData &dataNeutral){
        SAMS::cout << "Setting default values" << std::endl;
        dataNeutral.alpha0=1.0;
        dataNeutral.is_neutral=true;
        SAMS::cout << "dataNeutral=" << dataNeutral.is_neutral << std::endl;
        SAMS::cout << "data=" << data.is_neutral << std::endl;
        //printf("Here \n");
    }


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*void PIP::allocate_neutral(SAMS::harness &harness, LARE::simulationData &data)
    {
        //pw::portableArrayManager& manager;
        
        LARE::T_sizeType nx, ny, nz;

        auto &axRegistry = harness.axisRegistry;
        auto &varRegistry = harness.variableRegistry;
        // Centred since LARE thinks in terms of cell centres for nx, ny, nz
        nx = axRegistry.getLocalDomainElements("X", SAMS::staggerType::CENTRED);
        ny = axRegistry.getLocalDomainElements("Y", SAMS::staggerType::CENTRED);
        nz = axRegistry.getLocalDomainElements("Z", SAMS::staggerType::CENTRED);

        // Get the ranges for the whole local domain
        data.xcLocalRange = axRegistry.getLocalRange("X", SAMS::staggerType::CENTRED);
        data.ycLocalRange = axRegistry.getLocalRange("Y", SAMS::staggerType::CENTRED);
        data.zcLocalRange = axRegistry.getLocalRange("Z", SAMS::staggerType::CENTRED);
        data.xbLocalRange = axRegistry.getLocalRange("X", SAMS::staggerType::HALF_CELL);
        data.ybLocalRange = axRegistry.getLocalRange("Y", SAMS::staggerType::HALF_CELL);
        data.zbLocalRange = axRegistry.getLocalRange("Z", SAMS::staggerType::HALF_CELL);

        // Get the ranges for the ghost cells for centred axes
        data.xcminBCRange = axRegistry.getLocalNonDomainRange("X", SAMS::staggerType::CENTRED, SAMS::domain::edges::lower );
        data.xcmaxBCRange = axRegistry.getLocalNonDomainRange("X", SAMS::staggerType::CENTRED, SAMS::domain::edges::upper);
        data.ycminBCRange = axRegistry.getLocalNonDomainRange("Y", SAMS::staggerType::CENTRED, SAMS::domain::edges::lower);
        data.ycmaxBCRange = axRegistry.getLocalNonDomainRange("Y", SAMS::staggerType::CENTRED, SAMS::domain::edges::upper);
        data.zcminBCRange = axRegistry.getLocalNonDomainRange("Z", SAMS::staggerType::CENTRED, SAMS::domain::edges::lower);
        data.zcmaxBCRange = axRegistry.getLocalNonDomainRange("Z", SAMS::staggerType::CENTRED, SAMS::domain::edges::upper);

        // Get the ranges for the ghost cells for half cell staggered axes
        data.xbminBCRange = axRegistry.getLocalNonDomainRange("X", SAMS::staggerType::HALF_CELL, SAMS::domain::edges::lower);
        data.xbmaxBCRange = axRegistry.getLocalNonDomainRange("X", SAMS::staggerType::HALF_CELL, SAMS::domain::edges::upper);
        data.ybminBCRange = axRegistry.getLocalNonDomainRange("Y", SAMS::staggerType::HALF_CELL, SAMS::domain::edges::lower);
        data.ybmaxBCRange = axRegistry.getLocalNonDomainRange("Y", SAMS::staggerType::HALF_CELL, SAMS::domain::edges::upper);
        data.zbminBCRange = axRegistry.getLocalNonDomainRange("Z", SAMS::staggerType::HALF_CELL, SAMS::domain::edges::lower);
        data.zbmaxBCRange = axRegistry.getLocalNonDomainRange("Z", SAMS::staggerType::HALF_CELL, SAMS::domain::edges::upper);


        // Get the ranges for the actual domain (no ghost cells)
        data.xcLocalDomainRange = axRegistry.getLocalDomainRange("X", SAMS::staggerType::CENTRED);
        data.ycLocalDomainRange = axRegistry.getLocalDomainRange("Y", SAMS::staggerType::CENTRED);
        data.zcLocalDomainRange = axRegistry.getLocalDomainRange("Z", SAMS::staggerType::CENTRED);
        data.xbLocalDomainRange = axRegistry.getLocalDomainRange("X", SAMS::staggerType::HALF_CELL);
        data.ybLocalDomainRange = axRegistry.getLocalDomainRange("Y", SAMS::staggerType::HALF_CELL);
        data.zbLocalDomainRange = axRegistry.getLocalDomainRange("Z", SAMS::staggerType::HALF_CELL);

        data.nx = nx;
        data.ny = ny;
        data.nz = nz;

        data.nx_global = axRegistry.getDimension("X").getGlobalDomainCount(SAMS::staggerType::CENTRED);
        data.ny_global = axRegistry.getDimension("Y").getGlobalDomainCount(SAMS::staggerType::CENTRED);
        data.nz_global = axRegistry.getDimension("Z").getGlobalDomainCount(SAMS::staggerType::CENTRED);

        using Range = pw::Range;
        // Grab the final variable sizes from the registry and wrap the arrays
        //varRegistry.fillPPArray("energy_electron", data.energy_electron);
        //pw::assign(data.energy_electron, 0.0);
        //varRegistry.fillPPArray("energy_ion", data.energy_ion);
        //pw::assign(data.energy_ion, 0.0);
        varRegistry.fillPPArray("energy_neutral", data.energy_neutral);
        pw::assign(data.energy_neutral, 0.0);
        varRegistry.fillPPArray("rho_n", data.rho);
        pw::assign(data.rho, 0.0);
        varRegistry.fillPPArray("vx_n", data.vx);
        pw::assign(data.vx, 0.0);
        varRegistry.fillPPArray("vy_n", data.vy);
        pw::assign(data.vy, 0.0);
        varRegistry.fillPPArray("vz_n", data.vz);
        pw::assign(data.vz, 0.0);
        varRegistry.fillPPArray("bx_n", data.bx);
        pw::assign(data.bx, 0.0);
        varRegistry.fillPPArray("by_n", data.by);
        pw::assign(data.by, 0.0);
        varRegistry.fillPPArray("bz_n", data.bz);
        pw::assign(data.bz, 0.0);
        varRegistry.fillPPArray("LARE/vx1_n", data.vx1);
        pw::assign(data.vx1, 0.0);
        varRegistry.fillPPArray("LARE/vy1_n", data.vy1);
        pw::assign(data.vy1, 0.0);
        varRegistry.fillPPArray("LARE/vz1_n", data.vz1);
        pw::assign(data.vz1, 0.0);
        varRegistry.fillPPArray("LARE/dm_n", data.dm);
        pw::assign(data.dm, 0.0);

        data.isxLB = harness.MPIManager.isEdge(0, SAMS::domain::edges::lower);
        data.isxUB = harness.MPIManager.isEdge(0, SAMS::domain::edges::upper);
        data.isyLB = harness.MPIManager.isEdge(1, SAMS::domain::edges::lower);
        data.isyUB = harness.MPIManager.isEdge(1, SAMS::domain::edges::upper);
        data.iszLB = harness.MPIManager.isEdge(2, SAMS::domain::edges::lower);
        data.iszUB = harness.MPIManager.isEdge(2, SAMS::domain::edges::upper);

        SAMS::debugAll3 << "Edge detection: "
                        << " XLB: " << data.isxLB << " XUB: " << data.isxUB
                        << " YLB: " << data.isyLB << " YUB: " << data.isyUB
                        << " ZLB: " << data.iszLB << " ZUB: " << data.iszUB
                        << std::endl;

        manager.allocate(data.p_visc, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.eta, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.dxab, data.xbLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.dyab, data.xcLocalRange, data.ybLocalRange, data.zcLocalRange);
        manager.allocate(data.dzab, data.xcLocalRange, data.ycLocalRange, data.zbLocalRange);
        manager.allocate(data.dxac, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.dyac, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.dzac, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.cv, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.cv1, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.cvc, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);

        Range xcp = pw::Range(0, data.nx + 1);
        Range ycp = pw::Range(0, data.ny + 1);
        Range zcp = pw::Range(0, data.nz + 1);
        Range ycpp = pw::Range(0, data.ny + 2);
        Range zcpp = pw::Range(0, data.nz + 2);
        Range xbp = pw::Range(-1, data.nx + 1);
        Range ybp = pw::Range(-1, data.ny + 1);
        Range zbp = pw::Range(-1, data.nz + 1);
        // Allocate arrays using the portableArrayManager
        manager.allocate(data.bx1, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.by1, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.bz1, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.alpha1, xcp, ycpp, zcpp);
        manager.allocate(data.alpha2, xbp, ycp, zcpp);
        manager.allocate(data.alpha3, data.xcLocalRange, data.ycLocalRange, zcp);
        manager.allocate(data.visc_heat, xcp, ycp, zcp);
        manager.allocate(data.pressure, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.p_e, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.p_i, data.xcLocalRange, data.ycLocalRange, data.zcLocalRange);
        manager.allocate(data.rho_v, xbp, ybp, zbp);
        manager.allocate(data.cv_v, xbp, ybp, zbp);
        manager.allocate(data.fx, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);
        manager.allocate(data.fy, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);
        manager.allocate(data.fz, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);
        manager.allocate(data.fx_visc, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);
        manager.allocate(data.fy_visc, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);
        manager.allocate(data.fz_visc, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);
        manager.allocate(data.flux_x, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);
        manager.allocate(data.flux_y, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);
        manager.allocate(data.flux_z, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);
        manager.allocate(data.curlb, data.xbLocalDomainRange, data.ybLocalDomainRange, data.zbLocalDomainRange);

        axRegistry.fillPPLocalAxis("X", data.xc, SAMS::staggerType::CENTRED);
        axRegistry.fillPPLocalAxis("Y", data.yc, SAMS::staggerType::CENTRED);
        axRegistry.fillPPLocalAxis("Z", data.zc, SAMS::staggerType::CENTRED);
        axRegistry.fillPPLocalAxis("X", data.xb, SAMS::staggerType::HALF_CELL);
        axRegistry.fillPPLocalAxis("Y", data.yb, SAMS::staggerType::HALF_CELL);
        axRegistry.fillPPLocalAxis("Z", data.zb, SAMS::staggerType::HALF_CELL);
        axRegistry.fillPPLocalAxis("X", data.xb_host, SAMS::staggerType::HALF_CELL);
        axRegistry.fillPPLocalAxis("Y", data.yb_host, SAMS::staggerType::HALF_CELL);
        axRegistry.fillPPLocalAxis("Z", data.zb_host, SAMS::staggerType::HALF_CELL);
        axRegistry.fillPPLocalAxis("X", data.xc_host, SAMS::staggerType::CENTRED);
        axRegistry.fillPPLocalAxis("Y", data.yc_host, SAMS::staggerType::CENTRED);
        axRegistry.fillPPLocalAxis("Z", data.zc_host, SAMS::staggerType::CENTRED);
        axRegistry.fillPPAxis("X", data.xb_global, SAMS::staggerType::CENTRED);
        axRegistry.fillPPAxis("Y", data.yb_global, SAMS::staggerType::CENTRED);
        axRegistry.fillPPAxis("Z", data.zb_global, SAMS::staggerType::CENTRED);

        axRegistry.fillPPLocalDelta("X", data.dxc, SAMS::staggerType::CENTRED);
        axRegistry.fillPPLocalDelta("Y", data.dyc, SAMS::staggerType::CENTRED);
        axRegistry.fillPPLocalDelta("Z", data.dzc, SAMS::staggerType::CENTRED);
        axRegistry.fillPPLocalDelta("X", data.dxb, SAMS::staggerType::HALF_CELL);
        axRegistry.fillPPLocalDelta("Y", data.dyb, SAMS::staggerType::HALF_CELL);
        axRegistry.fillPPLocalDelta("Z", data.dzb, SAMS::staggerType::HALF_CELL);

        manager.allocate(data.hy, Range(-2, nx + 2));
        manager.allocate(data.hz, Range(-2, nx + 2), Range(-2, ny + 2));
        manager.allocate(data.hyc, Range(-1, nx + 2));
        manager.allocate(data.hzc, Range(-1, nx + 2), Range(-1, ny + 2));
        manager.allocate(data.hz1, Range(-2, nx + 2), Range(-1, ny + 2));
        manager.allocate(data.hz2, Range(-1, nx + 2), Range(-2, ny + 2));
        manager.allocate(data.x, Range(-2, nx + 2), Range(-2, ny + 2), Range(-2, nz + 2));
        manager.allocate(data.y, Range(-2, nx + 2), Range(-2, ny + 2), Range(-2, nz + 2));
        manager.allocate(data.z, Range(-2, nx + 2), Range(-2, ny + 2), Range(-2, nz + 2));
        manager.allocate(data.xp, Range(-2, nx + 2), Range(-2, ny + 2), Range(-2, nz + 2));
        manager.allocate(data.yp, Range(-2, nx + 2), Range(-2, ny + 2), Range(-2, nz + 2));
        manager.allocate(data.zp, Range(-2, nx + 2), Range(-2, ny + 2), Range(-2, nz + 2));
        if (data.rke)
        {
            manager.allocate(data.delta_ke, Range(-1, nx + 2), Range(-1, ny + 2), Range(-1, nz + 2));
        }

        data.mpiType = SAMS::gettypeRegistry().getMPIType<LARE::T_dataType>();
    }
    */
////////////////////////////////////////////////////////////////////////////////////////
    void PIP::two_fluid_source(LARE::simulationData &data,LARE::simulationData &dataNeutral){

        //data.two_fluid_timestep=1.0;
        
        data_two_fluid_source plasma_source;
        data_two_fluid_source neutral_source;
        two_fluid_properties two_fluid_flags;
        
        portableWrapper::portableArrayManager SourceManager;
        using Range = portableWrapper::Range;
        
        SourceManager.allocate(plasma_source.ac, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(plasma_source.source_mass, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(plasma_source.source_v_x, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(plasma_source.source_v_y, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(plasma_source.source_v_z, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(plasma_source.source_energy, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(plasma_source.source_electron_energy, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(neutral_source.source_mass, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(neutral_source.source_v_x, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(neutral_source.source_v_y, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(neutral_source.source_v_z, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(neutral_source.source_energy, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        
        portableWrapper::assign(plasma_source.source_mass,0.0);
        portableWrapper::assign(plasma_source.source_v_x,0.0);
        portableWrapper::assign(plasma_source.source_v_y,0.0);
        portableWrapper::assign(plasma_source.source_v_z,0.0);
        portableWrapper::assign(plasma_source.source_energy,0.0);
        portableWrapper::assign(plasma_source.source_electron_energy,0.0);
        portableWrapper::assign(neutral_source.source_mass,0.0);
        portableWrapper::assign(neutral_source.source_v_x,0.0);
        portableWrapper::assign(neutral_source.source_v_y,0.0);
        portableWrapper::assign(neutral_source.source_v_z,0.0);
        portableWrapper::assign(neutral_source.source_energy,0.0);
        portableWrapper::assign(plasma_source.ac,0.0);
        
        //Get collisional coefficient
        get_ac(data,dataNeutral,plasma_source);
        
        
        //Get the ionisation rates
        if (two_fluid_flags.ion_rec_empirical){        
            ion_rec_rates_empirical(data,dataNeutral);
        }
        if (two_fluid_flags.ion_rec_nlevel){        
        //    ion_rec_rates_nlevel(data,dataNeutral);
        }
        
        //Calculate the source terms for the two-fluid interactions
        get_collisional_source_terms(data,dataNeutral,plasma_source,neutral_source);
        
        //Calculate the source terms for Ionisation/recombination
        if (two_fluid_flags.ion_rec_empirical) get_ion_rec_source_terms(data,dataNeutral,plasma_source,neutral_source);
        
        
        // Make sure the timestep is the same in both species NEEDS MOVING ELSEWHERE
        //Set dt to be the minimum of the neutral and plasma times
        /*if (first_step){        
            //Set the timestep for the collisions
            set_dt_collisional(data, dataNeutral, plasma_ir_source);
            
            //Set the ionisation/recombination timestep
            if (data.ion_rec_empirical) set_dt_ion_rec(data,dataNeutral);
            
            printf("dt (plasma, neutral, two-fluid)=%f %f %f \n",data.dt,dataNeutral.dt,data.two_fluid_timestep);
            
            data.dt=std::min({dataNeutral.dt,data.dt,data.two_fluid_timestep});
            dataNeutral.dt=data.dt;
        }
        */
        if (two_fluid_flags.collisions){
            using Range = portableWrapper::Range;
            portableWrapper::applyKernel(LAMBDA(LARE::T_indexType ix, LARE::T_indexType iy, LARE::T_indexType iz) {
                //Get Temperatures
                //T_dataType temperature_ion = data.gas_gamma*data.energy_ion(ix,iy,iz)*(data.gas_gamma-1.0);
                //T_dataType temperature_electron = data.gas_gamma*data.energy_electron(ix,iy,iz)*(data.gas_gamma-1.0);
                //T_dataType temperature_neutral = data.gas_gamma*dataNeutral.energy_neutral(ix,iy,iz)*(data.gas_gamma-1.0);
                
                //T_dataType ac;
                //get_ac(data.alpha0,temperature_ion,temperature_neutral);
                
                //Note that the factor of 0.5 in these is due to Strang splitting
                //Mass exchange terms
                data.rho(ix,iy,iz)+=0.5*data.dt*plasma_source.source_mass(ix,iy,iz);
                dataNeutral.rho(ix,iy,iz)+=0.5*data.dt*neutral_source.source_mass(ix,iy,iz);
                
                //Apply the velocity exchange terms
                data.vx(ix,iy,iz)       +=0.5*data.dt*plasma_source.source_v_x(ix,iy,iz);
                dataNeutral.vx(ix,iy,iz)+=0.5*data.dt*neutral_source.source_v_x(ix,iy,iz);
                
                data.vy(ix,iy,iz)       +=0.5*data.dt*plasma_source.source_v_y(ix,iy,iz);
                dataNeutral.vy(ix,iy,iz)+=0.5*data.dt*neutral_source.source_v_y(ix,iy,iz);
                
                data.vz(ix,iy,iz)       +=0.5*data.dt*plasma_source.source_v_z(ix,iy,iz);
                dataNeutral.vz(ix,iy,iz)+=0.5*data.dt*neutral_source.source_v_z(ix,iy,iz);
                
                //Energy source terms - the 3/2 here needs fixing
                data.energy_ion(ix,iy,iz)+=0.5*data.dt*plasma_source.source_energy(ix,iy,iz);
                data.energy_electron(ix,iy,iz)+=0.5*data.dt*plasma_source.source_electron_energy(ix,iy,iz);
                dataNeutral.energy_neutral(ix,iy,iz)+=0.5*data.dt*neutral_source.source_energy(ix,iy,iz);                 
            }, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));
        }
        
    };

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //Get the collisional coupling coefficient
        void get_ac(LARE::simulationData &data, LARE::simulationData &dataNeutral, data_two_fluid_source &plasma_source){

            using Range = portableWrapper::Range;
            portableWrapper::applyKernel(LAMBDA(LARE::T_indexType ix, LARE::T_indexType iy, LARE::T_indexType iz) {
                //Get Temperatures
                LARE::T_dataType temperature_electron = data.gas_gamma*data.energy_electron(ix,iy,iz)*(data.gas_gamma-1.0);
                LARE::T_dataType temperature_ion = data.gas_gamma*data.energy_ion(ix,iy,iz)*(data.gas_gamma-1.0);
                LARE::T_dataType temperature_neutral = data.gas_gamma*dataNeutral.energy_neutral(ix,iy,iz)*(data.gas_gamma-1.0);
                
                plasma_source.ac(ix,iy,iz)=dataNeutral.alpha0*std::sqrt(0.5*(temperature_neutral+temperature_ion));
                
        //        printf("ix,iy,iz, t_i t_n ac :  %li %li %li %f %f %f \n",ix,iy,iz,dataNeutral.rho(ix,iy,iz),dataNeutral.energy_neutral(ix,iy,iz),plasma_ir_source.ac(ix,iy,iz));
                //printf("ix,iy,iz, t_i t_n ac :  %li %li %li %f %f %f \n",ix,iy,iz,temperature_ion,temperature_neutral,plasma_ir_source.ac(ix,iy,iz));
                
            	}, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
            //return alpha0*std::sqrt(0.5*(temperature_neutral+temperature_ion));

        }
        
////////////////////////////////////////////////////////////////////////////////////////
        //Formulation from Snow+2021 paper
        //Empirical estimates for the rates
        //Controlled using the data.ion_rec_empirical in control.cpp
        void ion_rec_rates_empirical(LARE::simulationData &data, LARE::simulationData &dataNeutral){

            //Much of this should go elsewhere
            LARE::T_dataType T0=data.T_reference; //Reference temperature
            LARE::T_dataType n0=data.ne_reference; //Reference electron number density
            LARE::T_dataType t_ir=1.0e0; //Reference recombination timescale (relative to collisional timescale)

            LARE::T_dataType Te_0=T0/1.1604e4; //Calculate electron temperature in eV
            LARE::T_dataType rec_fac=2.6e-19*(n0*1.0e6)/std::sqrt(Te_0);  //reference recombination rate (n0 converted to m^-3)

            //initial equilibrium fractions
            LARE::T_dataType ioneq=(2.6e-19/std::sqrt(Te_0))/(2.91e-14/(0.232+13.6/Te_0)*std::pow(13.6/Te_0,0.39)*std::exp(-13.6/Te_0));
            LARE::T_dataType f_n=ioneq/(ioneq+1.0);
            LARE::T_dataType f_p=1.0-f_n;
            LARE::T_dataType f_p_p=2.0*f_p/(f_n+2.0*f_p);
            
            LARE::T_dataType tfac=0.5*f_p_p/f_p; //Normalisation assumes bulk sound speed normalisation
            

            using Range = portableWrapper::Range;
            portableWrapper::applyKernel(LAMBDA(LARE::T_indexType ix, LARE::T_indexType iy, LARE::T_indexType iz) {
                //Get Temperatures
                LARE::T_dataType temperature_electron = 2.0*data.gas_gamma*data.energy_electron(ix,iy,iz)*(data.gas_gamma-1.0);
                LARE::T_dataType numberDensity_electron=data.rho(ix,iy,iz); 

                //Get ionisation and recomination rates
            	data.Gm_rec(ix,iy,iz)=numberDensity_electron/std::sqrt(temperature_electron)*t_ir/f_p*std::sqrt(tfac);
            	data.Gm_ion(ix,iy,iz)=2.91e-14*(n0*1.0e6)*numberDensity_electron*std::exp(-13.6/Te_0/temperature_electron*tfac)*std::pow(13.6/Te_0/temperature_electron*tfac,0.39);
            	data.Gm_ion(ix,iy,iz)=data.Gm_ion(ix,iy,iz)/(0.232+13.6/Te_0/temperature_electron*tfac)/rec_fac/f_p *t_ir;    
            	
            	printf("%f %f %f %f %f \n",f_p,data.energy_electron(ix,iy,iz)*(data.gas_gamma-1.0)*data.rho(ix,iy,iz), temperature_electron,data.Gm_rec(ix,iy,iz),data.Gm_ion(ix,iy,iz));    
            }, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        };
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
//Get the source terms for the IR rates
void get_collisional_source_terms(LARE::simulationData &data, LARE::simulationData &dataNeutral, data_two_fluid_source &plasma_source, data_two_fluid_source &neutral_source){	

    using Range = portableWrapper::Range;
    portableWrapper::applyKernel(LAMBDA(LARE::T_indexType ix, LARE::T_indexType iy, LARE::T_indexType iz) {
            
        //Get Temperatures
        LARE::T_dataType temperature_ion = data.gas_gamma*data.energy_ion(ix,iy,iz)*(data.gas_gamma-1.0);
        LARE::T_dataType temperature_electron = data.gas_gamma*data.energy_electron(ix,iy,iz)*(data.gas_gamma-1.0);
        LARE::T_dataType temperature_neutral = data.gas_gamma*dataNeutral.energy_neutral(ix,iy,iz)*(data.gas_gamma-1.0);
        
        //T_dataType ac;
        //get_ac(data.alpha0,temperature_ion,temperature_neutral);
        
        
        //Get ac and rho at the location of v (vertex)
        LARE::T_dataType ac_vertex=(plasma_source.ac(ix  , iy  , iz  ) + 
                                        plasma_source.ac(ix+1, iy  , iz  ) + 
                                        plasma_source.ac(ix  , iy+1, iz  ) + 
                                        plasma_source.ac(ix+1, iy+1, iz  ) + 
                                        plasma_source.ac(ix  , iy  , iz+1) + 
                                        plasma_source.ac(ix+1, iy  , iz+1) + 
                                        plasma_source.ac(ix  , iy+1, iz+1) + 
                                        plasma_source.ac(ix+1, iy+1, iz+1))* 
                                        0.125;
        LARE::T_dataType rho_plasma_vertex=  (data.rho(ix  , iy  , iz  ) + 
                                        data.rho(ix+1, iy  , iz  ) + 
                                        data.rho(ix  , iy+1, iz  ) + 
                                        data.rho(ix+1, iy+1, iz  ) + 
                                        data.rho(ix  , iy  , iz+1) + 
                                        data.rho(ix+1, iy  , iz+1) + 
                                        data.rho(ix  , iy+1, iz+1) + 
                                        data.rho(ix+1, iy+1, iz+1))* 
                                        0.125;
        LARE::T_dataType rho_neutral_vertex=  (dataNeutral.rho(ix  , iy  , iz  ) + 
                                         dataNeutral.rho(ix+1, iy  , iz  ) + 
                                         dataNeutral.rho(ix  , iy+1, iz  ) + 
                                         dataNeutral.rho(ix+1, iy+1, iz  ) + 
                                         dataNeutral.rho(ix  , iy  , iz+1) + 
                                         dataNeutral.rho(ix+1, iy  , iz+1) + 
                                         dataNeutral.rho(ix  , iy+1, iz+1) + 
                                         dataNeutral.rho(ix+1, iy+1, iz+1))* 
                                         0.125;
        
                
        //Apply the velocity exchange terms
        plasma_source.source_v_x(ix,iy,iz)+=ac_vertex*dataNeutral.rho(ix,iy,iz)*(dataNeutral.vx(ix,iy,iz)-data.vx(ix,iy,iz));
        neutral_source.source_v_x(ix,iy,iz)-=ac_vertex*data.rho(ix,iy,iz)*(dataNeutral.vx(ix,iy,iz)-data.vx(ix,iy,iz));
        
        plasma_source.source_v_y(ix,iy,iz)+=ac_vertex*dataNeutral.rho(ix,iy,iz)*(dataNeutral.vy(ix,iy,iz)-data.vy(ix,iy,iz));
        neutral_source.source_v_y(ix,iy,iz)-=ac_vertex*data.rho(ix,iy,iz)*(dataNeutral.vy(ix,iy,iz)-data.vy(ix,iy,iz));
        
        plasma_source.source_v_z(ix,iy,iz)+=ac_vertex*dataNeutral.rho(ix,iy,iz)*(dataNeutral.vz(ix,iy,iz)-data.vz(ix,iy,iz));
        neutral_source.source_v_z(ix,iy,iz)-=ac_vertex*data.rho(ix,iy,iz)*(dataNeutral.vz(ix,iy,iz)-data.vz(ix,iy,iz));
        
        //Get velocity at cell centre
        LARE::T_dataType vx_centre=(data.vx(ix,iy,iz)+
                              data.vx(ix  ,iy-1,iz)+
                              data.vx(ix  ,iy  ,iz-1)+
                              data.vx(ix  ,iy-1,iz-1)+
                              data.vx(ix-1,iy  ,iz  )+
                              data.vx(ix-1,iy-1,iz  )+
                              data.vx(ix-1,iy  ,iz-1)+
                              data.vx(ix-1,iy-1,iz-1))*
                              0.125;
        LARE::T_dataType vy_centre=(data.vy(ix,iy,iz)+
                              data.vy(ix  ,iy-1,iz)+
                              data.vy(ix  ,iy  ,iz-1)+
                              data.vy(ix  ,iy-1,iz-1)+
                              data.vy(ix-1,iy  ,iz  )+
                              data.vy(ix-1,iy-1,iz  )+
                              data.vy(ix-1,iy  ,iz-1)+
                              data.vy(ix-1,iy-1,iz-1))*
                              0.125;
        LARE::T_dataType vz_centre=(data.vz(ix,iy,iz)+
                              data.vz(ix  ,iy-1,iz)+
                              data.vz(ix  ,iy  ,iz-1)+
                              data.vz(ix  ,iy-1,iz-1)+
                              data.vz(ix-1,iy  ,iz  )+
                              data.vz(ix-1,iy-1,iz  )+
                              data.vz(ix-1,iy  ,iz-1)+
                              data.vz(ix-1,iy-1,iz-1))*
                              0.125;
        LARE::T_dataType vx_n_centre=(dataNeutral.vx(ix,iy,iz)+
                              dataNeutral.vx(ix  ,iy-1,iz)+
                              dataNeutral.vx(ix  ,iy  ,iz-1)+
                              dataNeutral.vx(ix  ,iy-1,iz-1)+
                              dataNeutral.vx(ix-1,iy  ,iz  )+
                              dataNeutral.vx(ix-1,iy-1,iz  )+
                              dataNeutral.vx(ix-1,iy  ,iz-1)+
                              dataNeutral.vx(ix-1,iy-1,iz-1))*
                              0.125;
        LARE::T_dataType vy_n_centre=(dataNeutral.vy(ix,iy,iz)+
                              dataNeutral.vy(ix  ,iy-1,iz)+
                              dataNeutral.vy(ix  ,iy  ,iz-1)+
                              dataNeutral.vy(ix  ,iy-1,iz-1)+
                              dataNeutral.vy(ix-1,iy  ,iz  )+
                              dataNeutral.vy(ix-1,iy-1,iz  )+
                              dataNeutral.vy(ix-1,iy  ,iz-1)+
                              dataNeutral.vy(ix-1,iy-1,iz-1))*
                              0.125;
        LARE::T_dataType vz_n_centre=(dataNeutral.vz(ix,iy,iz)+
                              dataNeutral.vz(ix  ,iy-1,iz)+
                              dataNeutral.vz(ix  ,iy  ,iz-1)+
                              dataNeutral.vz(ix  ,iy-1,iz-1)+
                              dataNeutral.vz(ix-1,iy  ,iz  )+
                              dataNeutral.vz(ix-1,iy-1,iz  )+
                              dataNeutral.vz(ix-1,iy  ,iz-1)+
                              dataNeutral.vz(ix-1,iy-1,iz-1))*
                              0.125;
        
        //Energy source terms - the 3/2 here needs fixing
        /*plasma_ir_source.source_energy(ix,iy,iz)=plasma_ir_source.ac(ix,iy,iz)*dataNeutral.rho(ix,iy,iz)*(0.5*(\
                        (dataNeutral.vx(ix,iy,iz)*dataNeutral.vx(ix,iy,iz)-data.vx(ix,iy,iz)*data.vx(ix,iy,iz))+\
                        (dataNeutral.vy(ix,iy,iz)*dataNeutral.vy(ix,iy,iz)-data.vy(ix,iy,iz)*data.vy(ix,iy,iz))+\
                        (dataNeutral.vz(ix,iy,iz)*dataNeutral.vz(ix,iy,iz)-data.vz(ix,iy,iz)*data.vz(ix,iy,iz)))\
                        + 3.0/data.gas_gamma/2.0*(temperature_neutral-temperature_ion));
        neutral_ir_source.source_energy(ix,iy,iz)=-plasma_ir_source.ac(ix,iy,iz)*dataNeutral.rho(ix,iy,iz)*(0.5*(\
                        (dataNeutral.vx(ix,iy,iz)*dataNeutral.vx(ix,iy,iz)-data.vx(ix,iy,iz)*data.vx(ix,iy,iz))+\
                        (dataNeutral.vy(ix,iy,iz)*dataNeutral.vy(ix,iy,iz)-data.vy(ix,iy,iz)*data.vy(ix,iy,iz))+\
                        (dataNeutral.vz(ix,iy,iz)*dataNeutral.vz(ix,iy,iz)-data.vz(ix,iy,iz)*data.vz(ix,iy,iz)))\
                        + 3.0/data.gas_gamma/2.0*(temperature_neutral-temperature_ion));  
        */
        plasma_source.source_energy(ix,iy,iz)=plasma_source.ac(ix,iy,iz)*dataNeutral.rho(ix,iy,iz)*(0.5*(\
                        (vx_n_centre*vx_n_centre-vx_centre*vx_centre)+\
                        (vy_n_centre*vy_n_centre-vy_centre*vy_centre)+\
                        (vz_n_centre*vz_n_centre-vz_centre*vz_centre))\
                        + 3.0/data.gas_gamma/2.0*(temperature_neutral-temperature_ion));
        neutral_source.source_energy(ix,iy,iz)=-plasma_source.ac(ix,iy,iz)*data.rho(ix,iy,iz)*(0.5*(\
                        (vx_n_centre*vx_n_centre-vx_centre*vx_centre)+\
                        (vy_n_centre*vy_n_centre-vy_centre*vy_centre)+\
                        (vz_n_centre*vz_n_centre-vz_centre*vz_centre))\
                        + 3.0/data.gas_gamma/2.0*(temperature_neutral-temperature_ion));  
                                   
        
    }, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
//Get the source terms for the IR rates
void get_ion_rec_source_terms(LARE::simulationData &data, LARE::simulationData &dataNeutral, data_two_fluid_source &plasma_source, data_two_fluid_source &neutral_source){	

    using Range = portableWrapper::Range;
    portableWrapper::applyKernel(LAMBDA(LARE::T_indexType ix, LARE::T_indexType iy, LARE::T_indexType iz) {
    
        //Mass source terms
        plasma_source.source_mass(ix,iy,iz)  += data.Gm_ion(ix,iy,iz)*dataNeutral.rho(ix,iy,iz)-data.Gm_rec(ix,iy,iz)*data.rho(ix,iy,iz);
        neutral_source.source_mass(ix,iy,iz) +=-data.Gm_ion(ix,iy,iz)*dataNeutral.rho(ix,iy,iz)+data.Gm_rec(ix,iy,iz)*data.rho(ix,iy,iz);
        
        LARE::T_dataType rho_plasma_vertex=  (data.rho(ix  , iy  , iz  ) + 
                                        data.rho(ix+1, iy  , iz  ) + 
                                        data.rho(ix  , iy+1, iz  ) + 
                                        data.rho(ix+1, iy+1, iz  ) + 
                                        data.rho(ix  , iy  , iz+1) + 
                                        data.rho(ix+1, iy  , iz+1) + 
                                        data.rho(ix  , iy+1, iz+1) + 
                                        data.rho(ix+1, iy+1, iz+1))* 
                                        0.125;
        LARE::T_dataType Gm_ion_vertex=      (data.Gm_ion(ix  , iy  , iz  ) + 
                                        data.Gm_ion(ix+1, iy  , iz  ) + 
                                        data.Gm_ion(ix  , iy+1, iz  ) + 
                                        data.Gm_ion(ix+1, iy+1, iz  ) + 
                                        data.Gm_ion(ix  , iy  , iz+1) + 
                                        data.Gm_ion(ix+1, iy  , iz+1) + 
                                        data.Gm_ion(ix  , iy+1, iz+1) + 
                                        data.Gm_ion(ix+1, iy+1, iz+1))* 
                                        0.125;
        LARE::T_dataType Gm_rec_vertex=      (data.Gm_rec(ix  , iy  , iz  ) + 
                                        data.Gm_rec(ix+1, iy  , iz  ) + 
                                        data.Gm_rec(ix  , iy+1, iz  ) + 
                                        data.Gm_rec(ix+1, iy+1, iz  ) + 
                                        data.Gm_rec(ix  , iy  , iz+1) + 
                                        data.Gm_rec(ix+1, iy  , iz+1) + 
                                        data.Gm_rec(ix  , iy+1, iz+1) + 
                                        data.Gm_rec(ix+1, iy+1, iz+1))* 
                                        0.125;
        LARE::T_dataType rho_neutral_vertex=  (dataNeutral.rho(ix  , iy  , iz  ) + 
                                         dataNeutral.rho(ix+1, iy  , iz  ) + 
                                         dataNeutral.rho(ix  , iy+1, iz  ) + 
                                         dataNeutral.rho(ix+1, iy+1, iz  ) + 
                                         dataNeutral.rho(ix  , iy  , iz+1) + 
                                         dataNeutral.rho(ix+1, iy  , iz+1) + 
                                         dataNeutral.rho(ix  , iy+1, iz+1) + 
                                         dataNeutral.rho(ix+1, iy+1, iz+1))* 
                                         0.125;
        
        //Velocity source terms
        LARE::T_dataType v_D_x  =  data.vx(ix,iy,iz) - dataNeutral.vx(ix,iy,iz); //Drift velocity in the x-direction
        LARE::T_dataType v_D_y  =  data.vy(ix,iy,iz) - dataNeutral.vy(ix,iy,iz); //Drift velocity in the y-direction
        LARE::T_dataType v_D_z  =  data.vz(ix,iy,iz) - dataNeutral.vz(ix,iy,iz); //Drift velocity in the z-direction
        plasma_source.source_v_x(ix,iy,iz) += -Gm_ion_vertex*rho_neutral_vertex*v_D_x/rho_plasma_vertex;
        plasma_source.source_v_y(ix,iy,iz) += -Gm_ion_vertex*rho_neutral_vertex*v_D_y/rho_plasma_vertex;
        plasma_source.source_v_z(ix,iy,iz) += -Gm_ion_vertex*rho_neutral_vertex*v_D_z/rho_plasma_vertex;
        neutral_source.source_v_x(ix,iy,iz) += Gm_rec_vertex*rho_plasma_vertex*v_D_x/rho_neutral_vertex;
        neutral_source.source_v_y(ix,iy,iz) += Gm_rec_vertex*rho_plasma_vertex*v_D_y/rho_neutral_vertex;
        neutral_source.source_v_z(ix,iy,iz) += Gm_rec_vertex*rho_plasma_vertex*v_D_z/rho_neutral_vertex;
        
        
        //Get velocity at cell centres
        LARE::T_dataType v_x_plasma_centre=  (data.vx(ix  , iy  , iz  ) + 
                                        data.vx(ix-1, iy  , iz  ) + 
                                        data.vx(ix  , iy-1, iz  ) + 
                                        data.vx(ix-1, iy-1, iz  ) + 
                                        data.vx(ix  , iy  , iz-1) + 
                                        data.vx(ix-1, iy  , iz-1) + 
                                        data.vx(ix  , iy-1, iz-1) + 
                                        data.vx(ix-1, iy-1, iz-1))* 
                                        0.125;
        LARE::T_dataType v_y_plasma_centre=  (data.vy(ix  , iy  , iz  ) + 
                                        data.vy(ix-1, iy  , iz  ) + 
                                        data.vy(ix  , iy-1, iz  ) + 
                                        data.vy(ix-1, iy-1, iz  ) + 
                                        data.vy(ix  , iy  , iz-1) + 
                                        data.vy(ix-1, iy  , iz-1) + 
                                        data.vy(ix  , iy-1, iz-1) + 
                                        data.vy(ix-1, iy-1, iz-1))* 
                                        0.125;
        LARE::T_dataType v_z_plasma_centre=  (data.vz(ix  , iy  , iz  ) + 
                                        data.vz(ix-1, iy  , iz  ) + 
                                        data.vz(ix  , iy-1, iz  ) + 
                                        data.vz(ix-1, iy-1, iz  ) + 
                                        data.vz(ix  , iy  , iz-1) + 
                                        data.vz(ix-1, iy  , iz-1) + 
                                        data.vz(ix  , iy-1, iz-1) + 
                                        data.vz(ix-1, iy-1, iz-1))* 
                                        0.125;
        LARE::T_dataType v_x_neutral_centre= (dataNeutral.vx(ix  , iy  , iz  ) + 
                                        dataNeutral.vx(ix-1, iy  , iz  ) + 
                                        dataNeutral.vx(ix  , iy-1, iz  ) + 
                                        dataNeutral.vx(ix-1, iy-1, iz  ) + 
                                        dataNeutral.vx(ix  , iy  , iz-1) + 
                                        dataNeutral.vx(ix-1, iy  , iz-1) + 
                                        dataNeutral.vx(ix  , iy-1, iz-1) + 
                                        dataNeutral.vx(ix-1, iy-1, iz-1))* 
                                        0.125;
        LARE::T_dataType v_y_neutral_centre= (dataNeutral.vy(ix  , iy  , iz  ) + 
                                        dataNeutral.vy(ix-1, iy  , iz  ) + 
                                        dataNeutral.vy(ix  , iy-1, iz  ) + 
                                        dataNeutral.vy(ix-1, iy-1, iz  ) + 
                                        dataNeutral.vy(ix  , iy  , iz-1) + 
                                        dataNeutral.vy(ix-1, iy  , iz-1) + 
                                        dataNeutral.vy(ix  , iy-1, iz-1) + 
                                        dataNeutral.vy(ix-1, iy-1, iz-1))* 
                                        0.125;
        LARE::T_dataType v_z_neutral_centre= (dataNeutral.vz(ix  , iy  , iz  ) + 
                                        dataNeutral.vz(ix-1, iy  , iz  ) + 
                                        dataNeutral.vz(ix  , iy-1, iz  ) + 
                                        dataNeutral.vz(ix-1, iy-1, iz  ) + 
                                        dataNeutral.vz(ix  , iy  , iz-1) + 
                                        dataNeutral.vz(ix-1, iy  , iz-1) + 
                                        dataNeutral.vz(ix  , iy-1, iz-1) + 
                                        dataNeutral.vz(ix-1, iy-1, iz-1))* 
                                        0.125;
        
        //Energy source terms
        plasma_source.source_energy(ix,iy,iz) += -0.5*(data.Gm_rec(ix,iy,iz)*(pow(v_x_plasma_centre,2)+
                                                                                 pow(v_y_plasma_centre,2)+
                                                                                 pow(v_z_plasma_centre,2))
                                                        -data.Gm_ion(ix,iy,iz)*(pow(v_x_neutral_centre,2)+
                                                                                pow(v_y_neutral_centre,2)+
                                                                                pow(v_z_neutral_centre,2))
                                                                              *dataNeutral.rho(ix,iy,iz)/data.rho(ix,iy,iz)                                                                                
                                                        )
                                                   -(data.Gm_rec(ix,iy,iz)*data.energy_ion(ix,iy,iz)-data.Gm_ion(ix,iy,iz)*dataNeutral.energy_neutral(ix,iy,iz)*dataNeutral.rho(ix,iy,iz)/data.rho(ix,iy,iz))/(data.gas_gamma-1.0); //Is this electron or ion energy (or mean energy)? is the half needed?
        
        neutral_source.source_energy(ix,iy,iz) += 0.5*(data.Gm_rec(ix,iy,iz)*(data.vx(ix,iy,iz)*data.vx(ix,iy,iz)+
                                                                                data.vy(ix,iy,iz)*data.vy(ix,iy,iz)+
                                                                                data.vz(ix,iy,iz)*data.vz(ix,iy,iz))
                                                                                *data.rho(ix,iy,iz)/dataNeutral.rho(ix,iy,iz)
                                                        -data.Gm_ion(ix,iy,iz)*(dataNeutral.vx(ix,iy,iz)*data.vx(ix,iy,iz)+
                                                                                dataNeutral.vy(ix,iy,iz)*data.vy(ix,iy,iz)+
                                                                                dataNeutral.vz(ix,iy,iz)*data.vz(ix,iy,iz))
                                                        )
                                                   +(data.Gm_rec(ix,iy,iz)*data.energy_ion(ix,iy,iz)*data.rho(ix,iy,iz)/dataNeutral.rho(ix,iy,iz)-data.Gm_ion(ix,iy,iz)*dataNeutral.energy_neutral(ix,iy,iz))/(data.gas_gamma-1.0); //Is this electron or ion energy (or mean energy)? is the half needed?
        
        //Work out how much energy is spent/gained by IR processes
        //if(data.ion_rec_empirical){ 
        //    LARE::T_dataType ionisation_energy=(data.Gm_rec(ix,iy,iz)-
        //                          data.Gm_ion(ix,iy,iz)*dataNeutral.rho(ix,iy,iz)/(data.gas_gamma-1.0)/data.rho(ix,iy,iz))*
        //                          13.6/kb_si/data.T_reference;     
        //    //plasma_ir_source.source_electron_energy(ix,iy,iz)+=ionisation_energy; 
        //}
        
    }, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //};
}

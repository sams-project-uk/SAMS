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
        //    ion_rec_rates_empirical(data,dataNeutral);
        }
        if (two_fluid_flags.ion_rec_nlevel){        
        //    ion_rec_rates_nlevel(data,dataNeutral);
        }
        
        //Calculate the source terms for the two-fluid interactions
        get_collisional_source_terms(data,dataNeutral,plasma_source,neutral_source);
        
        //Calculate the source terms for Ionisation/recombination
        //if (two_fluid_flags.ion_rec) get_ion_rec_source_terms(data,dataNeutral,plasma_ir_source,neutral_ir_source);
        
        
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
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //};
}

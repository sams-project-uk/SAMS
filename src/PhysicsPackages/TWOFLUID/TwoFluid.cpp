/*
* This is a two-fluid routine for all the multi-fluid physics modules
*/

//////////////////////No idea which ones of these are needed
#include <iostream>
#include <cstdint>
#include <cassert>
#include <string>
#include "constants.h"
#include "constants_neutral.h"
#include "pp/parallelWrapper.h"
#include "remapData.h"
#include "typedefs.h"
#include "remapData_neutral.h"
#include "typedefs_neutral.h"
#include "mpiManager.h"
#include "variableDef.h"
#include "harness.h"
#include "runner.h"
#include "io/writerProto.h"

#include "twofluid.h"

#include "shared_data.h"
#include "shared_data_neutral.h"
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
    
    //void get_ac(LARE::simulationData &data, LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source);
    //void two_fluid_source(LARE::simulationData &data,LARE_neutral::simulationData &dataNeutral);
    void ion_rec_rates_empirical(LARE::simulationData &data, LARE_neutral::simulationData &dataNeutral);
    void get_collisional_source_terms(LARE::simulationData &data, LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source);
    void get_ion_rec_source_terms(LARE::simulationData &data, LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source);
            
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
    * Default values
    */
    void PIP::defaultValues(LARE::simulationData &data,LARE_neutral::simulationData &dataNeutral){
        SAMS::cout << "Setting default values" << std::endl;
        dataNeutral.alpha0=1000.0;
        dataNeutral.is_neutral=true;
        SAMS::cout << "dataNeutral=" << dataNeutral.is_neutral << std::endl;
        SAMS::cout << "data=" << data.is_neutral << std::endl;
        //printf("Here \n");
    }


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Register variables with the portable array manager.
     */
    void PIP::registerVariables(SAMS::harness &harness)
    {

        auto &varRegistry = harness.variableRegistry;

        const int ghosts = 2; // 2 Ghost cells at top and bottom of each dimension

        varRegistry.registerVariable<LARE::T_dataType>("ac", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));

        varRegistry.registerVariable<LARE::T_dataType>("source_mass", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        
        varRegistry.registerVariable<LARE::T_dataType>("source_mass_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        
        varRegistry.registerVariable<LARE::T_dataType>("source_energy", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        
        varRegistry.registerVariable<LARE::T_dataType>("source_energy_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));

        varRegistry.registerVariable<LARE::T_dataType>("source_vx", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
        
        varRegistry.registerVariable<LARE::T_dataType>("source_vx_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));

        varRegistry.registerVariable<LARE::T_dataType>("source_vy", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
        
        varRegistry.registerVariable<LARE::T_dataType>("source_vy_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));

        varRegistry.registerVariable<LARE::T_dataType>("source_vz", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
        
        varRegistry.registerVariable<LARE::T_dataType>("source_vz_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
    }
/////////////////////////////////////////////////////////////////////////////////
    void PIP::allocate(LARE::simulationData &data,LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source,SAMS::harness &harness){
        //data_two_fluid_source plasma_source;
        //data_two_fluid_source neutral_source;
       
        auto &axRegistry = harness.axisRegistry;
        auto &varRegistry = harness.variableRegistry;
        
        printf("%li \n", data.nx);
        printf("%li \n", data.ny);
        printf("%li \n", data.nz);

        using Range = pw::Range;
        varRegistry.fillPPArray("ac", plasma_source.ac);
        pw::assign(plasma_source.ac, 0.0);
        varRegistry.fillPPArray("source_mass", plasma_source.source_mass);
        pw::assign(plasma_source.source_mass, 0.0);
        varRegistry.fillPPArray("source_mass_n", plasma_source.source_mass_n);
        pw::assign(plasma_source.source_mass_n, 0.0);
        varRegistry.fillPPArray("source_energy", plasma_source.source_energy);
        pw::assign(plasma_source.source_energy, 0.0);
        varRegistry.fillPPArray("source_energy_n", plasma_source.source_energy_n);
        pw::assign(plasma_source.source_energy_n, 0.0);
        varRegistry.fillPPArray("source_vx", plasma_source.source_v_x);
        pw::assign(plasma_source.source_v_x, 0.0);
        varRegistry.fillPPArray("source_vx_n", plasma_source.source_v_x_n);
        pw::assign(plasma_source.source_v_x_n, 0.0);
        varRegistry.fillPPArray("source_vy", plasma_source.source_v_y);
        pw::assign(plasma_source.source_v_y, 0.0);
        varRegistry.fillPPArray("source_vy_n", plasma_source.source_v_y_n);
        pw::assign(plasma_source.source_v_y_n, 0.0);
        varRegistry.fillPPArray("source_vz", plasma_source.source_v_z);
        pw::assign(plasma_source.source_v_z, 0.0);
        varRegistry.fillPPArray("source_vz_n", plasma_source.source_v_z_n);
        pw::assign(plasma_source.source_v_z_n, 0.0);
        
        /*SourceManager.allocate(plasma_source.ac, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(plasma_source.source_mass, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(plasma_source.source_v_x, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(plasma_source.source_v_y, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(plasma_source.source_v_z, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(plasma_source.source_energy, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(plasma_source.source_electron_energy, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(plasma_source.source_mass_n, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(plasma_source.source_v_x_n, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(plasma_source.source_v_y_n, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(plasma_source.source_v_z_n, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        SourceManager.allocate(plasma_source.source_energy_n, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
        
        portableWrapper::assign(plasma_source.source_mass,0.0);
        portableWrapper::assign(plasma_source.source_v_x,0.0);
        portableWrapper::assign(plasma_source.source_v_y,0.0);
        portableWrapper::assign(plasma_source.source_v_z,0.0);
        portableWrapper::assign(plasma_source.source_energy,0.0);
        portableWrapper::assign(plasma_source.source_electron_energy,0.0);
        portableWrapper::assign(plasma_source.source_mass_n,0.0);
        portableWrapper::assign(plasma_source.source_v_x_n,0.0);
        portableWrapper::assign(plasma_source.source_v_y_n,0.0);
        portableWrapper::assign(plasma_source.source_v_z_n,0.0);
        portableWrapper::assign(plasma_source.source_energy_n,0.0);
        portableWrapper::assign(plasma_source.ac,0.0);
        */
    }
////////////////////////////////////////////////////////////////////////////////////////
    void PIP::two_fluid_source(LARE::simulationData &data,LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source){

        two_fluid_properties two_fluid_flags; //Move to source structure
        
        //Get collisional coefficient
        //PIP::get_ac(data,dataNeutral,plasma_source); //Maybe already calcuated?
        
        
        //Get the ionisation rates
        //if (two_fluid_flags.ion_rec_empirical){        
        //    ion_rec_rates_empirical(data,dataNeutral);
        //}
        //if (two_fluid_flags.ion_rec_nlevel){        
        ////    ion_rec_rates_nlevel(data,dataNeutral);
        //}
        
        //Calculate the source terms for the two-fluid interactions
        //printf("getting source terms \n");
        get_collisional_source_terms(data,dataNeutral,plasma_source);
        
        //Calculate the source terms for Ionisation/recombination
        //if (two_fluid_flags.ion_rec_empirical) get_ion_rec_source_terms(data,dataNeutral,plasma_source);
        
        
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
        //printf("entering loop \n");
        if (two_fluid_flags.collisions){
            using Range = portableWrapper::Range;
            portableWrapper::applyKernel(LAMBDA(LARE::T_indexType ix, LARE::T_indexType iy, LARE::T_indexType iz) {
             //printf("%i,%i,%i \n",ix,iy,iz);
             //printf("rho %f \n",data.rho(ix,iy,iz));
             //printf("rho %f \n",dataNeutral.rho(ix,iy,iz));
             //printf("dt %f \n",data.dt);
             //printf("mass_source %f \n",plasma_source.source_mass(ix,iy,iz));
             //printf("mass_source_n %f \n",plasma_source.source_mass_n(ix,iy,iz));
             //printf("source_vx %f \n",plasma_source.source_v_x(ix,iy,iz));
             //printf("source_vx_n %f \n",plasma_source.source_v_x_n(ix,iy,iz));
             //printf("source_vy %f \n",plasma_source.source_v_y(ix,iy,iz));
             //printf("source_vy_n %f \n",plasma_source.source_v_y_n(ix,iy,iz));
             //printf("source_vz %f \n",plasma_source.source_v_z(ix,iy,iz));
             //printf("source_vz_n %f \n",plasma_source.source_v_z_n(ix,iy,iz));
                //Get Temperatures
                //T_dataType temperature_ion = data.gas_gamma*data.energy_ion(ix,iy,iz)*(data.gas_gamma-1.0);
                //T_dataType temperature_electron = data.gas_gamma*data.energy_electron(ix,iy,iz)*(data.gas_gamma-1.0);
                //T_dataType temperature_neutral = data.gas_gamma*dataNeutral.energy_neutral(ix,iy,iz)*(data.gas_gamma-1.0);
                
                //T_dataType ac;
                //get_ac(data.alpha0,temperature_ion,temperature_neutral);
                
                //Note that the factor of 0.5 in these is due to Strang splitting
                //Mass exchange terms
                data.rho(ix,iy,iz)+=0.5*data.dt*plasma_source.source_mass(ix,iy,iz);
                dataNeutral.rho(ix,iy,iz)+=0.5*data.dt*plasma_source.source_mass_n(ix,iy,iz);
                
                //Apply the velocity exchange terms
                data.vx(ix,iy,iz)       +=0.5*data.dt*plasma_source.source_v_x(ix,iy,iz);
                dataNeutral.vx(ix,iy,iz)+=0.5*data.dt*plasma_source.source_v_x_n(ix,iy,iz);
                
                data.vy(ix,iy,iz)       +=0.5*data.dt*plasma_source.source_v_y(ix,iy,iz);
                dataNeutral.vy(ix,iy,iz)+=0.5*data.dt*plasma_source.source_v_y_n(ix,iy,iz);
                
                data.vz(ix,iy,iz)       +=0.5*data.dt*plasma_source.source_v_z(ix,iy,iz);
                dataNeutral.vz(ix,iy,iz)+=0.5*data.dt*plasma_source.source_v_z_n(ix,iy,iz);
                
                //Energy source terms - the 3/2 here needs fixing
                data.energy_ion(ix,iy,iz)+=0.5*data.dt*plasma_source.source_energy(ix,iy,iz);
                //data.energy_electron(ix,iy,iz)+=0.5*data.dt*plasma_source.source_electron_energy(ix,iy,iz);
                dataNeutral.energy_neutral(ix,iy,iz)+=0.5*data.dt*plasma_source.source_energy_n(ix,iy,iz);                 
            }, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));
        }
        
    };

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //Get the collisional coupling coefficient
        void PIP::get_ac(LARE::simulationData &data, LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source){

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
        void ion_rec_rates_empirical(LARE::simulationData &data, LARE_neutral::simulationData &dataNeutral){

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
void get_collisional_source_terms(LARE::simulationData &data, LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source){	

    using Range = portableWrapper::Range;
    portableWrapper::applyKernel(LAMBDA(LARE::T_indexType ix, LARE::T_indexType iy, LARE::T_indexType iz) {
        //printf("getting source terms \n");    
        //Get Temperatures
        LARE::T_dataType temperature_ion = data.gas_gamma*data.energy_ion(ix,iy,iz)*(data.gas_gamma-1.0);
        LARE::T_dataType temperature_electron = data.gas_gamma*data.energy_electron(ix,iy,iz)*(data.gas_gamma-1.0);
        LARE::T_dataType temperature_neutral = data.gas_gamma*dataNeutral.energy_neutral(ix,iy,iz)*(data.gas_gamma-1.0);
        
        //T_dataType ac;
        //get_ac(data.alpha0,temperature_ion,temperature_neutral);
        
        
        //printf("getting vertex values \n");
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
        plasma_source.source_v_x_n(ix,iy,iz)-=ac_vertex*data.rho(ix,iy,iz)*(dataNeutral.vx(ix,iy,iz)-data.vx(ix,iy,iz));
        
        plasma_source.source_v_y(ix,iy,iz)+=ac_vertex*dataNeutral.rho(ix,iy,iz)*(dataNeutral.vy(ix,iy,iz)-data.vy(ix,iy,iz));
        plasma_source.source_v_y_n(ix,iy,iz)-=ac_vertex*data.rho(ix,iy,iz)*(dataNeutral.vy(ix,iy,iz)-data.vy(ix,iy,iz));
        
        plasma_source.source_v_z(ix,iy,iz)+=ac_vertex*dataNeutral.rho(ix,iy,iz)*(dataNeutral.vz(ix,iy,iz)-data.vz(ix,iy,iz));
        plasma_source.source_v_z_n(ix,iy,iz)-=ac_vertex*data.rho(ix,iy,iz)*(dataNeutral.vz(ix,iy,iz)-data.vz(ix,iy,iz));
        
        //printf("getting vx at centre \n");
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
       //printf("getting energy source terms \n");
       //printf("ac %f \n", plasma_source.ac(ix,iy,iz));
       //printf("rho_n %f \n", dataNeutral.rho(ix,iy,iz));
       //printf("vx_n %f \n", vx_n_centre);
       //printf("vx %f \n", vx_centre);
       //printf("vy_n %f \n", vy_n_centre);
       //printf("vy %f \n", vy_centre);
       //printf("vz_n %f \n", vz_n_centre);
       //printf("vz %f \n", vz_centre);
       //printf("T %f \n", temperature_ion);
       //printf("T_n %f \n", temperature_neutral);
       //printf("gm %f \n", data.gas_gamma);
       //printf("gm %f \n", plasma_source.source_energy(ix,iy,iz));
    plasma_source.source_energy(ix,iy,iz)=plasma_source.ac(ix,iy,iz)*dataNeutral.rho(ix,iy,iz)*(0.5*(\
                        (vx_n_centre*vx_n_centre-vx_centre*vx_centre)+\
                        (vy_n_centre*vy_n_centre-vy_centre*vy_centre)+\
                        (vz_n_centre*vz_n_centre-vz_centre*vz_centre))\
                        + 3.0/data.gas_gamma/2.0*(temperature_neutral-temperature_ion));
        //printf("getting neutral energy source terms \n");
        plasma_source.source_energy_n(ix,iy,iz)=-plasma_source.ac(ix,iy,iz)*data.rho(ix,iy,iz)*(0.5*(\
                        (vx_n_centre*vx_n_centre-vx_centre*vx_centre)+\
                        (vy_n_centre*vy_n_centre-vy_centre*vy_centre)+\
                        (vz_n_centre*vz_n_centre-vz_centre*vz_centre))\
                        + 3.0/data.gas_gamma/2.0*(temperature_neutral-temperature_ion));  
                                   
        //printf("%i,%i,%i \n",ix,iy,iz);
    }, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));

//printf("FINISHED LOOP \n");
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
//Get the source terms for the IR rates
void get_ion_rec_source_terms(LARE::simulationData &data, LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source){	

    using Range = portableWrapper::Range;
    portableWrapper::applyKernel(LAMBDA(LARE::T_indexType ix, LARE::T_indexType iy, LARE::T_indexType iz) {
    
        //Mass source terms
        plasma_source.source_mass(ix,iy,iz)  += data.Gm_ion(ix,iy,iz)*dataNeutral.rho(ix,iy,iz)-data.Gm_rec(ix,iy,iz)*data.rho(ix,iy,iz);
        plasma_source.source_mass_n(ix,iy,iz) +=-data.Gm_ion(ix,iy,iz)*dataNeutral.rho(ix,iy,iz)+data.Gm_rec(ix,iy,iz)*data.rho(ix,iy,iz);
        
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
        plasma_source.source_v_x_n(ix,iy,iz) += Gm_rec_vertex*rho_plasma_vertex*v_D_x/rho_neutral_vertex;
        plasma_source.source_v_y_n(ix,iy,iz) += Gm_rec_vertex*rho_plasma_vertex*v_D_y/rho_neutral_vertex;
        plasma_source.source_v_z_n(ix,iy,iz) += Gm_rec_vertex*rho_plasma_vertex*v_D_z/rho_neutral_vertex;
        
        
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
        
        plasma_source.source_energy_n(ix,iy,iz) += 0.5*(data.Gm_rec(ix,iy,iz)*(data.vx(ix,iy,iz)*data.vx(ix,iy,iz)+
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
////////////////////////////////////////////////////////////////////////////////////////
//Collisional timestep calculation
//Assuming normalisation to the sound speed
void PIP::set_dt_collisional(LARE::simulationData &data,LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source) {

    using Range = portableWrapper::Range;

    int i0 = data.geometry == LARE::geometryType::Cartesian ? 0:1;

    //Now need to do a map and reduction
    plasma_source.two_fluid_timestep = data.dt_multiplier * 
    portableWrapper::applyReduction(LAMBDA(LARE::T_indexType ix, LARE::T_indexType iy, LARE::T_indexType iz) {
        //Get Temperatures
        //T_dataType temperature_ion = data.gas_gamma*data.energy_ion(ix,iy,iz)*(data.gas_gamma-1.0);
        //T_dataType temperature_electron = data.gas_gamma*data.energy_electron(ix,iy,iz)*(data.gas_gamma-1.0);
        //T_dataType temperature_neutral = data.gas_gamma*dataNeutral.energy_neutral(ix,iy,iz)*(data.gas_gamma-1.0);

        //This needs temeprature dependence
        //T_dataType ac=data.alpha0*std::sqrt(0.5*(temperature_neutral+temperature_ion));
        
        LARE::T_dataType collisional_timestep_plasma=0.3/(plasma_source.ac(ix,iy,iz)*data.rho(ix,iy,iz));
        LARE::T_dataType collisional_timestep_neutral=0.3/(plasma_source.ac(ix,iy,iz)*dataNeutral.rho(ix,iy,iz));
        
        LARE::T_dataType t1 = std::min(collisional_timestep_plasma,collisional_timestep_neutral);
        
        //printf("ix,iy,iz, t1 :  %li %li %li %f %f \n",ix,iy,iz,collisional_timestep_plasma,plasma_ir_source.ac(ix,iy,iz));
        //if (collisional_timestep_plasma < collisional_timestep_neutral){
        //    t1=collisional_timestep_plasma;
        //} else{
        //    t1=collisional_timestep_neutral;
        //}
                return t1;
    }, LAMBDA(LARE::T_dataType &a, const LARE::T_dataType &b) {
        a=portableWrapper::min(a, b);
    }, data.largest_number,
    Range(i0, data.nx), Range(0, data.ny), Range(0, data.nz));

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //};
}

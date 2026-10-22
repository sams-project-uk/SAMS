/*
* This is a two-fluid routine for all the multi-fluid physics modules
*/

//////////////////////No idea which ones of these are needed
#include <iostream>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cassert>
#include <string>
#include "pp/parallelWrapper.h"
#include "mpiManager.h"
#include "variableDef.h"
#include "harness.h"
#include "runner.h"
#include "io/writerProto.h"

#include "twofluid.h"

#include "variableRegistry.h"
#include "axisRegistry.h"

#include <netcdf.h>

// Generated at build time by cmake/gitInfo.cmake (git commit, branch, uncommitted changes).
// Falls back to "unknown" if the header is absent, e.g. when built outside CMake.
#if __has_include("sams_git_info.h")
#include "sams_git_info.h"
#endif
#ifndef SAMS_GIT_HASH
#define SAMS_GIT_HASH "unknown"
#endif
#ifndef SAMS_GIT_BRANCH
#define SAMS_GIT_BRANCH "unknown"
#endif
#ifndef SAMS_GIT_DIRTY
#define SAMS_GIT_DIRTY "unknown"
#endif

namespace TWOFLUID
{
    namespace pw = portableWrapper;
 
    DEVICEPREFIX INLINE LARE::T_dataType interpolate_collisional_excitation(const data_two_fluid_source &ps, LARE::T_dataType temperature, LARE::T_indexType lower_level_num, LARE::T_indexType upper_level_num);
    DEVICEPREFIX INLINE LARE::T_dataType interpolate_collisional_ionisation(const data_two_fluid_source &ps, LARE::T_dataType temperature, LARE::T_indexType level_num);
    DEVICEPREFIX INLINE LARE::T_dataType interpolate_radiative_recombination(const data_two_fluid_source &ps, LARE::T_dataType temperature, LARE::T_indexType level_num);
    DEVICEPREFIX INLINE LARE::T_dataType get_radiative_excitation(const data_two_fluid_source &ps, LARE::T_indexType lower_level_num, LARE::T_indexType upper_level_num);
    DEVICEPREFIX INLINE LARE::T_dataType get_radiative_de_excitation(const data_two_fluid_source &ps, LARE::T_indexType lower_level_num, LARE::T_indexType upper_level_num);
    DEVICEPREFIX INLINE LARE::T_dataType get_radiative_ionisation(const data_two_fluid_source &ps, LARE::T_indexType level_num);
    
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
    * Default values
    */
    template<typename T_EOS>
    void PIP<T_EOS>::defaultValues(data_two_fluid_source & data){
        data.alpha0=1.0;
        //debug_rates(data,"defaultValues");
    }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Register variables with the portable array manager.
     */
    template<typename T_EOS> 
    void PIP<T_EOS>::registerVariables(SAMS::harness &harness,data_two_fluid_source &plasma_source)
    {
//debug_rates(plasma_source,"registerVariables entry");
        auto &varRegistry = harness.variableRegistry;

        const int ghosts = 2; // 2 Ghost cells at top and bottom of each dimension

//printf("before ac \n");
        //debug_rates(plasma_source,"before ac");
        
        varRegistry.registerVariable<T_dataType>("PIPSource/ac", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));

//printf("after ac \n");
//        debug_rates(plasma_source,"after ac");
        
        varRegistry.registerVariable<T_dataType>("PIPSource/mass", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        
        varRegistry.registerVariable<T_dataType>("PIPSource/mass_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
 
// printf("before energy \n");
//        debug_rates(plasma_source,"before energy");
               
        varRegistry.registerVariable<T_dataType>("PIPSource/energy", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));

//printf("after energy \n");
//        debug_rates(plasma_source,"after energy");        

        varRegistry.registerVariable<T_dataType>("PIPSource/energy_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));

        varRegistry.registerVariable<T_dataType>("PIPSource/vx", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
        
        varRegistry.registerVariable<T_dataType>("PIPSource/vx_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));

        varRegistry.registerVariable<T_dataType>("PIPSource/vy", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
        
        varRegistry.registerVariable<T_dataType>("PIPSource/vy_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));

        varRegistry.registerVariable<T_dataType>("PIPSource/vz", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
        
        varRegistry.registerVariable<T_dataType>("PIPSource/vz_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
        
        varRegistry.registerVariable<T_dataType>("PIPSource/gm_ion", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        
        varRegistry.registerVariable<T_dataType>("PIPSource/gm_rec", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        
        varRegistry.registerVariable<T_dataType>("PIPSource/ion_loss", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));

//printf("before ionheating \n");
//        debug_rates(plasma_source,"before ionheating");        
        varRegistry.registerVariable<T_dataType>("PIPSource/ion_heating", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        
        //////////////////////////////////////////////////////////////
        //if (plasma_source.
//        printf("before Levels \n");
//        debug_rates(plasma_source,"before levels");

if (plasma_source.ion_rec_nlevel) {

        varRegistry.registerVariable<LARE::T_dataType>("level_populations", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts), SAMS::dimension("species",0));

//printf("after Levels \n");
//debug_rates(plasma_source,"after populations");
        
        varRegistry.registerVariable<LARE::T_dataType>("level_rates", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts), SAMS::dimension("species",0), SAMS::dimension("species",0));

}
//printf("after rates \n");
//debug_rates(plasma_source,"after rates");        
        //////////////////////////////////////////////////////////////
        if (plasma_source.vertex_rates){
            varRegistry.registerVariable<T_dataType>("PIPSource/rho_p_ac_vertex",  pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
            
            varRegistry.registerVariable<T_dataType>("PIPSource/rho_n_ac_vertex",  pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
            
            varRegistry.registerVariable<T_dataType>("PIPSource/gm_rec_vertex",  pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
            
            varRegistry.registerVariable<T_dataType>("PIPSource/gm_ion_vertex",  pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
        }
        
        //////////////////////////////////////////////////////////////
        
        if (plasma_source.check_source) {
            varRegistry.registerVariable<T_dataType>("PIPconserve/rho", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        
            varRegistry.registerVariable<T_dataType>("PIPconserve/energy", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
            
            varRegistry.registerVariable<T_dataType>("PIPconserve/vx",  pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
            
            varRegistry.registerVariable<T_dataType>("PIPconserve/vy",  pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
            
            varRegistry.registerVariable<T_dataType>("PIPconserve/vz",  pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
            
            varRegistry.registerVariable<T_dataType>("PIPconserve/rho_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
            
            varRegistry.registerVariable<T_dataType>("PIPconserve/energy_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
            
            varRegistry.registerVariable<T_dataType>("PIPconserve/vx_n",  pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
            
            varRegistry.registerVariable<T_dataType>("PIPconserve/vy_n",  pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
            
            varRegistry.registerVariable<T_dataType>("PIPconserve/vz_n",  pw::arrayTags::accelerated, SAMS::dimension("X", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Y", ghosts, SAMS::staggerType::HALF_CELL), SAMS::dimension("Z", ghosts, SAMS::staggerType::HALF_CELL));
        }
        
//debug_rates(plasma_source,"registerVars end");
    }
/////////////////////////////////////////////////////////////////////////////////
    template<typename T_EOS>
    void PIP<T_EOS>::allocate(data_two_fluid_source &plasma_source,SAMS::harness &harness){
        //data_two_fluid_source plasma_source;
        //data_two_fluid_source neutral_source;
       
        auto &axRegistry = harness.axisRegistry;
        auto &varRegistry = harness.variableRegistry;
        
        using Range = pw::Range;
        varRegistry.fillPPArray("PIPSource/ac", plasma_source.ac);
        pw::assign(plasma_source.ac, 0.0);
        varRegistry.fillPPArray("PIPSource/mass", plasma_source.source_mass);
        pw::assign(plasma_source.source_mass, 0.0);
        varRegistry.fillPPArray("PIPSource/mass_n", plasma_source.source_mass_n);
        pw::assign(plasma_source.source_mass_n, 0.0);
        varRegistry.fillPPArray("PIPSource/energy", plasma_source.source_energy);
        pw::assign(plasma_source.source_energy, 0.0);
        varRegistry.fillPPArray("PIPSource/energy_n", plasma_source.source_energy_n);
        pw::assign(plasma_source.source_energy_n, 0.0);
        varRegistry.fillPPArray("PIPSource/vx", plasma_source.source_v_x);
        pw::assign(plasma_source.source_v_x, 0.0);
        varRegistry.fillPPArray("PIPSource/vx_n", plasma_source.source_v_x_n);
        pw::assign(plasma_source.source_v_x_n, 0.0);
        varRegistry.fillPPArray("PIPSource/vy", plasma_source.source_v_y);
        pw::assign(plasma_source.source_v_y, 0.0);
        varRegistry.fillPPArray("PIPSource/vy_n", plasma_source.source_v_y_n);
        pw::assign(plasma_source.source_v_y_n, 0.0);
        varRegistry.fillPPArray("PIPSource/vz", plasma_source.source_v_z);
        pw::assign(plasma_source.source_v_z, 0.0);
        varRegistry.fillPPArray("PIPSource/vz_n", plasma_source.source_v_z_n);
        pw::assign(plasma_source.source_v_z_n, 0.0);
        
        varRegistry.fillPPArray("PIPSource/gm_ion", plasma_source.gm_ion);
        pw::assign(plasma_source.gm_ion, 0.0);
        varRegistry.fillPPArray("PIPSource/gm_rec", plasma_source.gm_rec);
        pw::assign(plasma_source.gm_rec, 0.0);
        varRegistry.fillPPArray("PIPSource/ion_loss", plasma_source.ion_loss);
        pw::assign(plasma_source.ion_loss, 0.0);
        varRegistry.fillPPArray("PIPSource/ion_heating", plasma_source.ion_heating);
        pw::assign(plasma_source.ion_heating, 0.0);
        
        if (plasma_source.ion_rec_nlevel) {
            varRegistry.fillPPArray("level_populations", plasma_source.level_populations);
            pw::assign(plasma_source.level_populations, 0.0);
            varRegistry.fillPPArray("level_rates", plasma_source.level_rates);
            pw::assign(plasma_source.level_rates, 0.0);
        }
        
        if (plasma_source.vertex_rates){
            varRegistry.fillPPArray("PIPSource/rho_p_ac_vertex", plasma_source.rho_p_ac_vertex);
            pw::assign(plasma_source.rho_p_ac_vertex, 0.0);
            varRegistry.fillPPArray("PIPSource/rho_n_ac_vertex", plasma_source.rho_n_ac_vertex);
            pw::assign(plasma_source.rho_n_ac_vertex, 0.0);
            varRegistry.fillPPArray("PIPSource/gm_ion_vertex", plasma_source.gm_ion_vertex);
            pw::assign(plasma_source.gm_ion_vertex, 0.0);
            varRegistry.fillPPArray("PIPSource/gm_rec_vertex", plasma_source.gm_rec_vertex);
            pw::assign(plasma_source.gm_rec_vertex, 0.0);
        }
        
    }
    
    
/////////////////////////////////////////////////////////////////////////////////
    template<typename T_EOS>
    void PIP<T_EOS>::allocate_conserved(oldData &oldData,SAMS::harness &harness){
        //data_two_fluid_source plasma_source;
        //data_two_fluid_source neutral_source;
       
        auto &axRegistry = harness.axisRegistry;
        auto &varRegistry = harness.variableRegistry;
        
        using Range = pw::Range;
        varRegistry.fillPPArray("PIPconserve/rho", oldData.rho);
        pw::assign(oldData.rho, 0.0);
        varRegistry.fillPPArray("PIPconserve/energy", oldData.energy);
        pw::assign(oldData.energy, 0.0);
        varRegistry.fillPPArray("PIPconserve/vx", oldData.vx);
        pw::assign(oldData.vx, 0.0);
        varRegistry.fillPPArray("PIPconserve/vy", oldData.vy);
        pw::assign(oldData.vy, 0.0);
        varRegistry.fillPPArray("PIPconserve/vz", oldData.vz);
        pw::assign(oldData.vz, 0.0);
        varRegistry.fillPPArray("PIPconserve/rho_n", oldData.rho_n);
        pw::assign(oldData.rho_n, 0.0);
        varRegistry.fillPPArray("PIPconserve/energy_n", oldData.energy_n);
        pw::assign(oldData.energy_n, 0.0);
        varRegistry.fillPPArray("PIPconserve/vx_n", oldData.vx_n);
        pw::assign(oldData.vx_n, 0.0);
        varRegistry.fillPPArray("PIPconserve/vy_n", oldData.vy_n);
        pw::assign(oldData.vy_n, 0.0);
        varRegistry.fillPPArray("PIPconserve/vz", oldData.vz_n);
        pw::assign(oldData.vz_n, 0.0);        
    }    
////////////////////////////////////////////////////////////////////////////////////////
     template<typename T_EOS>
    void PIP<T_EOS>::get_two_fluid_source(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){

        //two_fluid_properties two_fluid_flags; //Move to source structure
        
        //if (two_fluid_flags.ion_rec_nlevel){        
        ////    ion_rec_rates_nlevel(data,dataNeutral);
        //}
        
        pw::assign(plasma_source.source_mass, 0.0);
        pw::assign(plasma_source.source_mass_n, 0.0);
        pw::assign(plasma_source.source_energy, 0.0);
        pw::assign(plasma_source.source_energy_n, 0.0);
        pw::assign(plasma_source.source_v_x, 0.0);
        pw::assign(plasma_source.source_v_x_n, 0.0);
        pw::assign(plasma_source.source_v_y, 0.0);
        pw::assign(plasma_source.source_v_y_n, 0.0);
        pw::assign(plasma_source.source_v_z, 0.0);
        pw::assign(plasma_source.source_v_z_n, 0.0);
        
        //Calculate the source terms for the two-fluid interactions
        get_collisional_source_terms(data,dataNeutral,plasma_source);
        
        //Calculate the source terms for Ionisation/recombination
        if (plasma_source.ion_rec_empirical) {
            ion_rec_rates_empirical(data,dataNeutral, plasma_source);
            get_ion_rec_source_terms(data,dataNeutral,plasma_source);
        };
        if (plasma_source.ion_rec_nlevel) {
            ion_rec_rates_nlevel(data,dataNeutral, plasma_source);
            get_ion_rec_source_terms(data,dataNeutral,plasma_source);
        };

    };
////////////////////////////////////////////////////////////////////////////////////////
     template<typename T_EOS>
    void PIP<T_EOS>::apply_two_fluid_source(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){

        //two_fluid_properties two_fluid_flags; //Move to source structure
        
        //Get the ionisation rates
        //if (two_fluid_flags.ion_rec_empirical){        
        //    ion_rec_rates_empirical(data,dataNeutral);
        //}
        //if (two_fluid_flags.ion_rec_nlevel){        
        ////    ion_rec_rates_nlevel(data,dataNeutral);
        //}
        
        //Calculate the source terms for the two-fluid interactions
        //get_collisional_source_terms(data,dataNeutral,plasma_source);
        
        //Calculate the source terms for Ionisation/recombination
        //if (two_fluid_flags.ion_rec_empirical) get_ion_rec_source_terms(data,dataNeutral,plasma_source);
        
        if (plasma_source.collisions){
            using Range = portableWrapper::Range;
            portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                //Note that the factor of 0.5 in these is due to Strang splitting
                //Mass exchange terms
                data.rho(ix,iy,iz)+=plasma_source.substep_dt*plasma_source.source_mass(ix,iy,iz);
                dataNeutral.rho(ix,iy,iz)+=plasma_source.substep_dt*plasma_source.source_mass_n(ix,iy,iz);
                
                //Apply the velocity exchange terms
                data.vx(ix,iy,iz)       +=plasma_source.substep_dt*plasma_source.source_v_x(ix,iy,iz);
                dataNeutral.vx(ix,iy,iz)+=plasma_source.substep_dt*plasma_source.source_v_x_n(ix,iy,iz);
                
                data.vy(ix,iy,iz)       +=plasma_source.substep_dt*plasma_source.source_v_y(ix,iy,iz);
                dataNeutral.vy(ix,iy,iz)+=plasma_source.substep_dt*plasma_source.source_v_y_n(ix,iy,iz);
                
                data.vz(ix,iy,iz)       +=plasma_source.substep_dt*plasma_source.source_v_z(ix,iy,iz);
                dataNeutral.vz(ix,iy,iz)+=plasma_source.substep_dt*plasma_source.source_v_z_n(ix,iy,iz);
                
                //Energy source terms - the 3/2 here needs fixing
                data.energy_ion(ix,iy,iz)+=plasma_source.substep_dt*plasma_source.source_energy(ix,iy,iz);
                //data.energy_electron(ix,iy,iz)+=0.5*data.dt*plasma_source.source_energy(ix,iy,iz);
                dataNeutral.energy(ix,iy,iz)+=plasma_source.substep_dt*plasma_source.source_energy_n(ix,iy,iz);                 
            }, Range(-1,data.nx), Range(-1,data.ny), Range(-1,data.nz));
        }
        
    };

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //Get the collisional coupling coefficient
     template<typename T_EOS>
        void PIP<T_EOS>::get_ac(LARE::LARE3DST<T_EOS>::simulationData &data, LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){

            using Range = portableWrapper::Range;
            portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                //Get Temperatures
                SAMS::T_dataType  temperature_ion = data.gas_gamma*data.energy_ion(ix,iy,iz)*(data.gas_gamma-1.0)*0.5;
                SAMS::T_dataType  temperature_neutral = data.gas_gamma*dataNeutral.energy(ix,iy,iz)*(data.gas_gamma-1.0);
                plasma_source.ac(ix,iy,iz)=plasma_source.alpha0*std::sqrt(0.5*(temperature_neutral+temperature_ion));
            	}, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
            	
        	if (plasma_source.vertex_rates){
            	using Range = portableWrapper::Range;
                portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                    //Get Temperatures
                    //SAMS::T_dataType  temperature_ion = data.gas_gamma*data.energy_ion(ix,iy,iz)*(data.gas_gamma-1.0)*0.5;
                    //SAMS::T_dataType  temperature_neutral = data.gas_gamma*dataNeutral.energy(ix,iy,iz)*(data.gas_gamma-1.0);
                    plasma_source.rho_p_ac_vertex(ix,iy,iz)=0.125*plasma_source.alpha0*(
                    data.rho(ix,iy,iz)*std::sqrt(0.5*(data.gas_gamma*dataNeutral.energy(ix,iy,iz)*(data.gas_gamma-1.0)+ data.gas_gamma*data.energy_ion(ix,iy,iz)*(data.gas_gamma-1.0)*0.5))+
                    data.rho(ix+1,iy,iz)*std::sqrt(0.5*(data.gas_gamma*dataNeutral.energy(ix+1,iy,iz)*(data.gas_gamma-1.0)+ data.gas_gamma*data.energy_ion(ix+1,iy,iz)*(data.gas_gamma-1.0)*0.5))+
                    data.rho(ix,iy+1,iz)*std::sqrt(0.5*(data.gas_gamma*dataNeutral.energy(ix,iy+1,iz)*(data.gas_gamma-1.0)+ data.gas_gamma*data.energy_ion(ix,iy+1,iz)*(data.gas_gamma-1.0)*0.5))+
                    data.rho(ix+1,iy+1,iz)*std::sqrt(0.5*(data.gas_gamma*dataNeutral.energy(ix+1,iy+1,iz)*(data.gas_gamma-1.0)+ data.gas_gamma*data.energy_ion(ix+1,iy+1,iz)*(data.gas_gamma-1.0)*0.5))+
                    data.rho(ix,iy,iz+1)*std::sqrt(0.5*(data.gas_gamma*dataNeutral.energy(ix,iy,iz+1)*(data.gas_gamma-1.0)+ data.gas_gamma*data.energy_ion(ix,iy,iz+1)*(data.gas_gamma-1.0)*0.5))+
                    data.rho(ix+1,iy,iz+1)*std::sqrt(0.5*(data.gas_gamma*dataNeutral.energy(ix+1,iy,iz+1)*(data.gas_gamma-1.0)+ data.gas_gamma*data.energy_ion(ix+1,iy,iz+1)*(data.gas_gamma-1.0)*0.5))+
                    data.rho(ix,iy+1,iz+1)*std::sqrt(0.5*(data.gas_gamma*dataNeutral.energy(ix,iy+1,iz+1)*(data.gas_gamma-1.0)+ data.gas_gamma*data.energy_ion(ix,iy+1,iz+1)*(data.gas_gamma-1.0)*0.5))+
                    data.rho(ix+1,iy+1,iz+1)*std::sqrt(0.5*(data.gas_gamma*dataNeutral.energy(ix+1,iy+1,iz+1)*(data.gas_gamma-1.0)+ data.gas_gamma*data.energy_ion(ix+1,iy+1,iz+1)*(data.gas_gamma-1.0)*0.5))
                    );
                    plasma_source.rho_n_ac_vertex(ix,iy,iz)=0.125*plasma_source.alpha0*(
                    dataNeutral.rho(ix,iy,iz)*std::sqrt(0.5*(data.gas_gamma*dataNeutral.energy(ix,iy,iz)*(data.gas_gamma-1.0)+ data.gas_gamma*data.energy_ion(ix,iy,iz)*(data.gas_gamma-1.0)*0.5))+
                    dataNeutral.rho(ix+1,iy,iz)*std::sqrt(0.5*(data.gas_gamma*dataNeutral.energy(ix+1,iy,iz)*(data.gas_gamma-1.0)+ data.gas_gamma*data.energy_ion(ix+1,iy,iz)*(data.gas_gamma-1.0)*0.5))+
                    dataNeutral.rho(ix,iy+1,iz)*std::sqrt(0.5*(data.gas_gamma*dataNeutral.energy(ix,iy+1,iz)*(data.gas_gamma-1.0)+ data.gas_gamma*data.energy_ion(ix,iy+1,iz)*(data.gas_gamma-1.0)*0.5))+
                    dataNeutral.rho(ix+1,iy+1,iz)*std::sqrt(0.5*(data.gas_gamma*dataNeutral.energy(ix+1,iy+1,iz)*(data.gas_gamma-1.0)+ data.gas_gamma*data.energy_ion(ix+1,iy+1,iz)*(data.gas_gamma-1.0)*0.5))+
                    dataNeutral.rho(ix,iy,iz+1)*std::sqrt(0.5*(data.gas_gamma*dataNeutral.energy(ix,iy,iz+1)*(data.gas_gamma-1.0)+ data.gas_gamma*data.energy_ion(ix,iy,iz+1)*(data.gas_gamma-1.0)*0.5))+
                    dataNeutral.rho(ix+1,iy,iz+1)*std::sqrt(0.5*(data.gas_gamma*dataNeutral.energy(ix+1,iy,iz+1)*(data.gas_gamma-1.0)+ data.gas_gamma*data.energy_ion(ix+1,iy,iz+1)*(data.gas_gamma-1.0)*0.5))+
                    dataNeutral.rho(ix,iy+1,iz+1)*std::sqrt(0.5*(data.gas_gamma*dataNeutral.energy(ix,iy+1,iz+1)*(data.gas_gamma-1.0)+ data.gas_gamma*data.energy_ion(ix,iy+1,iz+1)*(data.gas_gamma-1.0)*0.5))+
                    dataNeutral.rho(ix+1,iy+1,iz+1)*std::sqrt(0.5*(data.gas_gamma*dataNeutral.energy(ix+1,iy+1,iz+1)*(data.gas_gamma-1.0)+ data.gas_gamma*data.energy_ion(ix+1,iy+1,iz+1)*(data.gas_gamma-1.0)*0.5))
                    );
                	}, Range(-1,data.nx), Range(-1,data.ny), Range(-1,data.nz));
        	}

        };
        
////////////////////////////////////////////////////////////////////////////////////////
        //Formulation from Snow+2021 paper
        //Empirical estimates for the rates
        //Controlled using the data.ion_rec_empirical in control.cpp
     template<typename T_EOS>
        void PIP<T_EOS>::ion_rec_rates_empirical(LARE::LARE3DST<T_EOS>::simulationData &data, LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){

            //Much of this should go elsewhere
            SAMS::T_dataType T0=plasma_source.T0; //Reference temperature
            SAMS::T_dataType n0=plasma_source.n0;//data.ne_reference; //Reference electron number density
            SAMS::T_dataType t_ir=plasma_source.t_ir; //Reference recombination timescale (relative to collisional timescale)
            SAMS::T_dataType kb_ev=plasma_source.kb_ev; //Kb in eV/K

            SAMS::T_dataType  Te_0=T0/1.1604e4; //Calculate electron temperature in eV
            SAMS::T_dataType  rec_fac=2.6e-19*(n0*1.0e6)/std::sqrt(Te_0);  //reference recombination rate (n0 converted to m^-3)

            //initial equilibrium fractions
            SAMS::T_dataType  ioneq=(2.6e-19/std::sqrt(Te_0))/(2.91e-14/(0.232+13.6/Te_0)*std::pow(13.6/Te_0,0.39)*std::exp(-13.6/Te_0));
            SAMS::T_dataType  f_n=ioneq/(ioneq+1.0);
            SAMS::T_dataType  f_p=1.0-f_n;
            SAMS::T_dataType  f_p_p=2.0*f_p/(f_n+2.0*f_p);
            
            SAMS::T_dataType  tfac=0.5*data.gas_gamma*f_p_p/f_p; //Normalisation assumes bulk sound speed normalisation
            

            using Range = portableWrapper::Range;
            portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                //Get Temperatures
                //SAMS::T_dataType  temperature_electron = 2.0*data.gas_gamma*data.energy_electron(ix,iy,iz)*(data.gas_gamma-1.0);
                SAMS::T_dataType  temperature_electron = 0.5*data.gas_gamma*data.energy_ion(ix,iy,iz)*(data.gas_gamma-1.0);
                SAMS::T_dataType  numberDensity_electron=data.rho(ix,iy,iz); 

                //Get ionisation and recomination rates
            	plasma_source.gm_rec(ix,iy,iz)=numberDensity_electron/std::sqrt(temperature_electron)*t_ir/f_p*std::sqrt(tfac);
            	plasma_source.gm_ion(ix,iy,iz)=2.91e-14*(n0*1.0e6)*numberDensity_electron*std::exp(-13.6/Te_0/temperature_electron*tfac)*std::pow(13.6/Te_0/temperature_electron*tfac,0.39);
            	plasma_source.gm_ion(ix,iy,iz)=plasma_source.gm_ion(ix,iy,iz)/(0.232+13.6/Te_0/temperature_electron*tfac)/rec_fac/f_p *t_ir;    
                plasma_source.ion_loss(ix,iy,iz)=(-plasma_source.gm_ion(ix,iy,iz)*dataNeutral.rho(ix,iy,iz))*
                                  13.6/kb_ev/T0/data.gas_gamma;  	
            	//printf("%f %f %f %f %f \n",f_p,data.energy_ion(ix,iy,iz)*(data.gas_gamma-1.0)*data.rho(ix,iy,iz), temperature_electron,plasma_source.gm_rec(ix,iy,iz),plasma_source.gm_ion(ix,iy,iz));    
            }, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
            
        	if (plasma_source.vertex_rates){
            	using Range = portableWrapper::Range;
                portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                    plasma_source.gm_rec_vertex(ix,iy,iz)=0.125*t_ir/f_p*std::sqrt(tfac)*(
                    data.rho(ix  ,iy  ,iz  )/std::sqrt(0.5*data.gas_gamma*data.energy_ion(ix  ,iy  ,iz  )*(data.gas_gamma-1.0))+
                    data.rho(ix+1,iy  ,iz  )/std::sqrt(0.5*data.gas_gamma*data.energy_ion(ix+1,iy  ,iz  )*(data.gas_gamma-1.0))+
                    data.rho(ix  ,iy+1,iz  )/std::sqrt(0.5*data.gas_gamma*data.energy_ion(ix  ,iy+1,iz  )*(data.gas_gamma-1.0))+
                    data.rho(ix+1,iy+1,iz  )/std::sqrt(0.5*data.gas_gamma*data.energy_ion(ix+1,iy+1,iz  )*(data.gas_gamma-1.0))+
                    data.rho(ix  ,iy  ,iz+1)/std::sqrt(0.5*data.gas_gamma*data.energy_ion(ix  ,iy  ,iz+1)*(data.gas_gamma-1.0))+
                    data.rho(ix+1,iy  ,iz+1)/std::sqrt(0.5*data.gas_gamma*data.energy_ion(ix+1,iy  ,iz+1)*(data.gas_gamma-1.0))+
                    data.rho(ix  ,iy+1,iz+1)/std::sqrt(0.5*data.gas_gamma*data.energy_ion(ix  ,iy+1,iz+1)*(data.gas_gamma-1.0))+
                    data.rho(ix+1,iy+1,iz+1)/std::sqrt(0.5*data.gas_gamma*data.energy_ion(ix+1,iy+1,iz+1)*(data.gas_gamma-1.0))
                    );
                    
                    SAMS::T_dataType Te1=0.5*data.gas_gamma*data.energy_ion(ix  ,iy  ,iz  )*(data.gas_gamma-1.0);
                    SAMS::T_dataType Te2=0.5*data.gas_gamma*data.energy_ion(ix+1,iy  ,iz  )*(data.gas_gamma-1.0);
                    SAMS::T_dataType Te3=0.5*data.gas_gamma*data.energy_ion(ix  ,iy+1,iz  )*(data.gas_gamma-1.0);
                    SAMS::T_dataType Te4=0.5*data.gas_gamma*data.energy_ion(ix+1,iy+1,iz  )*(data.gas_gamma-1.0);
                    SAMS::T_dataType Te5=0.5*data.gas_gamma*data.energy_ion(ix  ,iy  ,iz+1)*(data.gas_gamma-1.0);
                    SAMS::T_dataType Te6=0.5*data.gas_gamma*data.energy_ion(ix+1,iy  ,iz+1)*(data.gas_gamma-1.0);
                    SAMS::T_dataType Te7=0.5*data.gas_gamma*data.energy_ion(ix  ,iy+1,iz+1)*(data.gas_gamma-1.0);
                    SAMS::T_dataType Te8=0.5*data.gas_gamma*data.energy_ion(ix+1,iy+1,iz+1)*(data.gas_gamma-1.0);
                    
                    plasma_source.gm_ion_vertex(ix,iy,iz)=2.91e-14*(n0*1.0e6)/rec_fac/f_p *t_ir*(
                    data.rho(ix  ,iy  ,iz  )*std::exp(-13.6/Te_0/Te1*tfac)*std::pow(13.6/Te_0/Te1*tfac,0.39)/(0.232+13.6/Te_0/Te1*tfac)+
                    data.rho(ix+1,iy  ,iz  )*std::exp(-13.6/Te_0/Te2*tfac)*std::pow(13.6/Te_0/Te2*tfac,0.39)/(0.232+13.6/Te_0/Te2*tfac)+
                    data.rho(ix  ,iy+1,iz  )*std::exp(-13.6/Te_0/Te3*tfac)*std::pow(13.6/Te_0/Te3*tfac,0.39)/(0.232+13.6/Te_0/Te3*tfac)+
                    data.rho(ix+1,iy+1,iz  )*std::exp(-13.6/Te_0/Te4*tfac)*std::pow(13.6/Te_0/Te4*tfac,0.39)/(0.232+13.6/Te_0/Te4*tfac)+
                    data.rho(ix  ,iy  ,iz+1)*std::exp(-13.6/Te_0/Te5*tfac)*std::pow(13.6/Te_0/Te5*tfac,0.39)/(0.232+13.6/Te_0/Te5*tfac)+
                    data.rho(ix+1,iy  ,iz+1)*std::exp(-13.6/Te_0/Te6*tfac)*std::pow(13.6/Te_0/Te6*tfac,0.39)/(0.232+13.6/Te_0/Te6*tfac)+
                    data.rho(ix  ,iy+1,iz+1)*std::exp(-13.6/Te_0/Te7*tfac)*std::pow(13.6/Te_0/Te7*tfac,0.39)/(0.232+13.6/Te_0/Te7*tfac)+
                    data.rho(ix+1,iy+1,iz+1)*std::exp(-13.6/Te_0/Te8*tfac)*std::pow(13.6/Te_0/Te8*tfac,0.39)/(0.232+13.6/Te_0/Te8*tfac)
                    );    
        	    }, Range(-1,data.nx), Range(-1,data.ny), Range(-1,data.nz));
    	    }
        };
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template<typename T_EOS>
        void PIP<T_EOS>::set_bc_heating(LARE::LARE3DST<T_EOS>::simulationData &data,data_two_fluid_source &plasma_source){
            using Range = portableWrapper::Range;
            portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
                plasma_source.ion_heating(ix,iy,iz)=-plasma_source.ion_loss(ix,iy,iz);
            }, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));        
        }
        
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template<typename T_EOS>
        void PIP<T_EOS>::set_reference_recombination(LARE::LARE3DST<T_EOS>::simulationData &data,data_two_fluid_source &plasma_source){  
            if (plasma_source.ion_rec_empirical){
                //Note that this isn't actually used at the moment.
                LARE::T_dataType Te_0=plasma_source.T0/1.1604e4; //Calculate electron temperature in eV
                plasma_source.ref_rec=2.6e-19*(plasma_source.n0*1.0e6)/std::sqrt(Te_0);  //reference recombination rate (n0 converted to m^-3)
            };
            if (plasma_source.ion_rec_nlevel) { 
                plasma_source.ref_rec=0.0;
                LARE::T_dataType kb_ev=8.617333e-5; //Kb in eV/K
                LARE::T_dataType h_ev=4.135668e-15; //plancks constant in ev s
                LARE::T_dataType mass_electron=9.10938356e-31; //electron mass in kg
                LARE::T_indexType nLevels=6;
                
                std::vector<LARE::T_dataType> Eion = {0,13.6,3.4,1.51,0.85,0.54,0.0};
                std::vector<LARE::T_dataType> gn = {0,2,8,18,32,50,1};
                
                for (LARE::T_indexType lower_level = 1; lower_level < nLevels; ++lower_level) {
                    LARE::T_dataType rate_coefficient = interpolate_collisional_ionisation(plasma_source,
                                                                                           plasma_source.T0, 
                                                                                           lower_level);
                    //Ionisation rate
                    LARE::T_dataType ion_rate=plasma_source.n0*rate_coefficient* 
                                            std::exp((Eion[lower_level]-Eion[nLevels])/(kb_ev*plasma_source.T0));
                    //Recombination rate
                    LARE::T_dataType sahaRatio= plasma_source.n0*gn[lower_level]* 
                    std::pow(2.0*std::numbers::pi*kb_ev*plasma_source.T0*mass_electron/h_ev,-3.0/2.0);
                    LARE::T_dataType rec_rate=sahaRatio*ion_rate;
                    
                    //printf("Reference recombination rate %G %G %G\n",rec_rate,plasma_source.n0,plasma_source.T0); 
                    plasma_source.ref_rec+=rec_rate;
                };
                plasma_source.ref_rec=plasma_source.ref_rec*plasma_source.n0; //This all needs checking
                printf("Reference recombination rate %e \n",plasma_source.ref_rec); 
            };    
            //exit(0);
        }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//multi-level hydrogen rates
    template<typename T_EOS>
        void PIP<T_EOS>::ion_rec_rates_nlevel(LARE::LARE3DST<T_EOS>::simulationData &data, LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){

            //Much of this should go elsewhere
            SAMS::T_dataType T0=plasma_source.T0; //Reference temperature
            SAMS::T_dataType n0=plasma_source.n0;//data.ne_reference; //Reference electron number density
            SAMS::T_dataType t_ir=plasma_source.t_ir; //Reference recombination timescale (relative to collisional timescale)
            LARE::T_indexType nLevels=6;

            LARE::T_dataType Te_0=T0/1.1604e4; //Calculate electron temperature in eV
            LARE::T_dataType rec_fac=2.6e-19*(n0*1.0e6)/std::sqrt(Te_0);  //reference recombination rate (n0 converted to m^-3)

            //initial equilibrium fractions
            LARE::T_dataType ioneq=(2.6e-19/std::sqrt(Te_0))/(2.91e-14/(0.232+13.6/Te_0)*std::pow(13.6/Te_0,0.39)*std::exp(-13.6/Te_0));
            LARE::T_dataType f_n=ioneq/(ioneq+1.0);
            LARE::T_dataType f_p=1.0-f_n;
            LARE::T_dataType f_p_p=2.0*f_p/(f_n+2.0*f_p);
            
            LARE::T_dataType tfac=0.5*f_p_p/f_p; //Normalisation assumes bulk sound speed normalisation
            
            LARE::T_dataType kb_ev=8.617333e-5; //Kb in eV/K
            LARE::T_dataType h_ev=4.135668e-15; //plancks constant in ev s
            LARE::T_dataType mass_electron=9.10938356e-31; //electron mass in kg
            
            std::vector<LARE::T_dataType> Eion = {0,13.6,3.4,1.51,0.85,0.54,0.0};
            //for (int i = 1; i <= 6; ++i) Eion[i] = Eion[i] / 13.6 * 2.18e-18;
            
            std::vector<LARE::T_dataType> gn = {0,2,8,18,32,50,1};
            
            /////////////////////////////
            using Range = portableWrapper::Range;
            portableWrapper::applyKernel(LAMBDA(LARE::T_indexType ix, LARE::T_indexType iy, LARE::T_indexType iz) {
                //Get Temperature and number density
                LARE::T_dataType temperature_electron = 0.5*data.gas_gamma*data.energy_ion(ix,iy,iz)*(data.gas_gamma-1.0)*T0;
                LARE::T_dataType numberDensity_electron=data.rho(ix,iy,iz)*n0; 
                
                
                for (LARE::T_indexType lower_level = 1; lower_level < nLevels; ++lower_level) {
                    //Loop over (de)excitation
                    for (LARE::T_indexType upper_level = lower_level + 1; upper_level < nLevels; ++upper_level) {
                        //Get collisional rate coefficient
                        LARE::T_dataType rate_coefficient = interpolate_collisional_excitation(plasma_source, temperature_electron, lower_level, upper_level);
                        // triangular work for this cell
                        //Excitation rate
                        plasma_source.level_rates(ix,iy,iz,lower_level,upper_level)=gn[lower_level]/gn[upper_level]*numberDensity_electron*rate_coefficient;
                        //De-Excitation rate
                        plasma_source.level_rates(ix,iy,iz,upper_level,lower_level)=numberDensity_electron*rate_coefficient*
                                                std::exp((Eion[lower_level]-Eion[upper_level])/(kb_ev*temperature_electron));
                                                
                        plasma_source.ion_loss(ix,iy,iz)-=Eion[lower_level]*
                                                          plasma_source.level_populations(ix,iy,iz,lower_level)*
                                                          plasma_source.level_rates(ix,iy,iz,lower_level,upper_level)
                                                          /kb_ev/T0/data.gas_gamma;
                        plasma_source.ion_heating(ix,iy,iz)+=Eion[upper_level]*
                                                             plasma_source.level_populations(ix,iy,iz,upper_level)*
                                                             plasma_source.level_rates(ix,iy,iz,upper_level,lower_level)
                                                             /kb_ev/T0/data.gas_gamma;
                                                
                        // Radiative Rates
                        LARE::T_dataType rad_exciation = get_radiative_excitation(plasma_source, lower_level, upper_level);
                        LARE::T_dataType rad_deexciation = get_radiative_de_excitation(plasma_source, lower_level, upper_level);
                    }
                    //Calculate ionisation and recombination
                    //LARE::T_dataType rate_coefficient = interpolate_rates(plasma_source, temperature_electron, 
                    //                                        lower_level, nLevels);
                    LARE::T_dataType rate_coefficient = interpolate_collisional_ionisation(plasma_source,
                                                                                           temperature_electron, 
                                                                                           lower_level);
                    //Ionisation rate
                    plasma_source.level_rates(ix,iy,iz,lower_level,nLevels)=numberDensity_electron*rate_coefficient* 
                                            std::exp((Eion[lower_level]-Eion[nLevels])/(kb_ev*temperature_electron));
                    //Recombination rate
                    LARE::T_dataType sahaRatio= numberDensity_electron*gn[lower_level]* 
                    std::pow(2.0*std::numbers::pi*kb_ev*temperature_electron*mass_electron/h_ev,-3.0/2.0);
                    plasma_source.level_rates(ix,iy,iz,nLevels,lower_level)=sahaRatio*plasma_source.level_rates(ix,iy,iz,
                                                                                                                lower_level,
                                                                                                                nLevels);
                                                                                                                
                    //Radiative ionisation
                    LARE::T_dataType rad_ion_coefficient = get_radiative_ionisation(plasma_source, lower_level);
                    LARE::T_dataType rad_rec_coefficient = sahaRatio*interpolate_radiative_recombination(plasma_source, temperature_electron, lower_level);
                    
                }
                //Get ionisation and recomination rates
                plasma_source.gm_rec(ix,iy,iz)=0.0;
                plasma_source.gm_ion(ix,iy,iz)=0.0;
            	for (LARE::T_indexType lower_level = 1; lower_level < nLevels; ++lower_level) {
            	    plasma_source.gm_rec(ix,iy,iz)+=plasma_source.level_rates(ix,iy,iz,nLevels,lower_level);
            	    plasma_source.gm_ion(ix,iy,iz)+=plasma_source.level_rates(ix,iy,iz,lower_level,nLevels)*plasma_source.level_populations(ix,iy,iz,lower_level);
            	}
            	plasma_source.gm_rec(ix,iy,iz)=plasma_source.gm_rec(ix,iy,iz)/plasma_source.ref_rec;
            	plasma_source.gm_ion(ix,iy,iz)=plasma_source.gm_ion(ix,iy,iz)/dataNeutral.rho(ix,iy,iz)/plasma_source.ref_rec; //Need to normalise the ionisation rate 
            }, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
            portableWrapper::fence();
            /*
fprintf(stdout, "  radiative_excitation     (%d->%d) : %e\n",
            static_cast<int>(lower_level), static_cast<int>(upper_level),
            get_radiative_excitation(plasma_source, lower_level, upper_level));
    fprintf(stdout, "  radiative_de_excitation  (%d->%d) : %e\n",
            static_cast<int>(upper_level), static_cast<int>(lower_level),
            get_radiative_de_excitation(plasma_source, lower_level, upper_level));
    fprintf(stdout, "  collisional_ionisation   (level=%d) : %e\n",
            static_cast<int>(lower_level),
            interpolate_collisional_ionisation(plasma_source, T_test, lower_level));
    fprintf(stdout, "  collisional_ionisation   (level=%d) : %e\n",
            static_cast<int>(upper_level),
            interpolate_collisional_ionisation(plasma_source, T_test, upper_level));
    fprintf(stdout, "  radiative_recombination  (level=%d) : %e\n",
            static_cast<int>(lower_level),
            interpolate_radiative_recombination(plasma_source, T_test, lower_level));
    fprintf(stdout, "  radiative_recombination  (level=%d) : %e\n",
            static_cast<int>(upper_level),
            interpolate_radiative_recombination(plasma_source, T_test, upper_level));
    fprintf(stdout, "  radiative_ionisation     (level=%d) : %e\n",
            static_cast<int>(lower_level),
            get_radiative_ionisation(plasma_source, lower_level));
    fprintf(stdout, "  radiative_ionisation     (level=%d) : %e\n",
            static_cast<int>(upper_level),
            get_radiative_ionisation(plasma_source, upper_level));
            	printf("%f %f %f %f %f \n",f_p,data.energy_electron(ix,iy,iz)*(data.gas_gamma-1.0)*data.rho(ix,iy,iz), temperature_electron,plasma_source.gm_rec(ix,iy,iz),plasma_source.gm_ion(ix,iy,iz));    
            
            */
        };
////////////////////////////////////////////////////////////////////////////////////////
//Get the source terms for the IR rates
    template<typename T_EOS>
 void PIP<T_EOS>::get_collisional_source_terms(LARE::LARE3DST<T_EOS>::simulationData &data, LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){	

    if (plasma_source.vertex_rates){
        //This part of the loop uses the rho*ac at the vertex for the momentum source terms directly
        using Range = portableWrapper::Range;
        portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {

            //Apply the velocity exchange terms
            plasma_source.source_v_x(ix,iy,iz)=plasma_source.rho_n_ac_vertex(ix,iy,iz)*(dataNeutral.vx(ix,iy,iz)-data.vx(ix,iy,iz));
            plasma_source.source_v_x_n(ix,iy,iz)=-plasma_source.rho_p_ac_vertex(ix,iy,iz)*(dataNeutral.vx(ix,iy,iz)-data.vx(ix,iy,iz));
            
            //printf("v_n %f %f \n", plasma_source.source_v_x(ix,iy,iz),plasma_source.source_v_x_n(ix,iy,iz));
            
            plasma_source.source_v_y(ix,iy,iz)=plasma_source.rho_n_ac_vertex(ix,iy,iz)*(dataNeutral.vy(ix,iy,iz)-data.vy(ix,iy,iz));
            plasma_source.source_v_y_n(ix,iy,iz)=-plasma_source.rho_p_ac_vertex(ix,iy,iz)*(dataNeutral.vy(ix,iy,iz)-data.vy(ix,iy,iz));
            
            plasma_source.source_v_z(ix,iy,iz)=plasma_source.rho_n_ac_vertex(ix,iy,iz)*(dataNeutral.vz(ix,iy,iz)-data.vz(ix,iy,iz));
            plasma_source.source_v_z_n(ix,iy,iz)=-plasma_source.rho_p_ac_vertex(ix,iy,iz)*(dataNeutral.vz(ix,iy,iz)-data.vz(ix,iy,iz));
        }, Range(-1,data.nx), Range(-1,data.ny), Range(-1,data.nz));
    } else {
        //This part of the loop interpolates the ac and rho to the vertex for the momentum source terms
        using Range = portableWrapper::Range;
        portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
            //Get ac and rho at the location of v (vertex)
            SAMS::T_dataType  ac_vertex=(plasma_source.ac(ix  , iy  , iz  ) + 
                                            plasma_source.ac(ix+1, iy  , iz  ) + 
                                            plasma_source.ac(ix  , iy+1, iz  ) + 
                                            plasma_source.ac(ix+1, iy+1, iz  ) + 
                                            plasma_source.ac(ix  , iy  , iz+1) + 
                                            plasma_source.ac(ix+1, iy  , iz+1) + 
                                            plasma_source.ac(ix  , iy+1, iz+1) + 
                                            plasma_source.ac(ix+1, iy+1, iz+1))* 
                                            0.125;
            SAMS::T_dataType  rho_plasma_vertex=  (data.rho(ix  , iy  , iz  ) + 
                                            data.rho(ix+1, iy  , iz  ) + 
                                            data.rho(ix  , iy+1, iz  ) + 
                                            data.rho(ix+1, iy+1, iz  ) + 
                                            data.rho(ix  , iy  , iz+1) + 
                                            data.rho(ix+1, iy  , iz+1) + 
                                            data.rho(ix  , iy+1, iz+1) + 
                                            data.rho(ix+1, iy+1, iz+1))* 
                                            0.125;
            SAMS::T_dataType  rho_neutral_vertex=  (dataNeutral.rho(ix  , iy  , iz  ) + 
                                             dataNeutral.rho(ix+1, iy  , iz  ) + 
                                             dataNeutral.rho(ix  , iy+1, iz  ) + 
                                             dataNeutral.rho(ix+1, iy+1, iz  ) + 
                                             dataNeutral.rho(ix  , iy  , iz+1) + 
                                             dataNeutral.rho(ix+1, iy  , iz+1) + 
                                             dataNeutral.rho(ix  , iy+1, iz+1) + 
                                             dataNeutral.rho(ix+1, iy+1, iz+1))* 
                                             0.125;
            
                    
            //Apply the velocity exchange terms
            plasma_source.source_v_x(ix,iy,iz)=ac_vertex*rho_neutral_vertex*(dataNeutral.vx(ix,iy,iz)-data.vx(ix,iy,iz));
            plasma_source.source_v_x_n(ix,iy,iz)=-ac_vertex*rho_plasma_vertex*(dataNeutral.vx(ix,iy,iz)-data.vx(ix,iy,iz));

            plasma_source.source_v_y(ix,iy,iz)=ac_vertex*rho_neutral_vertex*(dataNeutral.vy(ix,iy,iz)-data.vy(ix,iy,iz));
            plasma_source.source_v_y_n(ix,iy,iz)=-ac_vertex*rho_plasma_vertex*(dataNeutral.vy(ix,iy,iz)-data.vy(ix,iy,iz));
            
            plasma_source.source_v_z(ix,iy,iz)=ac_vertex*rho_neutral_vertex*(dataNeutral.vz(ix,iy,iz)-data.vz(ix,iy,iz));
            plasma_source.source_v_z_n(ix,iy,iz)=-ac_vertex*rho_plasma_vertex*(dataNeutral.vz(ix,iy,iz)-data.vz(ix,iy,iz));
        }, Range(-1,data.nx), Range(-1,data.ny), Range(-1,data.nz));
    }
        
        using Range = portableWrapper::Range;
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
    
        //Get Temperatures
        SAMS::T_dataType  temperature_ion = data.gas_gamma*data.energy_ion(ix,iy,iz)*(data.gas_gamma-1.0)/2.0;
        //SAMS::T_dataType  temperature_electron = data.gas_gamma*data.energy_electron(ix,iy,iz)*(data.gas_gamma-1.0);
        SAMS::T_dataType  temperature_neutral = data.gas_gamma*dataNeutral.energy(ix,iy,iz)*(data.gas_gamma-1.0);
        //printf("getting vx at centre \n");
        //Get velocity at cell centre
        SAMS::T_dataType  vx_centre=(data.vx(ix,iy,iz)+
                              data.vx(ix  ,iy-1,iz)+
                              data.vx(ix  ,iy  ,iz-1)+
                              data.vx(ix  ,iy-1,iz-1)+
                              data.vx(ix-1,iy  ,iz  )+
                              data.vx(ix-1,iy-1,iz  )+
                              data.vx(ix-1,iy  ,iz-1)+
                              data.vx(ix-1,iy-1,iz-1))*
                              0.125;
        SAMS::T_dataType  vy_centre=(data.vy(ix,iy,iz)+
                              data.vy(ix  ,iy-1,iz)+
                              data.vy(ix  ,iy  ,iz-1)+
                              data.vy(ix  ,iy-1,iz-1)+
                              data.vy(ix-1,iy  ,iz  )+
                              data.vy(ix-1,iy-1,iz  )+
                              data.vy(ix-1,iy  ,iz-1)+
                              data.vy(ix-1,iy-1,iz-1))*
                              0.125;
        SAMS::T_dataType  vz_centre=(data.vz(ix,iy,iz)+
                              data.vz(ix  ,iy-1,iz)+
                              data.vz(ix  ,iy  ,iz-1)+
                              data.vz(ix  ,iy-1,iz-1)+
                              data.vz(ix-1,iy  ,iz  )+
                              data.vz(ix-1,iy-1,iz  )+
                              data.vz(ix-1,iy  ,iz-1)+
                              data.vz(ix-1,iy-1,iz-1))*
                              0.125;
        SAMS::T_dataType  vx_n_centre=(dataNeutral.vx(ix,iy,iz)+
                              dataNeutral.vx(ix  ,iy-1,iz)+
                              dataNeutral.vx(ix  ,iy  ,iz-1)+
                              dataNeutral.vx(ix  ,iy-1,iz-1)+
                              dataNeutral.vx(ix-1,iy  ,iz  )+
                              dataNeutral.vx(ix-1,iy-1,iz  )+
                              dataNeutral.vx(ix-1,iy  ,iz-1)+
                              dataNeutral.vx(ix-1,iy-1,iz-1))*
                              0.125;
        SAMS::T_dataType  vy_n_centre=(dataNeutral.vy(ix,iy,iz)+
                              dataNeutral.vy(ix  ,iy-1,iz)+
                              dataNeutral.vy(ix  ,iy  ,iz-1)+
                              dataNeutral.vy(ix  ,iy-1,iz-1)+
                              dataNeutral.vy(ix-1,iy  ,iz  )+
                              dataNeutral.vy(ix-1,iy-1,iz  )+
                              dataNeutral.vy(ix-1,iy  ,iz-1)+
                              dataNeutral.vy(ix-1,iy-1,iz-1))*
                              0.125;
        SAMS::T_dataType  vz_n_centre=(dataNeutral.vz(ix,iy,iz)+
                              dataNeutral.vz(ix  ,iy-1,iz)+
                              dataNeutral.vz(ix  ,iy  ,iz-1)+
                              dataNeutral.vz(ix  ,iy-1,iz-1)+
                              dataNeutral.vz(ix-1,iy  ,iz  )+
                              dataNeutral.vz(ix-1,iy-1,iz  )+
                              dataNeutral.vz(ix-1,iy  ,iz-1)+
                              dataNeutral.vz(ix-1,iy-1,iz-1))*
                              0.125;
        
        //Energy source terms - the 3/2 here needs fixing
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
       
       SAMS::T_dataType dvx = vx_n_centre - vx_centre;
       SAMS::T_dataType dvy = vy_n_centre - vy_centre;
       SAMS::T_dataType dvz = vz_n_centre - vz_centre;

       SAMS::T_dataType vd2 = dvx*dvx + dvy*dvy + dvz*dvz;
       
    plasma_source.source_energy(ix,iy,iz)=plasma_source.ac(ix,iy,iz)*dataNeutral.rho(ix,iy,iz)*(\
                        0.5*vd2 \
                        + (temperature_neutral-temperature_ion)/(data.gas_gamma*(data.gas_gamma-1)));
        //printf("getting neutral energy source terms \n");
        plasma_source.source_energy_n(ix,iy,iz)=plasma_source.ac(ix,iy,iz)*data.rho(ix,iy,iz)*(\
                        0.5*vd2 \
                        - (temperature_neutral-temperature_ion)/(data.gas_gamma*(data.gas_gamma-1)));  
                                   
        //printf("%i,%i,%i \n",ix,iy,iz);
    }, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));

//printf("FINISHED LOOP \n");
  }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
//Get the source terms for the IR rates
    template<typename T_EOS>
 void PIP<T_EOS>::get_ion_rec_source_terms(LARE::LARE3DST<T_EOS>::simulationData &data, LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){	

    LARE::T_dataType T0=10000.0;//data.T_reference; //Reference temperature
    LARE::T_dataType Te_0=T0/1.1604e4;
    LARE::T_dataType kb_ev=8.617333e-5; //Kb in eV/K
    
    using Range = portableWrapper::Range;
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
    
        //Mass source terms
        plasma_source.source_mass(ix,iy,iz)  += plasma_source.gm_ion(ix,iy,iz)*dataNeutral.rho(ix,iy,iz)-plasma_source.gm_rec(ix,iy,iz)*data.rho(ix,iy,iz);
        plasma_source.source_mass_n(ix,iy,iz) +=-plasma_source.gm_ion(ix,iy,iz)*dataNeutral.rho(ix,iy,iz)+plasma_source.gm_rec(ix,iy,iz)*data.rho(ix,iy,iz);
        
        SAMS::T_dataType  rho_plasma_vertex=  (data.rho(ix  , iy  , iz  ) + 
                                        data.rho(ix+1, iy  , iz  ) + 
                                        data.rho(ix  , iy+1, iz  ) + 
                                        data.rho(ix+1, iy+1, iz  ) + 
                                        data.rho(ix  , iy  , iz+1) + 
                                        data.rho(ix+1, iy  , iz+1) + 
                                        data.rho(ix  , iy+1, iz+1) + 
                                        data.rho(ix+1, iy+1, iz+1))* 
                                        0.125;
        SAMS::T_dataType  Gm_ion_vertex=      (plasma_source.gm_ion(ix  , iy  , iz  ) + 
                                        plasma_source.gm_ion(ix+1, iy  , iz  ) + 
                                        plasma_source.gm_ion(ix  , iy+1, iz  ) + 
                                        plasma_source.gm_ion(ix+1, iy+1, iz  ) + 
                                        plasma_source.gm_ion(ix  , iy  , iz+1) + 
                                        plasma_source.gm_ion(ix+1, iy  , iz+1) + 
                                        plasma_source.gm_ion(ix  , iy+1, iz+1) + 
                                        plasma_source.gm_ion(ix+1, iy+1, iz+1))* 
                                        0.125;
        SAMS::T_dataType  Gm_rec_vertex=      (plasma_source.gm_rec(ix  , iy  , iz  ) + 
                                        plasma_source.gm_rec(ix+1, iy  , iz  ) + 
                                        plasma_source.gm_rec(ix  , iy+1, iz  ) + 
                                        plasma_source.gm_rec(ix+1, iy+1, iz  ) + 
                                        plasma_source.gm_rec(ix  , iy  , iz+1) + 
                                        plasma_source.gm_rec(ix+1, iy  , iz+1) + 
                                        plasma_source.gm_rec(ix  , iy+1, iz+1) + 
                                        plasma_source.gm_rec(ix+1, iy+1, iz+1))* 
                                        0.125;
        SAMS::T_dataType  rho_neutral_vertex=  (dataNeutral.rho(ix  , iy  , iz  ) + 
                                         dataNeutral.rho(ix+1, iy  , iz  ) + 
                                         dataNeutral.rho(ix  , iy+1, iz  ) + 
                                         dataNeutral.rho(ix+1, iy+1, iz  ) + 
                                         dataNeutral.rho(ix  , iy  , iz+1) + 
                                         dataNeutral.rho(ix+1, iy  , iz+1) + 
                                         dataNeutral.rho(ix  , iy+1, iz+1) + 
                                         dataNeutral.rho(ix+1, iy+1, iz+1))* 
                                         0.125;
        
        //Velocity source terms
        SAMS::T_dataType  v_D_x  =  data.vx(ix,iy,iz) - dataNeutral.vx(ix,iy,iz); //Drift velocity in the x-direction
        SAMS::T_dataType  v_D_y  =  data.vy(ix,iy,iz) - dataNeutral.vy(ix,iy,iz); //Drift velocity in the y-direction
        SAMS::T_dataType  v_D_z  =  data.vz(ix,iy,iz) - dataNeutral.vz(ix,iy,iz); //Drift velocity in the z-direction
        plasma_source.source_v_x(ix,iy,iz) -= Gm_ion_vertex*rho_neutral_vertex*v_D_x/rho_plasma_vertex;
        plasma_source.source_v_y(ix,iy,iz) -= Gm_ion_vertex*rho_neutral_vertex*v_D_y/rho_plasma_vertex;
        plasma_source.source_v_z(ix,iy,iz) -= Gm_ion_vertex*rho_neutral_vertex*v_D_z/rho_plasma_vertex;
        plasma_source.source_v_x_n(ix,iy,iz) += Gm_rec_vertex*rho_plasma_vertex*v_D_x/rho_neutral_vertex;
        plasma_source.source_v_y_n(ix,iy,iz) += Gm_rec_vertex*rho_plasma_vertex*v_D_y/rho_neutral_vertex;
        plasma_source.source_v_z_n(ix,iy,iz) += Gm_rec_vertex*rho_plasma_vertex*v_D_z/rho_neutral_vertex;
        
        
        //Get velocity at cell centres
        SAMS::T_dataType  vx_centre=  (data.vx(ix  , iy  , iz  ) + 
                                        data.vx(ix-1, iy  , iz  ) + 
                                        data.vx(ix  , iy-1, iz  ) + 
                                        data.vx(ix-1, iy-1, iz  ) + 
                                        data.vx(ix  , iy  , iz-1) + 
                                        data.vx(ix-1, iy  , iz-1) + 
                                        data.vx(ix  , iy-1, iz-1) + 
                                        data.vx(ix-1, iy-1, iz-1))* 
                                        0.125;
        SAMS::T_dataType  vy_centre=  (data.vy(ix  , iy  , iz  ) + 
                                        data.vy(ix-1, iy  , iz  ) + 
                                        data.vy(ix  , iy-1, iz  ) + 
                                        data.vy(ix-1, iy-1, iz  ) + 
                                        data.vy(ix  , iy  , iz-1) + 
                                        data.vy(ix-1, iy  , iz-1) + 
                                        data.vy(ix  , iy-1, iz-1) + 
                                        data.vy(ix-1, iy-1, iz-1))* 
                                        0.125;
        SAMS::T_dataType  vz_centre=  (data.vz(ix  , iy  , iz  ) + 
                                        data.vz(ix-1, iy  , iz  ) + 
                                        data.vz(ix  , iy-1, iz  ) + 
                                        data.vz(ix-1, iy-1, iz  ) + 
                                        data.vz(ix  , iy  , iz-1) + 
                                        data.vz(ix-1, iy  , iz-1) + 
                                        data.vz(ix  , iy-1, iz-1) + 
                                        data.vz(ix-1, iy-1, iz-1))* 
                                        0.125;
        SAMS::T_dataType  vx_n_centre= (dataNeutral.vx(ix  , iy  , iz  ) + 
                                        dataNeutral.vx(ix-1, iy  , iz  ) + 
                                        dataNeutral.vx(ix  , iy-1, iz  ) + 
                                        dataNeutral.vx(ix-1, iy-1, iz  ) + 
                                        dataNeutral.vx(ix  , iy  , iz-1) + 
                                        dataNeutral.vx(ix-1, iy  , iz-1) + 
                                        dataNeutral.vx(ix  , iy-1, iz-1) + 
                                        dataNeutral.vx(ix-1, iy-1, iz-1))* 
                                        0.125;
        SAMS::T_dataType  vy_n_centre= (dataNeutral.vy(ix  , iy  , iz  ) + 
                                        dataNeutral.vy(ix-1, iy  , iz  ) + 
                                        dataNeutral.vy(ix  , iy-1, iz  ) + 
                                        dataNeutral.vy(ix-1, iy-1, iz  ) + 
                                        dataNeutral.vy(ix  , iy  , iz-1) + 
                                        dataNeutral.vy(ix-1, iy  , iz-1) + 
                                        dataNeutral.vy(ix  , iy-1, iz-1) + 
                                        dataNeutral.vy(ix-1, iy-1, iz-1))* 
                                        0.125;
        SAMS::T_dataType  vz_n_centre= (dataNeutral.vz(ix  , iy  , iz  ) + 
                                        dataNeutral.vz(ix-1, iy  , iz  ) + 
                                        dataNeutral.vz(ix  , iy-1, iz  ) + 
                                        dataNeutral.vz(ix-1, iy-1, iz  ) + 
                                        dataNeutral.vz(ix  , iy  , iz-1) + 
                                        dataNeutral.vz(ix-1, iy  , iz-1) + 
                                        dataNeutral.vz(ix  , iy-1, iz-1) + 
                                        dataNeutral.vz(ix-1, iy-1, iz-1))* 
                                        0.125;
        
        //SAMS::T_dataType  vp2=v_x_plasma_centre*v_x_plasma_centre+
        //                      v_y_plasma_centre*v_y_plasma_centre+
        //                      v_z_plasma_centre*v_z_plasma_centre;
                              
        //SAMS::T_dataType  vn2=v_x_neutral_centre*v_x_neutral_centre+
        //                      v_y_neutral_centre*v_y_neutral_centre+
        //                      v_z_neutral_centre*v_z_neutral_centre;
        
        //SAMS::T_dataType  vpvn= v_x_plasma_centre*v_x_neutral_centre+
        //                        v_y_plasma_centre*v_y_neutral_centre+
        //                        v_z_plasma_centre*v_z_neutral_centre;   
        
        SAMS::T_dataType dvx = vx_n_centre - vx_centre;
        SAMS::T_dataType dvy = vy_n_centre - vy_centre;
        SAMS::T_dataType dvz = vz_n_centre - vz_centre;

        SAMS::T_dataType vd2 = dvx*dvx + dvy*dvy + dvz*dvz;
       
        plasma_source.source_energy(ix,iy,iz)+=plasma_source.gm_ion(ix,iy,iz)*dataNeutral.rho(ix,iy,iz)*\
                        0.5*vd2 \
                        /data.rho(ix,iy,iz);
        //printf("getting neutral energy source terms \n");
        plasma_source.source_energy_n(ix,iy,iz)+=plasma_source.gm_rec(ix,iy,iz)*data.rho(ix,iy,iz)*\
                        0.5*vd2\
                        /dataNeutral.rho(ix,iy,iz);
        
        // Thermal energy exchanged by ionisation/recombination (Snow & Hillier 2021, eqs. 3 and 7),
        // written for the SPECIFIC internal energy eps that SAMS evolves (P = (gamma-1) rho eps).
        //
        // The paper's terms are rates of change of internal energy DENSITY rho*eps. Since
        //     d(rho eps)/dt = rho deps/dt + eps drho/dt,
        // the specific energy changes by  deps/dt = [d(rho eps)/dt - eps drho/dt] / rho.
        // The second ("dilution") term accounts for the mass gained or lost by each species:
        //   - ionisation moves mass from neutrals to plasma carrying the neutral specific energy eps_n,
        //     so the plasma relaxes towards eps_n and the neutral specific energy is unchanged;
        //   - recombination removes plasma mass but only the proton half of its thermal energy (the
        //     electron's thermal energy stays in the plasma), so the plasma specific energy rises and
        //     the neutrals relax towards eps_p/2.
        // The two terms sum to zero in internal energy density (a pure transfer between species), and
        // both vanish in ionisation equilibrium (Gamma_ion rho_n = Gamma_rec rho_p with T_n = T_p).
        //
        // This replaces the earlier TeIR form, which added (Gamma_ion P_n - Gamma_rec P_p/2)/(gamma-1)
        // with the same sign to both species and omitted the dilution terms, so total energy was not
        // conserved away from ionisation equilibrium.
        const SAMS::T_dataType eps_p = data.energy_ion(ix,iy,iz);
        const SAMS::T_dataType eps_n = dataNeutral.energy(ix,iy,iz);
        const SAMS::T_dataType ion_flux = plasma_source.gm_ion(ix,iy,iz)*dataNeutral.rho(ix,iy,iz); // mass rate neutral -> plasma
        const SAMS::T_dataType rec_flux = plasma_source.gm_rec(ix,iy,iz)*data.rho(ix,iy,iz);        // mass rate plasma -> neutral

        plasma_source.source_energy(ix,iy,iz)   += (ion_flux*(eps_n - eps_p) + 0.5*rec_flux*eps_p)/data.rho(ix,iy,iz);
        plasma_source.source_energy_n(ix,iy,iz) += rec_flux*(0.5*eps_p - eps_n)/dataNeutral.rho(ix,iy,iz);

        //Work out how much energy is spent/gained by IR processes
//        if (plasma_source.ion_rec_empirical) { 
//            //printf("ionisation energy, rho = %f %f \n",ionisation_energy, dataNeutral.rho(ix,iy,iz));
//            plasma_source.source_energy(ix,iy,iz)+=(plasma_source.ion_heating(ix,iy,iz)+plasma_source.ion_loss(ix,iy,iz))/data.rho(ix,iy,iz);//factor of pho comes from denergy density being specified
//        }
        
    }, Range(0,data.nx), Range(0,data.ny), Range(0,data.nz));

 }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
//Collisional timestep calculation
//Assuming normalisation to the sound speed
    template<typename T_EOS> 
void PIP<T_EOS>::set_dt_collisional(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source) {

    using Range = portableWrapper::Range;

    int i0 = data.geometry == LARE::geometryType::Cartesian ? 0:1;
    //Now need to do a map and reduction
    plasma_source.two_fluid_timestep = 0.1*data.dt_multiplier * 
    portableWrapper::applyReduction(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        //Get Temperatures
        
        SAMS::T_dataType  t1 = std::min({1.0/std::abs(plasma_source.source_mass(ix,iy,iz)),
                                       1.0/std::abs(plasma_source.source_mass_n(ix,iy,iz)),
                                       1.0/std::abs(plasma_source.source_v_x(ix,iy,iz)),
                                       1.0/std::abs(plasma_source.source_v_x_n(ix,iy,iz)),
                                       1.0/std::abs(plasma_source.source_v_y(ix,iy,iz)),
                                       1.0/std::abs(plasma_source.source_v_y_n(ix,iy,iz)),
                                       1.0/std::abs(plasma_source.source_v_z(ix,iy,iz)),
                                       1.0/std::abs(plasma_source.source_v_z_n(ix,iy,iz)),
                                       1.0/std::abs(plasma_source.source_energy(ix,iy,iz)),
                                       1.0/std::abs(plasma_source.source_energy_n(ix,iy,iz))
                                       });
        
                return t1;
    }, LAMBDA(SAMS::T_dataType  &a, const SAMS::T_dataType  &b) {
        a=portableWrapper::min(a, b);
    }, data.largest_number,
    Range(i0, data.nx), Range(0, data.ny), Range(0, data.nz));

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T_EOS> 
void PIP<T_EOS>::get_equilibrium_ion_fraction(SAMS::T_dataType  T0,SAMS::T_dataType  &xi_n) {
    SAMS::T_dataType  Te_0=T0/1.1604e4;
    SAMS::T_dataType  ioneq=(2.6e-19/std::sqrt(Te_0))/(2.91e-14/(0.232+13.6/Te_0)*std::pow(13.6/Te_0,0.39)*std::exp(-13.6/Te_0));
    xi_n=ioneq/(ioneq+1.0);
  }
  
////////////////////////////////////////////////////////////////////////////////////////
static LARE::T_indexType find_logT_bracket(
    const data_two_fluid_source &ps,
    LARE::T_dataType logT);
static LARE::T_dataType loglinear_interp(LARE::T_dataType v0, LARE::T_dataType v1, LARE::T_dataType t);
/////////////////////////////////////////////////////////////////////////////////////
// Interpolation and lookup functions for the rates ...                            //
//                                                                                 //
// These functions are marked DEVICEPREFIX INLINE because they are called          //
// from inside LAMBDA kernels that execute on the GPU (via applyKernel).           //
// DEVICEPREFIX (expands to __device__ __host__ for CUDA/HIP, KOKKOS_FUNCTION      //
// for Kokkos, or empty for CPU-only) instructs the compiler to generate both      //
// a host and a device version of each function so they can be called from         //
// device code. INLINE (expands to always_inline or KOKKOS_FORCEINLINE_FUNCTION)   //
// forces the compiler to inline the function body at every call site, which       //
// avoids GPU function-call overhead inside kernels and enables the compiler       //
// to optimise across the call boundary (e.g. eliminating redundant log10/bracket  //
// evaluations when multiple rate functions are called with the same temperature). //
/////////////////////////////////////////////////////////////////////////////////////
DEVICEPREFIX INLINE LARE::T_dataType interpolate_collisional_excitation(
    const data_two_fluid_source &ps, LARE::T_dataType temperature,
    LARE::T_indexType lower_level_num, LARE::T_indexType upper_level_num)
{
    const LARE::T_indexType li = lower_level_num - static_cast<LARE::T_indexType>(ps.level_offset);
    const LARE::T_indexType ui = upper_level_num - static_cast<LARE::T_indexType>(ps.level_offset);
    if (li < 0 || ui < 0 || li >= ps.collisional_excitation_rates.getSize(1) ||
        ui >= ps.collisional_excitation_rates.getSize(2)) return 0.0;
    const LARE::T_dataType logT = std::log10(temperature);
    const LARE::T_indexType i0 = find_logT_bracket(ps, logT);
    const LARE::T_indexType i1 = i0 + 1;
    const LARE::T_dataType t = (logT - ps.grid_logT(i0)) / (ps.grid_logT(i1) - ps.grid_logT(i0));
    return loglinear_interp(ps.collisional_excitation_rates(i0, li, ui),
                            ps.collisional_excitation_rates(i1, li, ui), t);
}

DEVICEPREFIX INLINE LARE::T_dataType interpolate_collisional_ionisation(
    const data_two_fluid_source &ps, LARE::T_dataType temperature,
    LARE::T_indexType level_num)
{
//if (temperature==10000.0){printf("ION DEBUG: lower=%d offset=%d li=%d size=%d\n",
//       level_num,
//       ps.level_offset,
//       level_num - ps.level_offset,
//       ps.collisional_ionisation_rates.getSize(1));}
       
    const LARE::T_indexType li = level_num - static_cast<LARE::T_indexType>(ps.level_offset);
    if (li < 0 || li >= ps.collisional_ionisation_rates.getSize(1)) return 0.0;
    const LARE::T_dataType logT = std::log10(temperature);
    const LARE::T_indexType i0 = find_logT_bracket(ps, logT);
    const LARE::T_indexType i1 = i0 + 1;
    const LARE::T_dataType t = (logT - ps.grid_logT(i0)) / (ps.grid_logT(i1) - ps.grid_logT(i0));
    return loglinear_interp(ps.collisional_ionisation_rates(i0, li),
                            ps.collisional_ionisation_rates(i1, li), t);
}

DEVICEPREFIX INLINE LARE::T_dataType interpolate_radiative_recombination(
    const data_two_fluid_source &ps, LARE::T_dataType temperature,
    LARE::T_indexType level_num)
{
    const LARE::T_indexType li = level_num - static_cast<LARE::T_indexType>(ps.level_offset);
    if (li < 0 || li >= ps.radiative_recombination_rates.getSize(1)) return 0.0;
    const LARE::T_dataType logT = std::log10(temperature);
    const LARE::T_indexType i0 = find_logT_bracket(ps, logT);
    const LARE::T_indexType i1 = i0 + 1;
    const LARE::T_dataType t = (logT - ps.grid_logT(i0)) / (ps.grid_logT(i1) - ps.grid_logT(i0));
    return loglinear_interp(ps.radiative_recombination_rates(i0, li),
                            ps.radiative_recombination_rates(i1, li), t);
}

DEVICEPREFIX INLINE LARE::T_dataType get_radiative_excitation(
    const data_two_fluid_source &ps,
    LARE::T_indexType lower_level_num, LARE::T_indexType upper_level_num)
{
    const LARE::T_indexType li = lower_level_num - static_cast<LARE::T_indexType>(ps.level_offset);
    const LARE::T_indexType ui = upper_level_num - static_cast<LARE::T_indexType>(ps.level_offset);
    if (li < 0 || ui < 0 || li >= ps.radiative_excitation_rates.getSize(0) ||
        ui >= ps.radiative_excitation_rates.getSize(1)) return 0.0;
    return ps.radiative_excitation_rates(li, ui);
}

DEVICEPREFIX INLINE LARE::T_dataType get_radiative_de_excitation(
    const data_two_fluid_source &ps,
    LARE::T_indexType lower_level_num, LARE::T_indexType upper_level_num)
{
    const LARE::T_indexType li = lower_level_num - static_cast<LARE::T_indexType>(ps.level_offset);
    const LARE::T_indexType ui = upper_level_num - static_cast<LARE::T_indexType>(ps.level_offset);
    if (li < 0 || ui < 0 || li >= ps.radiative_de_excitation_rates.getSize(0) ||
        ui >= ps.radiative_de_excitation_rates.getSize(1)) return 0.0;
    return ps.radiative_de_excitation_rates(li, ui);
}

DEVICEPREFIX INLINE LARE::T_dataType get_radiative_ionisation(
    const data_two_fluid_source &ps,
    LARE::T_indexType level_num)
{
    const LARE::T_indexType li = level_num - static_cast<LARE::T_indexType>(ps.level_offset);
    if (li < 0 || li >= ps.radiative_ionisation_rates.getSize(0)) return 0.0;
    return ps.radiative_ionisation_rates(li);
}
////////////////////////////////////////////////////////////////////////////
// Code below relates to handling and reading of offline atomic rate data //
////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////
// Routine for the reading the rates //
///////////////////////////////////////
template<typename T_EOS> 
void PIP<T_EOS>::two_fluid_read_rates(data_two_fluid_source &plasma_source){

    int ncid = -1;
    std::string data_path=plasma_source.data_path;
    int nc_status = nc_open(data_path.c_str(), NC_NOWRITE, &ncid);
    if (nc_status != NC_NOERR) {
        fprintf(stderr, "two_fluid_read_rates: nc_open failed for '%s': %s\n",
                data_path.c_str(), nc_strerror(nc_status));
        return;
    }
    fprintf(stdout, "two_fluid_read_rates: using file '%s'\n", data_path.c_str());

    // Read dimensions
    int dim_tsamp = -1, dim_lower = -1, dim_upper = -1;
    size_t n_tsamp = 0, n_lower = 0, n_upper = 0;

    auto check = [&](int status, const char *msg) -> bool {
        if (status != NC_NOERR) {
            fprintf(stderr, "two_fluid_read_rates: %s: %s\n", msg, nc_strerror(status));
            return false;
        }
        return true;
    };

    if (!check(nc_inq_dimid(ncid, "n_tsample",     &dim_tsamp), "missing dim 'n_tsample'"))    { nc_close(ncid); return; }
    if (!check(nc_inq_dimid(ncid, "n_lower_level", &dim_lower), "missing dim 'n_lower_level'")) { nc_close(ncid); return; }
    if (!check(nc_inq_dimid(ncid, "n_upper_level", &dim_upper), "missing dim 'n_upper_level'")) { nc_close(ncid); return; }

    if (!check(nc_inq_dimlen(ncid, dim_tsamp, &n_tsamp), "dimlen 'n_tsample'"))     { nc_close(ncid); return; }
    if (!check(nc_inq_dimlen(ncid, dim_lower, &n_lower), "dimlen 'n_lower_level'")) { nc_close(ncid); return; }
    if (!check(nc_inq_dimlen(ncid, dim_upper, &n_upper), "dimlen 'n_upper_level'")) { nc_close(ncid); return; }

    if (n_tsamp == 0 || n_lower == 0 || n_upper == 0) {
        fprintf(stderr, "two_fluid_read_rates: invalid dimensions (%zu, %zu, %zu)\n", n_tsamp, n_lower, n_upper);
        nc_close(ncid); return;
    }

    // Read level mapping arrays
    int var_lower_lvl = -1, var_upper_lvl = -1;
    if (!check(nc_inq_varid(ncid, "lower_level", &var_lower_lvl), "missing var 'lower_level'")) { nc_close(ncid); return; }
    if (!check(nc_inq_varid(ncid, "upper_level", &var_upper_lvl), "missing var 'upper_level'")) { nc_close(ncid); return; }

    plasma_source.lower_level_map.resize(n_lower);
    plasma_source.upper_level_map.resize(n_upper);
    if (!check(nc_get_var_int(ncid, var_lower_lvl, plasma_source.lower_level_map.data()), "read 'lower_level'")) { nc_close(ncid); return; }
    if (!check(nc_get_var_int(ncid, var_upper_lvl, plasma_source.upper_level_map.data()), "read 'upper_level'")) { nc_close(ncid); return; }
    plasma_source.level_offset = plasma_source.lower_level_map[0];

    // Allocate portable arrays 
    using Range = pw::Range;
    Range T_range (0, static_cast<LARE::T_indexType>(n_tsamp - 1));
    Range lo_range(0, static_cast<LARE::T_indexType>(n_lower - 1));
    Range up_range(0, static_cast<LARE::T_indexType>(n_upper - 1));

    static pw::portableArrayManager svManager;
    
    svManager.allocate(plasma_source.grid_logT,                     T_range);
    svManager.allocate(plasma_source.collisional_excitation_rates,  T_range, lo_range, up_range);
    svManager.allocate(plasma_source.collisional_ionisation_rates,  T_range, lo_range);
    svManager.allocate(plasma_source.radiative_recombination_rates, T_range, lo_range);
    svManager.allocate(plasma_source.radiative_excitation_rates,    lo_range, up_range);
    svManager.allocate(plasma_source.radiative_de_excitation_rates, lo_range, up_range);
    svManager.allocate(plasma_source.radiative_ionisation_rates,    lo_range);
//printf("manager address = %p\n", &svManager);

    // Read logT
    int var_logT = -1;
    if (!check(nc_inq_varid(ncid, "logT", &var_logT), "missing var 'logT'")) { nc_close(ncid); return; }
    if (!check(nc_get_var_double(ncid, var_logT, plasma_source.grid_logT.data()), "read 'logT'")) { nc_close(ncid); return; }

    // Helpers to read variables into flat buffers then copy into arrays
    auto read3D = [&](const char *name, LARE::hostVolumeArray &arr,
                      size_t s0, size_t s1, size_t s2) -> bool {
        int varid = -1;
        if (!check(nc_inq_varid(ncid, name, &varid), name)) return false;
        std::vector<double> flat(s0 * s1 * s2);
        if (!check(nc_get_var_double(ncid, varid, flat.data()), name)) return false;
        for (size_t i = 0; i < s0; ++i)
            for (size_t j = 0; j < s1; ++j)
                for (size_t k = 0; k < s2; ++k)
                    arr(i, j, k) = flat[(i * s1 + j) * s2 + k];
        return true;
    };
    auto read2D = [&](const char *name, LARE::hostPlaneArray &arr,
                      size_t s0, size_t s1) -> bool {
        int varid = -1;
        if (!check(nc_inq_varid(ncid, name, &varid), name)) return false;
        std::vector<double> flat(s0 * s1);
        if (!check(nc_get_var_double(ncid, varid, flat.data()), name)) return false;
        for (size_t i = 0; i < s0; ++i)
            for (size_t j = 0; j < s1; ++j)
                arr(i, j) = flat[i * s1 + j];
        return true;
    };
    auto read1D = [&](const char *name, LARE::hostLineArray &arr, size_t s0) -> bool {
        int varid = -1;
        if (!check(nc_inq_varid(ncid, name, &varid), name)) return false;
        std::vector<double> flat(s0);
        if (!check(nc_get_var_double(ncid, varid, flat.data()), name)) return false;
        for (size_t i = 0; i < s0; ++i)
            arr(i) = flat[i];
        return true;
    };

    if (!read3D("collisional_excitation_rates",  plasma_source.collisional_excitation_rates,  n_tsamp, n_lower, n_upper)) { nc_close(ncid); return; }
    if (!read2D("collisional_ionisation_rates",  plasma_source.collisional_ionisation_rates,  n_tsamp, n_lower)) { nc_close(ncid); return; }
    if (!read2D("radiative_recombination_rates", plasma_source.radiative_recombination_rates, n_tsamp, n_lower)) { nc_close(ncid); return; }
    if (!read2D("radiative_excitation_rates",    plasma_source.radiative_excitation_rates,    n_lower, n_upper)) { nc_close(ncid); return; }
    if (!read2D("radiative_de_excitation_rates", plasma_source.radiative_de_excitation_rates, n_lower, n_upper)) { nc_close(ncid); return; }
    if (!read1D("radiative_ionisation_rates",    plasma_source.radiative_ionisation_rates,    n_lower)) { nc_close(ncid); return; }

    nc_close(ncid);
    fprintf(stdout, "Rates read successfully. n_tsample=%zu, n_lower=%zu, n_upper=%zu, level_offset=%i\n",
            n_tsamp, n_lower, n_upper, plasma_source.level_offset);

    return;
}

template<typename T_EOS> 
void PIP<T_EOS>::two_fluid_test_rates(const data_two_fluid_source &plasma_source)
{
    constexpr LARE::T_dataType  logT_test    = 4.0;
    constexpr LARE::T_indexType lower_level  = 1;
    constexpr LARE::T_indexType upper_level  = 5;

    const LARE::T_dataType T_test = std::pow(10.0, logT_test);
    
//    printf("grid size = %ld\n", plasma_source.grid_logT.getSize(0));
//    printf("rate size = %ld\n", plasma_source.collisional_ionisation_rates.getSize(0));
//    printf("first rate = %e\n",
//       plasma_source.collisional_ionisation_rates(0,0));

    fprintf(stdout, "\n--- two_fluid_test_rates: logT=%.1f, lower=%d, upper=%d ---\n",
            logT_test, static_cast<int>(lower_level), static_cast<int>(upper_level));
    fprintf(stdout, "  collisional_excitation   (%d->%d) : %e\n",
            static_cast<int>(lower_level), static_cast<int>(upper_level),
            interpolate_collisional_excitation(plasma_source, T_test, lower_level, upper_level));
    fprintf(stdout, "  radiative_excitation     (%d->%d) : %e\n",
            static_cast<int>(lower_level), static_cast<int>(upper_level),
            get_radiative_excitation(plasma_source, lower_level, upper_level));
    fprintf(stdout, "  radiative_de_excitation  (%d->%d) : %e\n",
            static_cast<int>(upper_level), static_cast<int>(lower_level),
            get_radiative_de_excitation(plasma_source, lower_level, upper_level));
    fprintf(stdout, "  collisional_ionisation   (level=%d) : %e\n",
            static_cast<int>(lower_level),
            interpolate_collisional_ionisation(plasma_source, T_test, lower_level));
    fprintf(stdout, "  collisional_ionisation   (level=%d) : %e\n",
            static_cast<int>(upper_level),
            interpolate_collisional_ionisation(plasma_source, T_test, upper_level));
    fprintf(stdout, "  radiative_recombination  (level=%d) : %e\n",
            static_cast<int>(lower_level),
            interpolate_radiative_recombination(plasma_source, T_test, lower_level));
    fprintf(stdout, "  radiative_recombination  (level=%d) : %e\n",
            static_cast<int>(upper_level),
            interpolate_radiative_recombination(plasma_source, T_test, upper_level));
    fprintf(stdout, "  radiative_ionisation     (level=%d) : %e\n",
            static_cast<int>(lower_level),
            get_radiative_ionisation(plasma_source, lower_level));
    fprintf(stdout, "  radiative_ionisation     (level=%d) : %e\n",
            static_cast<int>(upper_level),
            get_radiative_ionisation(plasma_source, upper_level));
    fprintf(stdout, "---\n\n");
    
    /*printf("plasma_source = %p\n", (void *)&plasma_source);
    
                    printf("CHECK rate[0,0] = %e\n",
       plasma_source.collisional_ionisation_rates(0,0));

printf("CHECK rate[50,0] = %e\n",
       plasma_source.collisional_ionisation_rates(50,0));

printf("CHECK rate[50,4] = %e\n",
       plasma_source.collisional_ionisation_rates(50,4));
       printf("dim0=%ld dim1=%ld\n",
       plasma_source.collisional_ionisation_rates.getSize(0),
       plasma_source.collisional_ionisation_rates.getSize(1));
    */

}

//////////////////////////////////////////////////////////////////////////////////
// Shared internal helpers for logT bracket search and log-linear interpolation //
//////////////////////////////////////////////////////////////////////////////////

static LARE::T_indexType find_logT_bracket(const data_two_fluid_source &ps, LARE::T_dataType logT)
{
    const LARE::T_indexType lb = ps.grid_logT.getLowerBound(0);
    const LARE::T_indexType ub = ps.grid_logT.getUpperBound(0);
    if (logT <= ps.grid_logT(lb)) return lb;
    if (logT >= ps.grid_logT(ub)) return ub - 1;
    LARE::T_indexType i0 = lb;
    for (LARE::T_indexType i = lb; i < ub; ++i) {
        if (ps.grid_logT(i) <= logT && logT < ps.grid_logT(i + 1)) { i0 = i; break; }
    }
    return i0;
}

static LARE::T_dataType loglinear_interp(LARE::T_dataType v0, LARE::T_dataType v1, LARE::T_dataType t)
{
    if (v0 <= 0.0 && v1 <= 0.0) return 0.0;
    if (v0 <= 0.0) return v1;
    if (v1 <= 0.0) return v0;
    return std::pow(10.0, std::log10(v0) + (std::log10(v1) - std::log10(v0)) * t);
}

///////////////////////////////////////////////////////////////////////////////////////////////
    template<typename T_EOS>
    void TWOFLUID::PIP<T_EOS>::copyState(
        LARE::LARE3DST<T_EOS>::simulationData &source,
        LARE::LARE3DNF<T_EOS>::simulationData &source_n,
        oldData &dest)
    {
       printf("%p %p\n",
       source.vx.data(),
       dest.vx.data());
        for (int ix=0; ix<=source.nx; ix++)
        {
            for (int iy=0; iy<=source.ny; iy++)
            {
                for (int iz=0; iz<=source.nz; iz++)
                {
                    dest.vx(ix,iy,iz)=source.vx(ix,iy,iz);
                    dest.vy(ix,iy,iz)=source.vy(ix,iy,iz);
                    dest.vz(ix,iy,iz)=source.vz(ix,iy,iz);
                    
                    dest.vx_n(ix,iy,iz)=source_n.vx(ix,iy,iz);
                    dest.vy_n(ix,iy,iz)=source_n.vy(ix,iy,iz);
                    dest.vz_n(ix,iy,iz)=source_n.vz(ix,iy,iz);
                }
            }
        }

        for (int ix=0; ix<source.nx; ix++)
        {
            for (int iy=0; iy<source.ny; iy++)
            {
                for (int iz=0; iz<source.nz; iz++)
                {
                    dest.rho(ix,iy,iz)=source.rho(ix,iy,iz);
                    dest.energy(ix,iy,iz)=source.energy_ion(ix,iy,iz);
                    
                    dest.rho_n(ix,iy,iz)=source_n.rho(ix,iy,iz);
                    dest.energy_n(ix,iy,iz)=source_n.energy(ix,iy,iz);
                }
            }
        }
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T_EOS>
void PIP<T_EOS>::checkConservation(
        LARE::LARE3DST<T_EOS>::simulationData &data,
        LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral,
        oldData &dataOld)
{

    using Range = portableWrapper::Range;


    const int nx = data.nx;
    const int ny = data.ny;
    const int nz = data.nz;


    int ncell = (nx+1)*(ny+1)*(nz+1);



    struct conservationBuffers
    {
        SAMS::T_dataType *mass_error;
        SAMS::T_dataType *momentum_x_error;
        SAMS::T_dataType *momentum_y_error;
        SAMS::T_dataType *momentum_z_error;
        SAMS::T_dataType *energy_error;

        SAMS::T_dataType *thermal_energy_error;
        SAMS::T_dataType *kinetic_energy_error;
    };


    conservationBuffers buffers;


    buffers.mass_error =
        new SAMS::T_dataType[ncell];

    buffers.momentum_x_error =
        new SAMS::T_dataType[ncell];

    buffers.momentum_y_error =
        new SAMS::T_dataType[ncell];

    buffers.momentum_z_error =
        new SAMS::T_dataType[ncell];

    buffers.energy_error =
        new SAMS::T_dataType[ncell];

    buffers.thermal_energy_error =
        new SAMS::T_dataType[ncell];

    buffers.kinetic_energy_error =
        new SAMS::T_dataType[ncell];



    portableWrapper::applyKernel(
    LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz)
    {

        int index =
              ix
            + (nx+1)*(iy
            + (ny+1)*iz);



        //
        // Density
        //
        SAMS::T_dataType drho_p =
            data.rho(ix,iy,iz)
            -
            dataOld.rho(ix,iy,iz);


        SAMS::T_dataType drho_n =
            dataNeutral.rho(ix,iy,iz)
            -
            dataOld.rho_n(ix,iy,iz);



        //
        // Momentum
        //
        SAMS::T_dataType dp_x =
            data.rho(ix,iy,iz)
            *
            data.vx(ix,iy,iz)
            -
            dataOld.rho(ix,iy,iz)
            *
            dataOld.vx(ix,iy,iz);


        SAMS::T_dataType dn_x =
            dataNeutral.rho(ix,iy,iz)
            *
            dataNeutral.vx(ix,iy,iz)
            -
            dataOld.rho_n(ix,iy,iz)
            *
            dataOld.vx_n(ix,iy,iz);



        SAMS::T_dataType dp_y =
            data.rho(ix,iy,iz)
            *
            data.vy(ix,iy,iz)
            -
            dataOld.rho(ix,iy,iz)
            *
            dataOld.vy(ix,iy,iz);


        SAMS::T_dataType dn_y =
            dataNeutral.rho(ix,iy,iz)
            *
            dataNeutral.vy(ix,iy,iz)
            -
            dataOld.rho_n(ix,iy,iz)
            *
            dataOld.vy_n(ix,iy,iz);



        SAMS::T_dataType dp_z =
            data.rho(ix,iy,iz)
            *
            data.vz(ix,iy,iz)
            -
            dataOld.rho(ix,iy,iz)
            *
            dataOld.vz(ix,iy,iz);


        SAMS::T_dataType dn_z =
            dataNeutral.rho(ix,iy,iz)
            *
            dataNeutral.vz(ix,iy,iz)
            -
            dataOld.rho_n(ix,iy,iz)
            *
            dataOld.vz_n(ix,iy,iz);



        //
        // Thermal energy
        //
        SAMS::T_dataType thermal_p =
            data.rho(ix,iy,iz)
            *
            data.energy_ion(ix,iy,iz)
            -
            dataOld.rho(ix,iy,iz)
            *
            dataOld.energy(ix,iy,iz);


        SAMS::T_dataType thermal_n =
            dataNeutral.rho(ix,iy,iz)
            *
            dataNeutral.energy(ix,iy,iz)
            -
            dataOld.rho_n(ix,iy,iz)
            *
            dataOld.energy_n(ix,iy,iz);


        SAMS::T_dataType thermal_error =
            thermal_p + thermal_n;



        //
        // Kinetic energy
        //
        SAMS::T_dataType v2_p =
              data.vx(ix,iy,iz)*data.vx(ix,iy,iz)
            + data.vy(ix,iy,iz)*data.vy(ix,iy,iz)
            + data.vz(ix,iy,iz)*data.vz(ix,iy,iz);


        SAMS::T_dataType v2_p_old =
              dataOld.vx(ix,iy,iz)*dataOld.vx(ix,iy,iz)
            + dataOld.vy(ix,iy,iz)*dataOld.vy(ix,iy,iz)
            + dataOld.vz(ix,iy,iz)*dataOld.vz(ix,iy,iz);



        SAMS::T_dataType v2_n =
              dataNeutral.vx(ix,iy,iz)*dataNeutral.vx(ix,iy,iz)
            + dataNeutral.vy(ix,iy,iz)*dataNeutral.vy(ix,iy,iz)
            + dataNeutral.vz(ix,iy,iz)*dataNeutral.vz(ix,iy,iz);


        SAMS::T_dataType v2_n_old =
              dataOld.vx_n(ix,iy,iz)*dataOld.vx_n(ix,iy,iz)
            + dataOld.vy_n(ix,iy,iz)*dataOld.vy_n(ix,iy,iz)
            + dataOld.vz_n(ix,iy,iz)*dataOld.vz_n(ix,iy,iz);



        SAMS::T_dataType kinetic_error =
              0.5
            *
              (
                data.rho(ix,iy,iz)*v2_p
                -
                dataOld.rho(ix,iy,iz)*v2_p_old
              )

            +

              0.5
            *
              (
                dataNeutral.rho(ix,iy,iz)*v2_n
                -
                dataOld.rho_n(ix,iy,iz)*v2_n_old
              );



        //
        // Store
        //
        buffers.mass_error[index] =
            drho_p + drho_n;


        buffers.momentum_x_error[index] =
            dp_x + dn_x;


        buffers.momentum_y_error[index] =
            dp_y + dn_y;


        buffers.momentum_z_error[index] =
            dp_z + dn_z;


        buffers.thermal_energy_error[index] =
            thermal_error;


        buffers.kinetic_energy_error[index] =
            kinetic_error;


        buffers.energy_error[index] =
            thermal_error + kinetic_error;


    },
    Range(0,nx),
    Range(0,ny),
    Range(0,nz));



    //
    // Reduction
    //
    SAMS::T_dataType total_mass_error = 0.0;
    SAMS::T_dataType total_momentum_x_error = 0.0;
    SAMS::T_dataType total_momentum_y_error = 0.0;
    SAMS::T_dataType total_momentum_z_error = 0.0;

    SAMS::T_dataType total_thermal_error = 0.0;
    SAMS::T_dataType total_kinetic_error = 0.0;
    SAMS::T_dataType total_energy_error = 0.0;



    for (int ix=0; ix<nx; ix++)
    {
        for (int iy=0; iy<ny; iy++)
        {
            for (int iz=0; iz<nz; iz++)
            {

                int index =
                      ix
                    + (nx+1)*(iy
                    + (ny+1)*iz);



                total_mass_error +=
                    buffers.mass_error[index];


                total_momentum_x_error +=
                    buffers.momentum_x_error[index];


                total_momentum_y_error +=
                    buffers.momentum_y_error[index];


                total_momentum_z_error +=
                    buffers.momentum_z_error[index];


                total_thermal_error +=
                    buffers.thermal_energy_error[index];


                total_kinetic_error +=
                    buffers.kinetic_energy_error[index];


                total_energy_error +=
                    buffers.energy_error[index];

            }
        }
    }



    printf("\nSource conservation diagnostic\n");
    printf("--------------------------------\n");
    printf("Mass            : %.12e\n", total_mass_error);
    printf("Momentum x      : %.12e\n", total_momentum_x_error);
    printf("Momentum y      : %.12e\n", total_momentum_y_error);
    printf("Momentum z      : %.12e\n", total_momentum_z_error);
    printf("Thermal energy  : %.12e\n", total_thermal_error);
    printf("Kinetic energy  : %.12e\n", total_kinetic_error);
    printf("Total energy    : %.12e\n", total_energy_error);
    printf("--------------------------------\n");



    delete[] buffers.mass_error;
    delete[] buffers.momentum_x_error;
    delete[] buffers.momentum_y_error;
    delete[] buffers.momentum_z_error;
    delete[] buffers.energy_error;
    delete[] buffers.thermal_energy_error;
    delete[] buffers.kinetic_energy_error;

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T_EOS>
void PIP<T_EOS>::checkSourceConservation(
        LARE::LARE3DST<T_EOS>::simulationData &data,
        LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral,
        data_two_fluid_source &plasma_source)
{
    using Range = portableWrapper::Range;

    const int nx = data.nx;
    const int ny = data.ny;
    const int nz = data.nz;

    SAMS::T_dataType total_mass_error = 0.0;
    SAMS::T_dataType total_momentum_x_error = 0.0;
    SAMS::T_dataType total_momentum_y_error = 0.0;
    SAMS::T_dataType total_momentum_z_error = 0.0;
    SAMS::T_dataType total_energy_error = 0.0;
    SAMS::T_dataType total_expected_thermal = 0.0;

    //
    // Use simple host arrays.
    //
    int ncell = (nx + 1) * (ny + 1) * (nz + 1);

    SAMS::T_dataType *mass_error =
        new SAMS::T_dataType[ncell];

    SAMS::T_dataType *momentum_x_error =
        new SAMS::T_dataType[ncell];

    SAMS::T_dataType *momentum_y_error =
        new SAMS::T_dataType[ncell];

    SAMS::T_dataType *momentum_z_error =
        new SAMS::T_dataType[ncell];
        
    SAMS::T_dataType *thermal_source =
        new SAMS::T_dataType[ncell];
    
    SAMS::T_dataType *kinetic_source =
        new SAMS::T_dataType[ncell];
        
    SAMS::T_dataType *expected_thermal =
        new SAMS::T_dataType[ncell];


    //
    // Calculate source conservation error at every cell.
    //
    portableWrapper::applyKernel(
        LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz)
        {
            int index =
                  ix
                + (nx + 1) *
                  (
                      iy
                    + (ny + 1) * iz
                  );


            //
            // Mass source
            //
            mass_error[index] =
                  plasma_source.source_mass(ix,iy,iz)
                + plasma_source.source_mass_n(ix,iy,iz);

            // Get rho at the vertex
            SAMS::T_dataType rho_plasma_vertex =
                  0.125 *
                  (
                      data.rho(ix  ,iy  ,iz  )
                    + data.rho(ix+1,iy  ,iz  )
                    + data.rho(ix  ,iy+1,iz  )
                    + data.rho(ix+1,iy+1,iz  )
                    + data.rho(ix  ,iy  ,iz+1)
                    + data.rho(ix+1,iy  ,iz+1)
                    + data.rho(ix  ,iy+1,iz+1)
                    + data.rho(ix+1,iy+1,iz+1)
                  );

            SAMS::T_dataType rho_neutral_vertex =
                  0.125 *
                  (
                      dataNeutral.rho(ix  ,iy  ,iz  )
                    + dataNeutral.rho(ix+1,iy  ,iz  )
                    + dataNeutral.rho(ix  ,iy+1,iz  )
                    + dataNeutral.rho(ix+1,iy+1,iz  )
                    + dataNeutral.rho(ix  ,iy  ,iz+1)
                    + dataNeutral.rho(ix+1,iy  ,iz+1)
                    + dataNeutral.rho(ix  ,iy+1,iz+1)
                    + dataNeutral.rho(ix+1,iy+1,iz+1)
                  );

            //
            // Momentum source
            //
            momentum_x_error[index] =
                  rho_plasma_vertex
                * plasma_source.source_v_x(ix,iy,iz)
                +
                  rho_neutral_vertex
                * plasma_source.source_v_x_n(ix,iy,iz);


            momentum_y_error[index] =
                  rho_plasma_vertex
                * plasma_source.source_v_y(ix,iy,iz)
                +
                  rho_neutral_vertex
                * plasma_source.source_v_y_n(ix,iy,iz);


            momentum_z_error[index] =
                  rho_plasma_vertex
                * plasma_source.source_v_z(ix,iy,iz)
                +
                  rho_neutral_vertex
                * plasma_source.source_v_z_n(ix,iy,iz);
    
            //
            // Thermal energy 
            //
            SAMS::T_dataType thermal_p =
                data.rho(ix,iy,iz)
                * plasma_source.source_energy(ix,iy,iz);

            SAMS::T_dataType thermal_n =
                dataNeutral.rho(ix,iy,iz)
                * plasma_source.source_energy_n(ix,iy,iz);

            thermal_source[index] =
                thermal_p + thermal_n;
                
            //
            // Kinetic energy
            //
            kinetic_source[index] =
                data.rho(ix,iy,iz)
                    * (
                          data.vx(ix,iy,iz) * plasma_source.source_v_x(ix,iy,iz)
                        + data.vy(ix,iy,iz) * plasma_source.source_v_y(ix,iy,iz)
                        + data.vz(ix,iy,iz) * plasma_source.source_v_z(ix,iy,iz)
                      )
                +
                dataNeutral.rho(ix,iy,iz)
                    * (
                          dataNeutral.vx(ix,iy,iz) * plasma_source.source_v_x_n(ix,iy,iz)
                        + dataNeutral.vy(ix,iy,iz) * plasma_source.source_v_y_n(ix,iy,iz)
                        + dataNeutral.vz(ix,iy,iz) * plasma_source.source_v_z_n(ix,iy,iz)
                      );
                      
            //
            // expected thermal energy
            //
            //
            // Get cell-centred velocities
            //
            SAMS::T_dataType vx_centre =
                (
                    data.vx(ix,iy,iz)
                  + data.vx(ix,iy-1,iz)
                  + data.vx(ix,iy,iz-1)
                  + data.vx(ix,iy-1,iz-1)
                  + data.vx(ix-1,iy,iz)
                  + data.vx(ix-1,iy-1,iz)
                  + data.vx(ix-1,iy,iz-1)
                  + data.vx(ix-1,iy-1,iz-1)
                ) * 0.125;

            SAMS::T_dataType vy_centre =
                (
                    data.vy(ix,iy,iz)
                  + data.vy(ix,iy-1,iz)
                  + data.vy(ix,iy,iz-1)
                  + data.vy(ix,iy-1,iz-1)
                  + data.vy(ix-1,iy,iz)
                  + data.vy(ix-1,iy-1,iz)
                  + data.vy(ix-1,iy,iz-1)
                  + data.vy(ix-1,iy-1,iz-1)
                ) * 0.125;

            SAMS::T_dataType vz_centre =
                (
                    data.vz(ix,iy,iz)
                  + data.vz(ix,iy-1,iz)
                  + data.vz(ix,iy,iz-1)
                  + data.vz(ix,iy-1,iz-1)
                  + data.vz(ix-1,iy,iz)
                  + data.vz(ix-1,iy-1,iz)
                  + data.vz(ix-1,iy,iz-1)
                  + data.vz(ix-1,iy-1,iz-1)
                ) * 0.125;

            SAMS::T_dataType vx_n_centre =
                (
                    dataNeutral.vx(ix,iy,iz)
                  + dataNeutral.vx(ix,iy-1,iz)
                  + dataNeutral.vx(ix,iy,iz-1)
                  + dataNeutral.vx(ix,iy-1,iz-1)
                  + dataNeutral.vx(ix-1,iy,iz)
                  + dataNeutral.vx(ix-1,iy-1,iz)
                  + dataNeutral.vx(ix-1,iy,iz-1)
                  + dataNeutral.vx(ix-1,iy-1,iz-1)
                ) * 0.125;

            SAMS::T_dataType vy_n_centre =
                (
                    dataNeutral.vy(ix,iy,iz)
                  + dataNeutral.vy(ix,iy-1,iz)
                  + dataNeutral.vy(ix,iy,iz-1)
                  + dataNeutral.vy(ix,iy-1,iz-1)
                  + dataNeutral.vy(ix-1,iy,iz)
                  + dataNeutral.vy(ix-1,iy-1,iz)
                  + dataNeutral.vy(ix-1,iy,iz-1)
                  + dataNeutral.vy(ix-1,iy-1,iz-1)
                ) * 0.125;

            SAMS::T_dataType vz_n_centre =
                (
                    dataNeutral.vz(ix,iy,iz)
                  + dataNeutral.vz(ix,iy-1,iz)
                  + dataNeutral.vz(ix,iy,iz-1)
                  + dataNeutral.vz(ix,iy-1,iz-1)
                  + dataNeutral.vz(ix-1,iy,iz)
                  + dataNeutral.vz(ix-1,iy-1,iz)
                  + dataNeutral.vz(ix-1,iy,iz-1)
                  + dataNeutral.vz(ix-1,iy-1,iz-1)
                ) * 0.125;

            SAMS::T_dataType dvx = vx_n_centre - vx_centre;
            SAMS::T_dataType dvy = vy_n_centre - vy_centre;
            SAMS::T_dataType dvz = vz_n_centre - vz_centre;

            SAMS::T_dataType vd2 = dvx*dvx + dvy*dvy + dvz*dvz;
            
            expected_thermal[index] =
                plasma_source.ac(ix,iy,iz)
                * data.rho(ix,iy,iz)
                * dataNeutral.rho(ix,iy,iz)
                * vd2;
        },
        Range(0,nx),
        Range(0,ny),
        Range(0,nz)
    );


    //
    // Reduction.
    //
    for (int ix = 0; ix < nx; ix++)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            for (int iz = 0; iz < nz; iz++)
            {
                int index =
                      ix
                    + (nx + 1) *
                      (
                          iy
                        + (ny + 1) * iz
                      );


                total_mass_error +=
                    mass_error[index];

                total_momentum_x_error +=
                    momentum_x_error[index];

                total_momentum_y_error +=
                    momentum_y_error[index];

                total_momentum_z_error +=
                    momentum_z_error[index];
                    
                total_expected_thermal +=
                    expected_thermal[index];
                    
                total_energy_error +=
                //    thermal_source[index]+kinetic_source[index];
                    thermal_source[index]-expected_thermal[index];
            }
        }
    }


    printf("\nSource conservation diagnostic\n");
    printf("--------------------------------\n");
    if ((plasma_source.ion_rec_empirical) || (plasma_source.ion_rec_nlevel)){
        printf("\nSource terms are complicated when ionisation is on\n");
        printf("\nErrors are because of the missing terms from drho/dt type terms\n");
        printf("--------------------------------\n");
    }

    printf(
        "Mass            : %.12e\n",
        total_mass_error
    );

    printf(
        "Momentum x      : %.12e\n",
        total_momentum_x_error
    );

    printf(
        "Momentum y      : %.12e\n",
        total_momentum_y_error
    );

    printf(
        "Momentum z      : %.12e\n",
        total_momentum_z_error
    );
    
    printf(
        "Energy error      : %.12e\n",
        total_energy_error
    );

    printf("--------------------------------\n");


    delete[] mass_error;
    delete[] momentum_x_error;
    delete[] momentum_y_error;
    delete[] momentum_z_error;
}

/////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////
// Implicit integration of the two-fluid source terms with PETSc
//
// All PETSc calls are in petscStiffSolver.cpp; this section only packs the SAMS arrays
// into a plain host array and evaluates the right hand side for the solver.
//
// State: 10 variables per cell, cell by cell, over Range(-1,n) in each direction (the same
// cells that apply_two_fluid_source updates):
//     +0 rho        +1 rho_n
//     +2 vx +3 vy +4 vz            (plasma velocity)
//     +5 vx_n +6 vy_n +7 vz_n      (neutral velocity)
//     +8 energy_ion +9 energy_n
// The right hand side is exactly get_two_fluid_source, which returns d/dt of these
// variables (the explicit scheme does u += dt*source).
//
// Because of the staggered grid the sources of one cell depend on the state of its
// neighbours (8-point averages). The right hand side keeps that coupling exactly; only
// the Jacobian is lagged (diagonal 10x10 blocks, see petscStiffSolver.cpp).
//
// Host memory only: the raw array is accessed inside applyKernel, which is valid for the
// serial / OpenMP backends but not for CUDA/HIP/SYCL builds.
/////////////////////////////////////////////////////////////////////////////////////

/*
* Start-up record of the run configuration. Everything that selects the physics or the solver
* is printed here, so a log file alone identifies the run that produced it.
* Add new data_two_fluid_source flags to this list when they are introduced.
*/
template<typename T_EOS>
void PIP<T_EOS>::printConfiguration(typename LARE::LARE3DST<T_EOS>::simulationData &data,
                                    typename LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral,
                                    data_two_fluid_source &ps) {
    auto onoff = [](bool b) { return b ? "on" : "off"; };
    const char *petsc_env = std::getenv("PETSC_OPTIONS");

    printf("=====================================================\n");
    printf("Two-fluid (PIP) run configuration\n");
    printf("=====================================================\n");
    printf("Build\n");
    printf("  git commit             : %s (branch %s)%s\n", SAMS_GIT_HASH, SAMS_GIT_BRANCH,
           std::string(SAMS_GIT_DIRTY) == "yes" ? "  ** UNCOMMITTED CHANGES **" : "");
    printf("  PETSc                  : %s\n", SAMS_PETSC::petscVersion().c_str());
    printf("Grid and time\n");
    printf("  cells (local)          : nx = %lld, ny = %lld, nz = %lld\n",
           static_cast<long long>(data.nx), static_cast<long long>(data.ny), static_cast<long long>(data.nz));
    printf("  domain                 : x [%g, %g], y [%g, %g], z [%g, %g]\n",
           static_cast<double>(data.x_min), static_cast<double>(data.x_max),
           static_cast<double>(data.y_min), static_cast<double>(data.y_max),
           static_cast<double>(data.z_min), static_cast<double>(data.z_max));
    printf("  t_end                  : %g (snapshots every %g)\n",
           static_cast<double>(data.t_end), static_cast<double>(data.dt_snapshots));
    printf("  dt_multiplier          : %g\n", static_cast<double>(data.dt_multiplier));
    printf("Fluids\n");
    printf("  gas_gamma              : plasma %g, neutral %g\n",
           static_cast<double>(data.gas_gamma), static_cast<double>(dataNeutral.gas_gamma));
    printf("  shock viscosity        : visc1 = %g, visc2 = %g\n",
           static_cast<double>(data.visc1), static_cast<double>(data.visc2));
    printf("  resistive MHD          : %s (eta_background = %g, eta0 = %g, j_max = %g)\n",
           onoff(data.resistiveMHD), static_cast<double>(data.eta_background),
           static_cast<double>(data.eta0), static_cast<double>(data.j_max));
    printf("Two-fluid coupling\n");
    printf("  collisions             : %s (alpha0 = %g)\n", onoff(ps.collisions), static_cast<double>(ps.alpha0));
    printf("  vertex_rates           : %s\n", onoff(ps.vertex_rates));
    printf("  ion_rec_empirical      : %s\n", onoff(ps.ion_rec_empirical));
    printf("  ion_rec_nlevel         : %s\n", onoff(ps.ion_rec_nlevel));
    if (ps.ion_rec_nlevel) {
        printf("  atomic rates file      : %s\n", ps.data_path.c_str());
    }
    printf("  T0 = %g K, n0 = %g, t_ir = %g\n",
           static_cast<double>(ps.T0), static_cast<double>(ps.n0), static_cast<double>(ps.t_ir));
    printf("Source-term time integration\n");
    if (ps.implicit) {
        printf("  method                 : IMPLICIT (PETSc), Strang split\n");
        printf("  implicit_rtol / atol   : %.3e / %.3e (PETSC_OPTIONS may override; see solver line at first step)\n",
               static_cast<double>(ps.implicit_rtol), static_cast<double>(ps.implicit_atol));
        printf("  implicit_limit_dt      : %s (%s)\n", onoff(ps.implicit_limit_dt),
               ps.implicit_limit_dt ? "dt also limited by the collisional timescale" : "dt set by the hydro CFL only");
    } else if (ps.substepping) {
        printf("  method                 : explicit with substepping, Strang split (max speed-up %g)\n",
               static_cast<double>(ps.substep_max_speedup));
    } else {
        printf("  method                 : explicit, Strang split, dt limited by the collisional timescale\n");
    }
    printf("  PETSC_OPTIONS          : %s\n", petsc_env ? petsc_env : "(not set)");
    printf("Diagnostics\n");
    printf("  check_conservation     : %s\n", onoff(ps.check_conservation));
    printf("  check_source           : %s\n", onoff(ps.check_source));
    printf("  watch_state            : %s (trend line every %d steps)\n", onoff(ps.watch_state), ps.watch_interval);
    printf("  debug_twofluid_steps   : %s\n", onoff(ps.debug_twofluid_steps));

    // Combinations that are legal to compile but almost certainly not what was intended
    int n_warn = 0;
    auto warn = [&](const char *msg) {
        if (n_warn++ == 0) printf("Configuration warnings\n");
        printf("  WARNING: %s\n", msg);
    };
    if (ps.ion_rec_empirical && ps.ion_rec_nlevel)
        warn("ion_rec_empirical and ion_rec_nlevel are both on; both sets of ionisation/recombination sources are computed.");
    if (ps.implicit && ps.ion_rec_nlevel)
        warn("implicit=true does not support ion_rec_nlevel; the run will abort at the first source solve.");
    if (ps.implicit && !ps.collisions)
        warn("implicit=true with collisions=false: the implicit solve is skipped entirely.");
    if (ps.implicit && ps.substepping)
        warn("implicit=true and substepping=true: substepping is ignored when implicit is on.");
    if (!ps.implicit && petsc_env)
        warn("PETSC_OPTIONS is set but implicit=false, so it has no effect.");
    if (std::string(SAMS_GIT_DIRTY) == "yes")
        warn("the build contains uncommitted changes; the git commit above does not fully identify the code.");
    printf("=====================================================\n");
    std::fflush(stdout);
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_CUDA) || defined(USE_CUDA) || defined(USE_HIP)
#define TWOFLUID_PETSC_DEVICE_BUILD 1
#endif

template<typename T_EOS>
void PIP<T_EOS>::ensureStiffSolver(typename LARE::LARE3DST<T_EOS>::simulationData &data, data_two_fluid_source &plasma_source) {
    if (this->stiff_solver) return;

#ifdef TWOFLUID_PETSC_DEVICE_BUILD
    std::fprintf(stderr, "The implicit PETSc source solver currently only supports host (serial/OpenMP) builds. Set implicit=false.\n");
    std::abort();
#endif
    if (plasma_source.ion_rec_nlevel) {
        std::fprintf(stderr, "The implicit PETSc source solver does not support ion_rec_nlevel (level populations are not part of the state vector). Use ion_rec_empirical.\n");
        std::abort();
    }

    const std::int64_t sx = data.nx + 2, sy = data.ny + 2, sz = data.nz + 2;
    const std::int64_t ncells = sx * sy * sz;

    // Cells 0..n (inclusive) are evolved and checked for physical validity. The outermost
    // ghost layer (index -1) is frozen during the solve and not checked.
    std::vector<unsigned char> check(static_cast<size_t>(ncells), 0);
    for (std::int64_t iz = 0; iz <= data.nz; ++iz) {
        for (std::int64_t iy = 0; iy <= data.ny; ++iy) {
            for (std::int64_t ix = 0; ix <= data.nx; ++ix) {
                check[static_cast<size_t>(((iz + 1) * sy + (iy + 1)) * sx + (ix + 1))] = 1;
            }
        }
    }
    const std::vector<int> positive = {0, 1, 8, 9}; // rho, rho_n, energy_ion, energy_n

    this->u_buf.assign(static_cast<size_t>(ncells) * petsc_stride, 0.0);
    this->u_start_buf.assign(static_cast<size_t>(ncells) * petsc_stride, 0.0);

    this->petsc_ctx.pip = this;
    this->petsc_ctx.data = &data;
    this->petsc_ctx.plasma_source = &plasma_source;

    this->stiff_solver = std::make_shared<SAMS_PETSC::StiffSolver>(
        ncells, petsc_stride, check, positive,
        &PIP<T_EOS>::RHSAdapter, &this->petsc_ctx,
        static_cast<double>(plasma_source.implicit_rtol), static_cast<double>(plasma_source.implicit_atol));

    // What PETSc is actually using, after any PETSC_OPTIONS / petscrc overrides
    printf("=====================================================\n"
           "PETSc implicit source solver created (settings in effect):\n%s"
           "=====================================================\n",
           this->stiff_solver->describe().c_str());
    std::fflush(stdout);
}

template<typename T_EOS>
void PIP<T_EOS>::finalize() {
    this->stiff_solver.reset();
    SAMS_PETSC::finalizePETSc();
}


/*
* Debugging aid: scan the whole state (ghost layer included) for NaN/Inf, remember the last clean
* check so the point where a NaN first appears can be bracketed, and optionally print a one-line
* trend summary. Reads the arrays on the host, so it is only valid for serial/OpenMP builds.
*/
template<typename T_EOS>
void PIP<T_EOS>::watchState(typename LARE::LARE3DST<T_EOS>::simulationData &data,
                            typename LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral,
                            data_two_fluid_source &plasma_source,
                            const char *label, double time, bool trail) {
    static const char *comp_name[10] = {"rho", "rho_n", "vx", "vy", "vz", "vx_n", "vy_n", "vz_n", "energy_ion", "energy_n"};
    static bool reported = false;
    static const char *last_clean_label = "(no clean check yet)";
    static double last_clean_time = -1.0;
    static long trail_count = 0;

    long long n_bad = 0, bad_min_x = LLONG_MAX, bad_max_x = -1, n_bad_left = 0;
    int bx = 0, by = 0, bz = 0, bcomp = -1;
    double bval = 0.0;
    double vmax = 0.0, emin = DBL_MAX, emax = -DBL_MAX, rmin = DBL_MAX, rmax = -DBL_MAX, rnmin = DBL_MAX;
    int mx = 0, my = 0, mz = 0;

    for (int iz = -1; iz <= data.nz; ++iz) {
        for (int iy = -1; iy <= data.ny; ++iy) {
            for (int ix = -1; ix <= data.nx; ++ix) {
                const double v[10] = {
                    static_cast<double>(data.rho(ix, iy, iz)),        static_cast<double>(dataNeutral.rho(ix, iy, iz)),
                    static_cast<double>(data.vx(ix, iy, iz)),         static_cast<double>(data.vy(ix, iy, iz)),
                    static_cast<double>(data.vz(ix, iy, iz)),         static_cast<double>(dataNeutral.vx(ix, iy, iz)),
                    static_cast<double>(dataNeutral.vy(ix, iy, iz)),  static_cast<double>(dataNeutral.vz(ix, iy, iz)),
                    static_cast<double>(data.energy_ion(ix, iy, iz)), static_cast<double>(dataNeutral.energy(ix, iy, iz))};
                bool finite = true;
                for (int k = 0; k < 10; ++k) {
                    if (!std::isfinite(v[k])) {
                        if (finite) { // first bad component of this cell
                            if (n_bad == 0) { bx = ix; by = iy; bz = iz; bcomp = k; bval = v[k]; }
                            ++n_bad;
                            bad_min_x = std::min<long long>(bad_min_x, ix);
                            bad_max_x = std::max<long long>(bad_max_x, ix);
                            if (ix <= data.nx / 2) ++n_bad_left;
                        }
                        finite = false;
                    }
                }
                if (finite && ix >= 0 && iy >= 0 && iz >= 0) {
                    const double speed = std::sqrt(v[2] * v[2] + v[3] * v[3] + v[4] * v[4]);
                    if (speed > vmax) { vmax = speed; mx = ix; my = iy; mz = iz; }
                    emin = std::min(emin, v[8]);  emax = std::max(emax, v[8]);
                    rmin = std::min(rmin, v[0]);  rmax = std::max(rmax, v[0]);
                    rnmin = std::min(rnmin, v[1]);
                }
            }
        }
    }

    if (n_bad > 0) {
        if (!reported) {
            reported = true;
            std::fprintf(stderr,
                "\nNON-FINITE state first detected: %s (t = %s%.8g)\n"
                "  %lld cells contain NaN/Inf; first at (%d,%d,%d), component %s, value %g%s\n"
                "  x index of all bad cells: %lld to %lld; %lld in the left half of the box, %lld in the right half\n"
                "  last clean check: %s (t = %.8g)\n"
                "  so the NaN was created between those two points\n\n",
                label, time < 0.0 ? "n/a " : "", time < 0.0 ? 0.0 : time, n_bad, bx, by, bz, comp_name[bcomp], bval,
                (bx == -1 || by == -1 || bz == -1) ? " (outermost ghost layer)" : "",
                bad_min_x, bad_max_x, n_bad_left, n_bad - n_bad_left,
                last_clean_label, last_clean_time);
        }
    } else {
        last_clean_label = label;
        last_clean_time = time;
        if (trail && plasma_source.watch_interval > 0 && (++trail_count % plasma_source.watch_interval) == 0) {
            printf("watch t = %.6g: max|v_plasma| = %.4e at (%d,%d,%d); energy_ion in [%.4e, %.4e]; rho in [%.4e, %.4e]; min rho_n = %.4e\n",
                   time, vmax, mx, my, mz, emin, emax, rmin, rmax, rnmin);
        }
    }
}

/*
* Solve  du/dt = S(u)  over dt_solve, starting from the current state in data/dataNeutral.
*
* The starting state and the final solution are validated here (every value finite; rho, rho_n,
* energy_ion, energy_n > 0 in the evolved cells 0..n). PETSc's own stage-validity hook is not used
* by default (see petscStiffSolver.cpp).
*/
template<typename T_EOS>
void PIP<T_EOS>::implicitSourceSolve(typename LARE::LARE3DST<T_EOS>::simulationData &data,
                                     typename LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral,
                                     data_two_fluid_source &plasma_source,
                                     SAMS::T_dataType dt_solve) {
    // apply_two_fluid_source only applies the exchange terms when collisions are on
    if (!plasma_source.collisions) return;
 
    this->ensureStiffSolver(data, plasma_source);
 
    // Refresh the callback context (the package object may have moved since the last solve)
    this->petsc_ctx.pip           = this;
    this->petsc_ctx.data          = &data;
    this->petsc_ctx.dataNeutral   = &dataNeutral;
    this->petsc_ctx.plasma_source = &plasma_source;
    this->stiff_solver->setUser(&this->petsc_ctx);
 
    // ---- helpers to validate a packed state -------------------------------------------------
    static const char *comp_name[10] = {"rho", "rho_n", "vx", "vy", "vz", "vx_n", "vy_n", "vz_n", "energy_ion", "energy_n"};
    static const int pos_comp[4] = {0, 1, 8, 9}; // must stay > 0
    const std::int64_t csx = data.nx + 2, csy = data.ny + 2;
 
    struct StateScan {
        long long n_bad = 0, n_checked = 0, first_bad_cell = -1;
        int first_bad_comp = -1;
        double min_val[4] = {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX}; // rho, rho_n, energy_ion, energy_n
        long long min_bad_x = LLONG_MAX, max_bad_x = -1, n_bad_left = 0; // where the bad cells are (x index)
    };
    auto scan_state = [&](const std::vector<double> &buf) {
        StateScan r;
        for (std::int64_t iz = 0; iz <= data.nz; ++iz) {
            for (std::int64_t iy = 0; iy <= data.ny; ++iy) {
                for (std::int64_t ix = 0; ix <= data.nx; ++ix) {
                    const std::int64_t cell = ((iz + 1) * csy + (iy + 1)) * csx + (ix + 1);
                    const double *v = &buf[static_cast<size_t>(cell) * petsc_stride];
                    ++r.n_checked;
                    int bad_comp = -1;
                    for (int k = 0; k < petsc_stride && bad_comp < 0; ++k) {
                        if (!std::isfinite(v[k])) bad_comp = k;
                    }
                    for (int q = 0; q < 4; ++q) {
                        if (std::isfinite(v[pos_comp[q]])) r.min_val[q] = std::min(r.min_val[q], v[pos_comp[q]]);
                        if (bad_comp < 0 && !(v[pos_comp[q]] > 0.0)) bad_comp = pos_comp[q];
                    }
                    if (bad_comp >= 0) {
                        ++r.n_bad;
                        if (r.first_bad_cell < 0) { r.first_bad_cell = cell; r.first_bad_comp = bad_comp; }
                        r.min_bad_x = std::min<long long>(r.min_bad_x, ix);
                        r.max_bad_x = std::max<long long>(r.max_bad_x, ix);
                        if (ix <= data.nx / 2) ++r.n_bad_left;
                    }
                }
            }
        }
        return r;
    };
    auto report_invalid = [&](const char *which, const StateScan &r, const std::vector<double> &buf, bool show_live) {
        const std::int64_t fx = (r.first_bad_cell % csx) - 1, fy = ((r.first_bad_cell / csx) % csy) - 1, fz = (r.first_bad_cell / (csx * csy)) - 1;
        const double *v = &buf[static_cast<size_t>(r.first_bad_cell) * petsc_stride];
        std::fprintf(stderr,
            "Implicit two-fluid source solve: the %s state is not physical.\n"
            "  invalid cells: %lld of %lld evolved cells (indices 0..n); first at (%lld,%lld,%lld), component %s\n"
            "  minimum over evolved cells: rho = %.6e, rho_n = %.6e, energy_ion = %.6e, energy_n = %.6e\n"
            "  state at that cell: rho=%.6e rho_n=%.6e vx=%.3e vy=%.3e vz=%.3e vx_n=%.3e vy_n=%.3e vz_n=%.3e E=%.6e E_n=%.6e\n",
            which, r.n_bad, r.n_checked, static_cast<long long>(fx), static_cast<long long>(fy), static_cast<long long>(fz),
            comp_name[r.first_bad_comp], r.min_val[0], r.min_val[1], r.min_val[2], r.min_val[3],
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]);
        std::fprintf(stderr, "  all %lld invalid cells: x index from %lld to %lld; %lld in the left half of the box, %lld in the right half\n"
                             "  (the first one above is just the first in scan order, which starts at x = 0)\n",
                     r.n_bad, r.min_bad_x, r.max_bad_x, r.n_bad_left, r.n_bad - r.n_bad_left);
        if (show_live) {
            std::fprintf(stderr, "  live arrays at that cell: rho=%.6e rho_n=%.6e E=%.6e E_n=%.6e\n"
                                 "  (if the live arrays are fine but the buffer is not, the packing failed)\n",
                static_cast<double>(data.rho(fx, fy, fz)), static_cast<double>(dataNeutral.rho(fx, fy, fz)),
                static_cast<double>(data.energy_ion(fx, fy, fz)), static_cast<double>(dataNeutral.energy(fx, fy, fz)));
        }
    };
 
    // ---- starting state ---------------------------------------------------------------------
    this->PackViewsToArray(data, dataNeutral, this->u_buf.data());
    {
        const StateScan pre = scan_state(this->u_buf);
        if (pre.n_bad > 0) {
            report_invalid("STARTING", pre, this->u_buf, true);
            std::abort();
        }
        static int preflight_prints = 0;
        if (preflight_prints++ < 3) {
            printf("Implicit solve pre-flight OK: %lld evolved cells, min rho = %.6e, rho_n = %.6e, energy_ion = %.6e, energy_n = %.6e\n",
                   pre.n_checked, pre.min_val[0], pre.min_val[1], pre.min_val[2], pre.min_val[3]);
        }
    }
 
    // The callback writes trial states into data/dataNeutral. Keep the starting state so a
    // failed solve can be rolled back.
    this->u_start_buf = this->u_buf;
 
    // ---- solve ------------------------------------------------------------------------------
    SAMS_PETSC::SolveInfo info;
    const bool ok = this->stiff_solver->solve(this->u_buf.data(), static_cast<double>(dt_solve), info);
    if (!ok) {
        this->UnpackArrayToViews(this->u_start_buf.data(), data, dataNeutral); // roll back
        auto describe = [&](long long idx, char *buf, size_t len) {
            if (idx < 0) { std::snprintf(buf, len, "none"); return; }
            const long long cell = idx / petsc_stride;
            const int comp = static_cast<int>(idx % petsc_stride);
            std::snprintf(buf, len, "cell (%lld,%lld,%lld), %s", static_cast<long long>((cell % csx) - 1),
                          static_cast<long long>((cell / csx) % csy) - 1, static_cast<long long>(cell / (csx * csy)) - 1, comp_name[comp]);
        };
        char dom[160], rhs[160];
        describe(info.domain_bad_index, dom, sizeof dom);
        describe(info.rhs_bad_index, rhs, sizeof rhs);
        std::fprintf(stderr,
            "Implicit two-fluid source solve FAILED (TSConvergedReason %d).\n"
            "  requested dt = %.6e, reached t = %.6e, last internal dt = %.6e\n"
            "  accepted steps = %d, rejected steps = %d\n"
            "  stages refused by the validity check = %lld\n"
            "  first refused stage: step size %.6e, %lld bad cells, first bad entry: %s, value %.6e\n"
            "                       checked vector: 2-norm %.6e, rho range [%.6e, %.6e]\n"
            "  right hand side evaluations containing NaN/Inf = %lld (first: %s)\n"
            "State rolled back to the start of the solve.\n",
            info.reason, static_cast<double>(dt_solve), info.time_reached, info.last_dt, info.steps, info.rejected,
            info.domain_rejects, info.domain_first_dt, info.domain_bad_count, dom, info.domain_bad_value,
            info.domain_vec_norm, info.domain_c0_min, info.domain_c0_max, info.rhs_nonfinite, rhs);
        std::abort();
    }
 
    // ---- final solution ---------------------------------------------------------------------
    {
        const StateScan post = scan_state(this->u_buf);
        if (post.n_bad > 0) {
            this->UnpackArrayToViews(this->u_start_buf.data(), data, dataNeutral); // roll back
            report_invalid("FINAL", post, this->u_buf, false);
            std::fprintf(stderr, "  (TS steps = %d, rejected = %d, dt = %.6e). The state has been rolled back to the start of the solve.\n"
                                 "  A smaller time step or tighter tolerances (PETSC_OPTIONS=\"-ts_rtol 1e-8\") may help.\n",
                         info.steps, info.rejected, static_cast<double>(dt_solve));
            std::abort();
        }
    }
    printf("Implicit PETSc source solve: dt = %.12e, TS steps = %d, rejected = %d\n",
           static_cast<double>(dt_solve), info.steps, info.rejected);
 
    // The live arrays hold the last stage state, so always copy the final solution back
    this->UnpackArrayToViews(this->u_buf.data(), data, dataNeutral);
}
 
// f = F(u): write the trial state into the live arrays, evaluate the exact two-fluid
// sources, and return them in the 10-stride layout.
template<typename T_EOS>
void PIP<T_EOS>::RHSAdapter(const double *u, double *f, void *user) {
    auto *c  = static_cast<PETScCtx*>(user);
    PIP *pip = c->pip;
 
    pip->UnpackArrayToViews(u, *c->data, *c->dataNeutral);
    pip->get_ac(*c->data, *c->dataNeutral, *c->plasma_source);
    pip->get_two_fluid_source(*c->data, *c->dataNeutral, *c->plasma_source);
    pip->PackSourcesToArray(*c->plasma_source, *c->data, f);
}
 
template<typename T_EOS>
void PIP<T_EOS>::PackViewsToArray(typename LARE::LARE3DST<T_EOS>::simulationData &data,
                                  typename LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral,
                                  double *u_raw) {
    using Range = portableWrapper::Range;
    const std::int64_t sx = data.nx + 2, sy = data.ny + 2;
 
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        const std::int64_t cell = ((static_cast<std::int64_t>(iz) + 1) * sy + (iy + 1)) * sx + (ix + 1);
        const std::int64_t idx = cell * 10;
 
        u_raw[idx + 0] = data.rho(ix, iy, iz);
        u_raw[idx + 1] = dataNeutral.rho(ix, iy, iz);
        u_raw[idx + 2] = data.vx(ix, iy, iz);
        u_raw[idx + 3] = data.vy(ix, iy, iz);
        u_raw[idx + 4] = data.vz(ix, iy, iz);
        u_raw[idx + 5] = dataNeutral.vx(ix, iy, iz);
        u_raw[idx + 6] = dataNeutral.vy(ix, iy, iz);
        u_raw[idx + 7] = dataNeutral.vz(ix, iy, iz);
        u_raw[idx + 8] = data.energy_ion(ix, iy, iz);
        u_raw[idx + 9] = dataNeutral.energy(ix, iy, iz);
    }, Range(-1, data.nx), Range(-1, data.ny), Range(-1, data.nz));
}
 
template<typename T_EOS>
void PIP<T_EOS>::UnpackArrayToViews(const double *u_raw,
                                    typename LARE::LARE3DST<T_EOS>::simulationData &data,
                                    typename LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral) {
    using Range = portableWrapper::Range;
    const std::int64_t sx = data.nx + 2, sy = data.ny + 2;
 
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        const std::int64_t cell = ((static_cast<std::int64_t>(iz) + 1) * sy + (iy + 1)) * sx + (ix + 1);
        const std::int64_t idx = cell * 10;
 
        data.rho(ix, iy, iz)           = u_raw[idx + 0];
        dataNeutral.rho(ix, iy, iz)    = u_raw[idx + 1];
        data.vx(ix, iy, iz)            = u_raw[idx + 2];
        data.vy(ix, iy, iz)            = u_raw[idx + 3];
        data.vz(ix, iy, iz)            = u_raw[idx + 4];
        dataNeutral.vx(ix, iy, iz)     = u_raw[idx + 5];
        dataNeutral.vy(ix, iy, iz)     = u_raw[idx + 6];
        dataNeutral.vz(ix, iy, iz)     = u_raw[idx + 7];
        data.energy_ion(ix, iy, iz)    = u_raw[idx + 8];
        dataNeutral.energy(ix, iy, iz) = u_raw[idx + 9];
    }, Range(-1, data.nx), Range(-1, data.ny), Range(-1, data.nz));
}
 
// Copy the source-term arrays into f using the same layout as the state.
// The outermost ghost layer (index -1) is frozen: the existing kernels only partly fill the
// sources there (e.g. the energy sources are computed on Range(0,n)), and boundary
// conditions refresh it anyway.
template<typename T_EOS>
void PIP<T_EOS>::PackSourcesToArray(data_two_fluid_source &plasma_source,
                                    typename LARE::LARE3DST<T_EOS>::simulationData &data,
                                    double *f_raw) {
    using Range = portableWrapper::Range;
    const std::int64_t sx = data.nx + 2, sy = data.ny + 2;
 
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        const std::int64_t cell = ((static_cast<std::int64_t>(iz) + 1) * sy + (iy + 1)) * sx + (ix + 1);
        const std::int64_t idx = cell * 10;
        const bool frozen = (ix == -1) || (iy == -1) || (iz == -1);
 
        f_raw[idx + 0] = frozen ? 0.0 : plasma_source.source_mass(ix, iy, iz);
        f_raw[idx + 1] = frozen ? 0.0 : plasma_source.source_mass_n(ix, iy, iz);
        f_raw[idx + 2] = frozen ? 0.0 : plasma_source.source_v_x(ix, iy, iz);
        f_raw[idx + 3] = frozen ? 0.0 : plasma_source.source_v_y(ix, iy, iz);
        f_raw[idx + 4] = frozen ? 0.0 : plasma_source.source_v_z(ix, iy, iz);
        f_raw[idx + 5] = frozen ? 0.0 : plasma_source.source_v_x_n(ix, iy, iz);
        f_raw[idx + 6] = frozen ? 0.0 : plasma_source.source_v_y_n(ix, iy, iz);
        f_raw[idx + 7] = frozen ? 0.0 : plasma_source.source_v_z_n(ix, iy, iz);
        f_raw[idx + 8] = frozen ? 0.0 : plasma_source.source_energy(ix, iy, iz);
        f_raw[idx + 9] = frozen ? 0.0 : plasma_source.source_energy_n(ix, iy, iz);
    }, Range(-1, data.nx), Range(-1, data.ny), Range(-1, data.nz));
}
 
// Explicit instantiation of PIP<idealGas> is done once, in twofluid.h (EOS_DEF). A second
// instantiation here is a "duplicate explicit instantiation" error.


}

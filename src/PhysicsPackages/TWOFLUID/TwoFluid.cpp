/*
* This is a two-fluid routine for all the multi-fluid physics modules
*/

//////////////////////No idea which ones of these are needed
#include <iostream>
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

        varRegistry.registerVariable<LARE::T_dataType>("level_populations", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts), SAMS::dimension("species",0));

//printf("after Levels \n");
//debug_rates(plasma_source,"after populations");
        
        varRegistry.registerVariable<LARE::T_dataType>("level_rates", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts), SAMS::dimension("species",0), SAMS::dimension("species",0));

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
        
        varRegistry.fillPPArray("level_populations", plasma_source.level_populations);
        pw::assign(plasma_source.level_populations, 0.0);
        varRegistry.fillPPArray("level_rates", plasma_source.level_rates);
        pw::assign(plasma_source.level_rates, 0.0);
        
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
                    
                    printf("Reference recombination rate %G %G %G\n",rec_rate,plasma_source.n0,plasma_source.T0); 
                    plasma_source.ref_rec+=rec_rate;
                };
                plasma_source.ref_rec=plasma_source.ref_rec*plasma_source.n0; //This all needs checking
                printf("Reference recombination rate %e \n",plasma_source.ref_rec); 
            };    
            exit(0);
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
        
        //Pressure 
        SAMS::T_dataType  pr_p=data.energy_ion(ix,iy,iz)*(data.gas_gamma-1.0)*data.rho(ix,iy,iz); //Note that this is electron+ion pressure hence the factor of 0.5 in the TeIR formula
        SAMS::T_dataType  pr_n=dataNeutral.energy(ix,iy,iz)*(data.gas_gamma-1.0)*dataNeutral.rho(ix,iy,iz);
        
        //work done on the neutrals through ionisation/recombiation
        //SAMS::T_dataType  Hn=0.5*(plasma_source.gm_rec(ix,iy,iz)*data.rho(ix,iy,iz)/dataNeutral.rho(ix,iy,iz)*(vp2-2.0*vpvn)+
        //                          plasma_source.gm_ion(ix,iy,iz)*vn2);
        //SAMS::T_dataType  Hp=0.5*(plasma_source.gm_ion(ix,iy,iz)*dataNeutral.rho(ix,iy,iz)/data.rho(ix,iy,iz)*(vn2-2.0*vpvn)+
        //                          plasma_source.gm_rec(ix,iy,iz)*vp2);
                                  
        //Thermal equalisation from IR
        SAMS::T_dataType  TeIR=(plasma_source.gm_ion(ix,iy,iz)*pr_n-0.5*plasma_source.gm_rec(ix,iy,iz)*pr_p)/(data.gas_gamma-1.0);
        
        
        //Corrected energy source terms
        plasma_source.source_energy(ix,iy,iz) +=TeIR/data.rho(ix,iy,iz);
        plasma_source.source_energy_n(ix,iy,iz) +=TeIR/dataNeutral.rho(ix,iy,iz);
        
        //Work out how much energy is spent/gained by IR processes
        if (plasma_source.ion_rec_empirical) { 
            //printf("ionisation energy, rho = %f %f \n",ionisation_energy, dataNeutral.rho(ix,iy,iz));
            plasma_source.source_energy(ix,iy,iz)+=(plasma_source.ion_heating(ix,iy,iz)+plasma_source.ion_loss(ix,iy,iz))/data.rho(ix,iy,iz);//factor of pho comes from denergy density being specified
        }
        
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
}

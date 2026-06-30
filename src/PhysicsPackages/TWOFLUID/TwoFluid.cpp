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
 
    DEVICEPREFIX INLINE LARE::T_dataType interpolate_rates(const data_two_fluid_source &plasma_source, LARE::T_dataType temperature,LARE::T_indexType lower_level, LARE::T_indexType upper_level);
    
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
    * Default values
    */
    template<typename T_EOS>
    void PIP<T_EOS>::defaultValues(data_two_fluid_source & data){
        data.alpha0=1.0;
    }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Register variables with the portable array manager.
     */
    template<typename T_EOS> 
    void PIP<T_EOS>::registerVariables(SAMS::harness &harness)
    {

        auto &varRegistry = harness.variableRegistry;

        const int ghosts = 2; // 2 Ghost cells at top and bottom of each dimension

        varRegistry.registerVariable<T_dataType>("PIPSource/ac", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));

        varRegistry.registerVariable<T_dataType>("PIPSource/mass", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        
        varRegistry.registerVariable<T_dataType>("PIPSource/mass_n", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        
        varRegistry.registerVariable<T_dataType>("PIPSource/energy", pw::arrayTags::accelerated, SAMS::dimension("X", ghosts), SAMS::dimension("Y", ghosts), SAMS::dimension("Z", ghosts));
        
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
        pw::assign(plasma_source.gm_rec, 0.0);
        
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
                //data.energy_electron(ix,iy,iz)+=0.5*data.dt*plasma_source.source_energy(ix,iy,iz);
                dataNeutral.energy(ix,iy,iz)+=0.5*data.dt*plasma_source.source_energy_n(ix,iy,iz);                 
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
                SAMS::T_dataType  temperature_ion = data.gas_gamma*data.energy_ion(ix,iy,iz)*(data.gas_gamma-1.0);
                SAMS::T_dataType  temperature_neutral = data.gas_gamma*dataNeutral.energy(ix,iy,iz)*(data.gas_gamma-1.0);
                plasma_source.ac(ix,iy,iz)=plasma_source.alpha0*std::sqrt(0.5*(temperature_neutral+temperature_ion));
            	}, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));

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
        };
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
                //Get Temperatures
                //LARE::T_dataType temperature_electron = 2.0*data.gas_gamma*data.energy_electron(ix,iy,iz)*(data.gas_gamma-1.0);
                LARE::T_dataType temperature_electron = 0.5*data.gas_gamma*data.energy_ion(ix,iy,iz)*(data.gas_gamma-1.0)*T0;
                LARE::T_dataType numberDensity_electron=data.rho(ix,iy,iz)*n0; 

                
                // --- Level population triangular loop---
                for (LARE::T_indexType lower_level = 1; lower_level < nLevels; ++lower_level) {
                    //Loop over (de)excitation
                    for (LARE::T_indexType upper_level = lower_level + 1; upper_level < nLevels; ++upper_level) {
                    
                        LARE::T_dataType rate_coefficient = interpolate_rates(plasma_source, temperature_electron, 
                                                            lower_level, upper_level);
                    
                    fprintf(stdout, "Excitation rate coefficient for levels %li to %li at temperature %e is %e vs %e \n", lower_level, upper_level, temperature_electron,rate_coefficient,plasma_source.hydrogen_excitation_rate(1,lower_level,upper_level));
                    

                        // triangular work for this cell
                        //fprintf(stdout, "Excitation rate coefficient for levels %li to %li at temperature %e is %e vs %e \n", lower_level, upper_level, temperature_electron, rate_coefficient,plasma_source.hydrogen_excitation_rate(1,lower_level,upper_level));
                        //Excitation rate
                        plasma_source.level_rates(ix,iy,iz,lower_level,upper_level)=gn[lower_level]/gn[upper_level]*numberDensity_electron*rate_coefficient;
                        //De-Excitation rate
                        plasma_source.level_rates(ix,iy,iz,upper_level,lower_level)=numberDensity_electron*rate_coefficient*std::exp((Eion[lower_level]-Eion[upper_level])/(kb_ev*temperature_electron));

                    }
                    //Calculate ionisation and recombination
                    LARE::T_dataType rate_coefficient = interpolate_rates(plasma_source, temperature_electron, 
                                                            lower_level, nLevels);
                    //Ionisation rate
                    plasma_source.level_rates(ix,iy,iz,lower_level,nLevels)=numberDensity_electron*rate_coefficient*std::exp((Eion[lower_level]-Eion[nLevels])/(kb_ev*temperature_electron));
                    //Recombination rate
                    LARE::T_dataType sahaRatio= numberDensity_electron*gn[lower_level]*std::pow(2.0*std::numbers::pi*kb_ev*temperature_electron*mass_electron/h_ev,-3.0/2.0);
                    plasma_source.level_rates(ix,iy,iz,nLevels,lower_level)=sahaRatio*plasma_source.level_rates(ix,iy,iz,lower_level,nLevels);
                    //fprintf(stdout, "Ionisation rate coefficient for levels %li to %li at temperature %e is %e \n", lower_level, nLevels, temperature_electron, rate_coefficient);
                }

                //Get ionisation and recomination rates
                plasma_source.gm_rec(ix,iy,iz)=0.0;
                plasma_source.gm_ion(ix,iy,iz)=0.0;
            	for (LARE::T_indexType lower_level = 1; lower_level < nLevels; ++lower_level) {
            	    plasma_source.gm_rec(ix,iy,iz)+=plasma_source.level_rates(ix,iy,iz,nLevels,lower_level);
            	    plasma_source.gm_ion(ix,iy,iz)+=plasma_source.level_rates(ix,iy,iz,lower_level,nLevels)*plasma_source.level_populations(ix,iy,iz,lower_level);
            	}
            	//plasma_source.gm_rec(ix,iy,iz)=plasma_source.gm_rec(ix,iy,iz)
            	plasma_source.gm_ion(ix,iy,iz)=plasma_source.gm_ion(ix,iy,iz)/dataNeutral.rho(ix,iy,iz); //Need to normalise the ionisation rate 
            	//THIS ALL NEEDS CHECKING
            	
            	//printf("%f %f %f %f %f \n",f_p,data.energy_electron(ix,iy,iz)*(data.gas_gamma-1.0)*data.rho(ix,iy,iz), temperature_electron,plasma_source.gm_rec(ix,iy,iz),plasma_source.gm_ion(ix,iy,iz));    
            }, Range(-1,data.nx+1), Range(-1,data.ny+1), Range(-1,data.nz+1));
            portableWrapper::fence();
        };
////////////////////////////////////////////////////////////////////////////////////////
//Get the source terms for the IR rates
    template<typename T_EOS>
 void PIP<T_EOS>::get_collisional_source_terms(LARE::LARE3DST<T_EOS>::simulationData &data, LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){	

    using Range = portableWrapper::Range;
    portableWrapper::applyKernel(LAMBDA(T_indexType ix, T_indexType iy, T_indexType iz) {
        //printf("getting source terms \n");    
        
        //SAMS::T_dataType   ac;
        //get_ac(data.alpha0,temperature_ion,temperature_neutral);
        
        
        //printf("getting vertex values \n");
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
        
        //printf("v_n %f %f \n", plasma_source.source_v_x(ix,iy,iz),plasma_source.source_v_x_n(ix,iy,iz));
        
        plasma_source.source_v_y(ix,iy,iz)=ac_vertex*rho_neutral_vertex*(dataNeutral.vy(ix,iy,iz)-data.vy(ix,iy,iz));
        plasma_source.source_v_y_n(ix,iy,iz)=-ac_vertex*rho_plasma_vertex*(dataNeutral.vy(ix,iy,iz)-data.vy(ix,iy,iz));
        
        plasma_source.source_v_z(ix,iy,iz)=ac_vertex*rho_neutral_vertex*(dataNeutral.vz(ix,iy,iz)-data.vz(ix,iy,iz));
        plasma_source.source_v_z_n(ix,iy,iz)=-ac_vertex*rho_plasma_vertex*(dataNeutral.vz(ix,iy,iz)-data.vz(ix,iy,iz));
        }, Range(-1,data.nx), Range(-1,data.ny), Range(-1,data.nz));
        
        
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
                        + 3.0/data.gas_gamma/2.0*(temperature_neutral-temperature_ion));
        //printf("getting neutral energy source terms \n");
        plasma_source.source_energy_n(ix,iy,iz)=-plasma_source.ac(ix,iy,iz)*data.rho(ix,iy,iz)*(\
                        0.5*vd2 \
                        + 3.0/data.gas_gamma/2.0*(temperature_neutral-temperature_ion));  
                                   
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
        plasma_source.source_v_x(ix,iy,iz) += -Gm_ion_vertex*rho_neutral_vertex*v_D_x/rho_plasma_vertex;
        plasma_source.source_v_y(ix,iy,iz) += -Gm_ion_vertex*rho_neutral_vertex*v_D_y/rho_plasma_vertex;
        plasma_source.source_v_z(ix,iy,iz) += -Gm_ion_vertex*rho_neutral_vertex*v_D_z/rho_plasma_vertex;
        plasma_source.source_v_x_n(ix,iy,iz) += Gm_rec_vertex*rho_plasma_vertex*v_D_x/rho_neutral_vertex;
        plasma_source.source_v_y_n(ix,iy,iz) += Gm_rec_vertex*rho_plasma_vertex*v_D_y/rho_neutral_vertex;
        plasma_source.source_v_z_n(ix,iy,iz) += Gm_rec_vertex*rho_plasma_vertex*v_D_z/rho_neutral_vertex;
        
        
        //Get velocity at cell centres
        SAMS::T_dataType  v_x_plasma_centre=  (data.vx(ix  , iy  , iz  ) + 
                                        data.vx(ix-1, iy  , iz  ) + 
                                        data.vx(ix  , iy-1, iz  ) + 
                                        data.vx(ix-1, iy-1, iz  ) + 
                                        data.vx(ix  , iy  , iz-1) + 
                                        data.vx(ix-1, iy  , iz-1) + 
                                        data.vx(ix  , iy-1, iz-1) + 
                                        data.vx(ix-1, iy-1, iz-1))* 
                                        0.125;
        SAMS::T_dataType  v_y_plasma_centre=  (data.vy(ix  , iy  , iz  ) + 
                                        data.vy(ix-1, iy  , iz  ) + 
                                        data.vy(ix  , iy-1, iz  ) + 
                                        data.vy(ix-1, iy-1, iz  ) + 
                                        data.vy(ix  , iy  , iz-1) + 
                                        data.vy(ix-1, iy  , iz-1) + 
                                        data.vy(ix  , iy-1, iz-1) + 
                                        data.vy(ix-1, iy-1, iz-1))* 
                                        0.125;
        SAMS::T_dataType  v_z_plasma_centre=  (data.vz(ix  , iy  , iz  ) + 
                                        data.vz(ix-1, iy  , iz  ) + 
                                        data.vz(ix  , iy-1, iz  ) + 
                                        data.vz(ix-1, iy-1, iz  ) + 
                                        data.vz(ix  , iy  , iz-1) + 
                                        data.vz(ix-1, iy  , iz-1) + 
                                        data.vz(ix  , iy-1, iz-1) + 
                                        data.vz(ix-1, iy-1, iz-1))* 
                                        0.125;
        SAMS::T_dataType  v_x_neutral_centre= (dataNeutral.vx(ix  , iy  , iz  ) + 
                                        dataNeutral.vx(ix-1, iy  , iz  ) + 
                                        dataNeutral.vx(ix  , iy-1, iz  ) + 
                                        dataNeutral.vx(ix-1, iy-1, iz  ) + 
                                        dataNeutral.vx(ix  , iy  , iz-1) + 
                                        dataNeutral.vx(ix-1, iy  , iz-1) + 
                                        dataNeutral.vx(ix  , iy-1, iz-1) + 
                                        dataNeutral.vx(ix-1, iy-1, iz-1))* 
                                        0.125;
        SAMS::T_dataType  v_y_neutral_centre= (dataNeutral.vy(ix  , iy  , iz  ) + 
                                        dataNeutral.vy(ix-1, iy  , iz  ) + 
                                        dataNeutral.vy(ix  , iy-1, iz  ) + 
                                        dataNeutral.vy(ix-1, iy-1, iz  ) + 
                                        dataNeutral.vy(ix  , iy  , iz-1) + 
                                        dataNeutral.vy(ix-1, iy  , iz-1) + 
                                        dataNeutral.vy(ix  , iy-1, iz-1) + 
                                        dataNeutral.vy(ix-1, iy-1, iz-1))* 
                                        0.125;
        SAMS::T_dataType  v_z_neutral_centre= (dataNeutral.vz(ix  , iy  , iz  ) + 
                                        dataNeutral.vz(ix-1, iy  , iz  ) + 
                                        dataNeutral.vz(ix  , iy-1, iz  ) + 
                                        dataNeutral.vz(ix-1, iy-1, iz  ) + 
                                        dataNeutral.vz(ix  , iy  , iz-1) + 
                                        dataNeutral.vz(ix-1, iy  , iz-1) + 
                                        dataNeutral.vz(ix  , iy-1, iz-1) + 
                                        dataNeutral.vz(ix-1, iy-1, iz-1))* 
                                        0.125;
        
        //Energy source terms
        plasma_source.source_energy(ix,iy,iz) += -0.5*(plasma_source.gm_rec(ix,iy,iz)*(pow(v_x_plasma_centre,2)+
                                                                                 pow(v_y_plasma_centre,2)+
                                                                                 pow(v_z_plasma_centre,2))
                                                        -plasma_source.gm_ion(ix,iy,iz)*(pow(v_x_neutral_centre,2)+
                                                                                pow(v_y_neutral_centre,2)+
                                                                                pow(v_z_neutral_centre,2))
                                                                              *dataNeutral.rho(ix,iy,iz)/data.rho(ix,iy,iz)                                                                                
                                                        )
                                                   -(plasma_source.gm_rec(ix,iy,iz)*data.energy_ion(ix,iy,iz)-plasma_source.gm_ion(ix,iy,iz)*dataNeutral.energy(ix,iy,iz)*dataNeutral.rho(ix,iy,iz)/data.rho(ix,iy,iz))/(data.gas_gamma-1.0); //Is this electron or ion energy (or mean energy)? is the half needed?
        
        plasma_source.source_energy_n(ix,iy,iz) += 0.5*(plasma_source.gm_rec(ix,iy,iz)*(data.vx(ix,iy,iz)*data.vx(ix,iy,iz)+
                                                                                data.vy(ix,iy,iz)*data.vy(ix,iy,iz)+
                                                                                data.vz(ix,iy,iz)*data.vz(ix,iy,iz))
                                                                                *data.rho(ix,iy,iz)/dataNeutral.rho(ix,iy,iz)
                                                        -plasma_source.gm_ion(ix,iy,iz)*(dataNeutral.vx(ix,iy,iz)*data.vx(ix,iy,iz)+
                                                                                dataNeutral.vy(ix,iy,iz)*data.vy(ix,iy,iz)+
                                                                                dataNeutral.vz(ix,iy,iz)*data.vz(ix,iy,iz))
                                                        )
                                                   +(plasma_source.gm_rec(ix,iy,iz)*data.energy_ion(ix,iy,iz)*data.rho(ix,iy,iz)/dataNeutral.rho(ix,iy,iz)-plasma_source.gm_ion(ix,iy,iz)*dataNeutral.energy(ix,iy,iz))/(data.gas_gamma-1.0); //Is this electron or ion energy (or mean energy)? is the half needed?
        
        //Work out how much energy is spent/gained by IR processes
        //if (two_fluid_flags.ion_rec_empirical) { 
            //printf("ionisation energy, rho = %f %f \n",ionisation_energy, dataNeutral.rho(ix,iy,iz));
            plasma_source.source_energy(ix,iy,iz)+=plasma_source.ion_loss(ix,iy,iz)/data.rho(ix,iy,iz);//factor of pho comes from denergy density being specified
        //}
        
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
//Routine for the reading the rates
template<typename T_EOS> 
    void PIP<T_EOS>::two_fluid_read_rates(data_two_fluid_source &plasma_source){

        int ncid = -1;
        int dim_samples = -1;
        int dim_start = -1;
        int dim_final = -1;
        size_t nsamps = 0;
        size_t nstarts = 0;
        size_t nfinals = 0;
        std::string data_path=plasma_source.data_path;
        int nc_status = nc_open(data_path.c_str(), NC_NOWRITE, &ncid);
        if (nc_status != NC_NOERR) {
            fprintf(stderr, "two_fluid_read_rates: nc_open failed for '%s': %s\n",
                    data_path.c_str(), nc_strerror(nc_status));
            return;
        }
        fprintf(stdout, "two_fluid_read_rates: using file '%s'\n", data_path.c_str());

        nc_status = nc_inq_dimid(ncid, "sample", &dim_samples);
        if (nc_status != NC_NOERR) {
            fprintf(stderr, "two_fluid_read_rates: missing dim 'sample': %s\n",
                    nc_strerror(nc_status));
            nc_close(ncid);
            return;
        }
        nc_status = nc_inq_dimid(ncid, "start_level", &dim_start);
        if (nc_status != NC_NOERR) {
            fprintf(stderr, "two_fluid_read_rates: missing dim 'start_level': %s\n",
                    nc_strerror(nc_status));
            nc_close(ncid);
            return;
        }
        nc_status = nc_inq_dimid(ncid, "final_level", &dim_final);
        if (nc_status != NC_NOERR) {
            fprintf(stderr, "two_fluid_read_rates: missing dim 'final_level': %s\n",
                    nc_strerror(nc_status));
            nc_close(ncid);
            return;
        }

        nc_status = nc_inq_dimlen(ncid, dim_samples, &nsamps);
        if (nc_status != NC_NOERR) {
            fprintf(stderr, "two_fluid_read_rates: dimlen 'sample' failed: %s\n",
                    nc_strerror(nc_status));
            nc_close(ncid);
            return;
        }
        nc_status = nc_inq_dimlen(ncid, dim_start, &nstarts);
        if (nc_status != NC_NOERR) {
            fprintf(stderr, "two_fluid_read_rates: dimlen 'start_level' failed: %s\n",
                    nc_strerror(nc_status));
            nc_close(ncid);
            return;
        }
        nc_status = nc_inq_dimlen(ncid, dim_final, &nfinals);
        if (nc_status != NC_NOERR) {
            fprintf(stderr, "two_fluid_read_rates: dimlen 'final_level' failed: %s\n",
                    nc_strerror(nc_status));
            nc_close(ncid);
            return;
        }

        if (nsamps == 0 || nstarts == 0 || nfinals == 0) {
            fprintf(stderr,
                    "two_fluid_read_rates: invalid dimensions (samples=%zu, start=%zu, final=%zu)\n",
                    nsamps, nstarts, nfinals);
            nc_close(ncid);
            return;
        }

        int var_logT = -1;
        int var_coeffs = -1;
        nc_status = nc_inq_varid(ncid, "logT", &var_logT);
        if (nc_status != NC_NOERR) {
            fprintf(stderr, "two_fluid_read_rates: missing var 'logT': %s\n",
                    nc_strerror(nc_status));
            nc_close(ncid);
            return;
        }
        nc_status = nc_inq_varid(ncid, "hydrogen_excitation_rate", &var_coeffs);
        if (nc_status != NC_NOERR) {
            fprintf(stderr, "two_fluid_read_rates: missing var 'hydrogen_excitation_rate': %s\n",
                    nc_strerror(nc_status));
            nc_close(ncid);
            return;
        }
        
        using Range = pw::Range;
        pw::portableArrayManager svManager;
        //pw::portableArray<LARE::T_dataType, 1> grid_logT;
        Range T_range = pw::Range(0, static_cast<LARE::T_indexType>(nsamps - 1));   // 0 .. nsamps-1
    Range n_start = pw::Range(0, static_cast<LARE::T_indexType>(nstarts - 1));  // 0 .. nstarts-1
    Range n_final = pw::Range(0, static_cast<LARE::T_indexType>(nfinals - 1));  // 0 .. nfinals-1
    svManager.allocate(plasma_source.hydrogen_excitation_rate, T_range, n_start, n_final);
    svManager.allocate(plasma_source.grid_logT, T_range);

        //manager.allocate(plasma_source.grid_logT,
        //                portableWrapper::Range(0, static_cast<LARE::T_indexType>(nsamps - 1)));
        std::vector<double> flat(nsamps * nstarts * nfinals, -1.0);

        nc_status = nc_get_var_double(ncid, var_logT, plasma_source.grid_logT.data());
        if (nc_status != NC_NOERR) {
            fprintf(stderr, "two_fluid_read_rates: read 'logT' failed: %s\n",
                    nc_strerror(nc_status));
            nc_close(ncid);
            return;
        }

        if (!flat.empty()) {
            nc_status = nc_get_var_double(ncid, var_coeffs, flat.data());
            if (nc_status != NC_NOERR) {
                fprintf(stderr, "two_fluid_read_rates: read 'hydrogen_excitation_rate' failed: %s\n",
                        nc_strerror(nc_status));
                nc_close(ncid);
                return;
            }
        }


        //Range T_range = pw::Range(0, nsamps-1);
        //Range n_start = pw::Range(0, static_cast<LARE::T_indexType>(nstarts ));
        //Range n_final = pw::Range(0, static_cast<LARE::T_indexType>(nfinals ));
        //svManager.allocate(plasma_source.hydrogen_excitation_rate, T_range, n_start, n_final);

        //manager.allocate(plasma_source.hydrogen_excitation_rate,
        //                 portableWrapper::Range(0, static_cast<LARE::T_indexType>(nsamps - 1)),
        //                 portableWrapper::Range(0, static_cast<LARE::T_indexType>(nstarts - 1)),
        //                 portableWrapper::Range(0, static_cast<LARE::T_indexType>(nfinals - 1)));
        for (size_t sample = 0; sample < nsamps; ++sample) {
            for (size_t start = 0; start < nstarts; ++start) {
                for (size_t final = 0; final < nfinals; ++final) {
                    const size_t idx = (sample * nstarts + start) * nfinals + final;
                    plasma_source.hydrogen_excitation_rate(static_cast<LARE::T_indexType>(sample),
                                static_cast<LARE::T_indexType>(start),
                                static_cast<LARE::T_indexType>(final)) = flat[idx];
                    //fprintf(stdout, "Rates read successfully \n start: %zu \n final: %zu \n sample: %zu \n %e \n",
            //start, final,sample ,plasma_source.hydrogen_excitation_rate(static_cast<LARE::T_indexType>(sample),
               //                 static_cast<LARE::T_indexType>(start),
               //                 static_cast<LARE::T_indexType>(final)));
                }
            }
        }
        
        nc_close(ncid);
        fprintf(stdout, "Rates read successfully \n Number of samples: %zu \n Number of coefficients: %zu \n",
            nsamps, nstarts * nfinals);
        return;
    }

////////////////////////////////////////////////////////////////////////////////////////
//Routine for interpolating the rates
    DEVICEPREFIX INLINE LARE::T_dataType interpolate_rates(const data_two_fluid_source &plasma_source, LARE::T_dataType temperature,
                             LARE::T_indexType lower_level, LARE::T_indexType upper_level){

        if (lower_level < 0 || upper_level < 0 || lower_level >= upper_level) {
            return 0.0;
        }

        const LARE::T_indexType nsamps = plasma_source.grid_logT.getSize(0);
        const LARE::T_indexType nstarts = plasma_source.hydrogen_excitation_rate.getSize(1);
        const LARE::T_indexType nfinals = plasma_source.hydrogen_excitation_rate.getSize(2);

        if (nsamps <= 0 || nstarts <= 0 || nfinals <= 0) {
            return 0.0;
        }

        if (lower_level >= nstarts || upper_level >= nfinals) {
            return 0.0;
        }

        const LARE::T_indexType lb = plasma_source.grid_logT.getLowerBound(0);
        const LARE::T_indexType ub = plasma_source.grid_logT.getUpperBound(0);
        if (ub <= lb) {
            return plasma_source.hydrogen_excitation_rate(lb, lower_level, upper_level);
        }

        const LARE::T_dataType logT = std::log10(temperature);
        
        const LARE::T_dataType logT_min = plasma_source.grid_logT(lb);
        const LARE::T_dataType logT_max = plasma_source.grid_logT(ub);
        if (logT <= logT_min) {
            return plasma_source.hydrogen_excitation_rate(lb, lower_level, upper_level);
        }
        if (logT >= logT_max) {
            return plasma_source.hydrogen_excitation_rate(ub, lower_level, upper_level);
        }

        LARE::T_indexType i0 = lb;
        for (LARE::T_indexType i = lb; i < ub; ++i) {
            if (plasma_source.grid_logT(i) <= logT && logT < plasma_source.grid_logT(i + 1)) {
                i0 = i;
                break;
            }
        }

        const LARE::T_dataType logT0 = plasma_source.grid_logT(i0);
        const LARE::T_dataType logT1 = plasma_source.grid_logT(i0 + 1);
        if (logT1 <= logT0) {
            return plasma_source.hydrogen_excitation_rate(i0, lower_level, upper_level);
        }

        const LARE::T_dataType t = (logT - logT0) / (logT1 - logT0);
    const LARE::T_dataType v0 = plasma_source.hydrogen_excitation_rate(i0, lower_level, upper_level);
    const LARE::T_dataType v1 = plasma_source.hydrogen_excitation_rate(i0 + 1, lower_level, upper_level);

    // Handle zeros safely
    if (v0 <= 0.0 && v1 <= 0.0) return 0.0;
    if (v0 <= 0.0) return v1;
    if (v1 <= 0.0) return v0;

    const LARE::T_dataType logv0 = std::log10(v0);
    const LARE::T_dataType logv1 = std::log10(v1);
    const LARE::T_dataType logv  = logv0 + (logv1 - logv0) * t;

    return std::pow(10.0, logv);
    }

}

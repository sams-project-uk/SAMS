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

namespace TWOFLUID
{
    namespace pw = portableWrapper;
    
    struct two_fluid_properties
    {
        bool collisions=true;
        bool ion_rec_empirical=false;
        bool ion_rec_nlevel=false;
    };
    
    template<typename T_EOS>
    void PIP<T_EOS>::allocate(data_two_fluid_source &plasma_source,SAMS::harness &harness){
        //data_two_fluid_source plasma_source;
        //data_two_fluid_source neutral_source;
       
        auto &axRegistry = harness.axisRegistry;
        auto &varRegistry = harness.variableRegistry;
        
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
        
        varRegistry.fillPPArray("gm_ion", plasma_source.gm_ion);
        pw::assign(plasma_source.gm_ion, 0.0);
        varRegistry.fillPPArray("gm_rec", plasma_source.gm_rec);
        pw::assign(plasma_source.gm_rec, 0.0);
        
    };
    
    template<typename T_EOS>
    void PIP<T_EOS>::get_two_fluid_source(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){

        two_fluid_properties two_fluid_flags; //Move to source structure
        
        //if (two_fluid_flags.ion_rec_nlevel){        
        ////    ion_rec_rates_nlevel(data,dataNeutral);
        //}
        
        //Calculate the source terms for the two-fluid interactions
        get_collisional_source_terms(data,dataNeutral,plasma_source);
        
        //Calculate the source terms for Ionisation/recombination
        if (two_fluid_flags.ion_rec_empirical) {
            ion_rec_rates_empirical(data,dataNeutral, plasma_source);
            get_ion_rec_source_terms(data,dataNeutral,plasma_source);
        };

    };
template<typename T_EOS>
    void PIP<T_EOS>::apply_two_fluid_source(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){

        two_fluid_properties two_fluid_flags; //Move to source structure
        
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
        
        if (two_fluid_flags.collisions){
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
                dataNeutral.energy_ion(ix,iy,iz)+=0.5*data.dt*plasma_source.source_energy_n(ix,iy,iz);                 
            }, Range(-1,data.nx), Range(-1,data.ny), Range(-1,data.nz));
        }
        
    };
};
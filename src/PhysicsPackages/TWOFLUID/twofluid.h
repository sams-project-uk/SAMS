/*
* This is a two-fluid routine for all the multi-fluid physics modules
*/

#ifndef TWOFLUID_H
#define TWOFLUID_H

#include <cstdint>
#include <cassert>
#include <string>
#include "constants.h"
#include "pp/parallelWrapper.h"
#include "mpiManager.h"
#include "variableDef.h"
#include "harness.h"
#include "runner.h"
#include "io/writerProto.h"

#include "shared_data.h"
#include "variableRegistry.h"
#include "axisRegistry.h"

#include "LARECommon/types.h"
#include "LARECommon/eos.h"
#include "LARE/shared_data.h"
#include "LARE3DSingleTemperature/shared_data.h"
#include "LARE3DNeutralFluid/shared_data.h"

namespace TWOFLUID
{
    namespace pw = portableWrapper;


    struct data_two_fluid_source
    {
        LARE::volumeArray source_mass; // mass source term
        LARE::volumeArray source_v_x; // velocity source term
        LARE::volumeArray source_v_y; // velocity source term
        LARE::volumeArray source_v_z; // velocity source term
        LARE::volumeArray source_energy; // energy source term
        LARE::volumeArray source_electron_energy; // energy source term
        
        LARE::volumeArray source_mass_n; // mass source term
        LARE::volumeArray source_v_x_n; // velocity source term
        LARE::volumeArray source_v_y_n; // velocity source term
        LARE::volumeArray source_v_z_n; // velocity source term
        LARE::volumeArray source_energy_n; // energy source term
        
        LARE::volumeArray gm_ion; //ionisation rate
        LARE::volumeArray gm_rec; //recombination rate
        LARE::volumeArray ion_loss; //ionisation loss term
        
        LARE::volumeArray ac; //coupling coeficient
        LARE::T_dataType two_fluid_timestep; //timestep

        LARE::T_dataType alpha0; // Coupling coefficient
        
        LARE::volumeArray4D level_populations; //4D array of level populations per cell
        LARE::volumeArray5D level_rates; //5D array of all the rates per cell
        
        std::string data_path = "./data/atomic_rates_2.nc";
        LARE::hostLineArray grid_logT;
        LARE::hostVolumeArray hydrogen_excitation_rate;
        
        //Physical parameters
        SAMS::T_dataType  T0=10000.0;//data.T_reference; //Reference temperature
        SAMS::T_dataType  n0=1.0e16;//data.ne_reference; //Reference electron number density
        SAMS::T_dataType  t_ir=1.0e-5; //Reference recombination timescale (relative to collisional timescale)
        SAMS::T_dataType kb_ev=8.617333e-5; //Kb in eV/K
       
        //Coupling physics
        bool collisions=true;
        bool ion_rec_empirical=false;
        bool ion_rec_nlevel=false;
    };
 
    using idealGas = LARE::idealGas;
    using T_indexType = SAMS::T_indexType;
    using T_dataType = SAMS::T_dataType;

    template<typename T_EOS=idealGas>
    class PIP
    {
        public:
            // Blocks compilation if the Equation Of State is not idealGas
            static_assert(std::is_same_v<T_EOS, idealGas>);
        
            static constexpr std::string_view name = "PIP";
            
            using dataPack = SAMS::dataPacks::multiPack<data_two_fluid_source>;
            
            using T_dataType = SAMS::T_dataType;

            void initialize(LARE::LARE3DST<T_EOS>::simulationData &data, LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){};
            void defaultValues(data_two_fluid_source & plasma_source);
            void allocate(data_two_fluid_source &plasma_source,SAMS::harness &harness);
            void registerVariables(SAMS::harness &harness);
            
            void initialiseSource(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){
                printf("Getting IC for two_fluid rates \n");
                get_ac(data,dataNeutral,plasma_source);
                get_two_fluid_source(data,dataNeutral,plasma_source);
            }
            
            void getVariables(data_two_fluid_source &plasma_source,SAMS::harness &harness){
                allocate(plasma_source, harness);
            }
            void beforeStartOfTimestep(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){
                get_ac(data,dataNeutral,plasma_source); //These might not be needed
                get_two_fluid_source(data,dataNeutral,plasma_source);
                //apply_two_fluid_source(data,dataNeutral,plasma_source);
            };
            
            void applySourceTermsStart(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){
                apply_two_fluid_source(data,dataNeutral,plasma_source);
            };

            void afterEndOfTimestep(LARE::LARE3DST<T_EOS>::simulationData &data, LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){
                get_ac(data,dataNeutral,plasma_source);
                get_two_fluid_source(data,dataNeutral,plasma_source);
                apply_two_fluid_source(data,dataNeutral,plasma_source); 
            };

            void calculateTimestep(SAMS::timeState &timeData,LARE::LARE3DST<T_EOS>::simulationData &data, LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){
                //get_ac(data,dataNeutral,plasma_source);
                //get_two_fluid_source(data,dataNeutral,plasma_source);
                set_dt_collisional(data,dataNeutral,plasma_source);
                printf("two_fluid timestep = %f \n",plasma_source.two_fluid_timestep);
                //set_dt(data);
                timeData.dt = plasma_source.two_fluid_timestep<timeData.dt ? plasma_source.two_fluid_timestep : timeData.dt;
            };
            void getTimestep(SAMS::timeState &timeData, LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral){
                //data.dt = timeData.dt;
            };
            void get_two_fluid_source(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source);
            
            void apply_two_fluid_source(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source);

            void get_ac(LARE::LARE3DST<T_EOS>::simulationData &data, LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source);
            
            void set_dt_collisional(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source);
            
            void plasma_source_allocate(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source);
            
            void get_equilibrium_ion_fraction(T_dataType T0,T_dataType &xi_n);
            
            template<typename T_writer>
            void writeOutputMeshes(writer<T_writer> &writer, LARE::LARE3DST<T_EOS>::simulationData &data);
        
            template<typename T_writer>
            void registerOutputMeshes(writer<T_writer> &writer, data_two_fluid_source &data);
            
            template<typename T>
            void registerOutputVariables(writer<T> &writer, data_two_fluid_source &plasma_source);

            template<typename T>
            void writeOutputVariables(writer<T> &writer, data_two_fluid_source &plasma_source);

            //void allocate_neutral(SAMS::harness &harness, LARE::LARE3DST<T_EOS>::simulationData &data);
        void ion_rec_rates_empirical(LARE::LARE3DST<T_EOS>::simulationData &data, LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source);
        void ion_rec_rates_nlevel(LARE::LARE3DST<T_EOS>::simulationData &data, LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source);
        void get_collisional_source_terms(LARE::LARE3DST<T_EOS>::simulationData &data, LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source);
        void get_ion_rec_source_terms(LARE::LARE3DST<T_EOS>::simulationData &data, LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source);
        void two_fluid_read_rates(data_two_fluid_source &plasma_source);

    };
}

//template class TWOFLUID::PIP<LARE::idealGas>;

#define EOS_DEF(value) template class TWOFLUID::PIP<value>;
EOS_DENSITY_ENERGY
#undef EOS_DEF

#endif // TWOFLUID_H

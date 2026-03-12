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
#include "remapData.h"
#include "typedefs.h"
#include "mpiManager.h"
#include "variableDef.h"
#include "harness.h"
#include "runner.h"
#include "io/writerProto.h"

#include "shared_data.h"
#include "variableRegistry.h"
#include "axisRegistry.h"

#include "constants_neutral.h"
#include "remapData_neutral.h"
#include "shared_data_neutral.h"
#include "typedefs_neutral.h"

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
        
        LARE::volumeArray ac; //coupling coeficient
        LARE::T_dataType two_fluid_timestep; //timestep 
    };
    
    class PIP
    {
        private:
            //SAMS::harness &harness;


        public:
        
            static constexpr std::string_view name = "PIP";
            
            using dataPack = SAMS::dataPacks::multiPack<LARE::simulationData, LARE::remapData, LARE_neutral::simulationData, LARE_neutral::remapData,data_two_fluid_source>;
            
            //pw::portableArrayManager& manager;

            /**
             * Initialize the simulation. Called by the runner at the start of the simulation.
             */
            void initialize(LARE::simulationData &data,LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source){
                //plasma_source_allocate(data,dataNeutral,plasma_source);
            };
            //}


            void allocate(LARE::simulationData &data,LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source,SAMS::harness &harness);
            void registerVariables(SAMS::harness &harness);
            
            void defaultVariables(LARE::simulationData &data,LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source,SAMS::harness &harness){
                allocate(data,dataNeutral,plasma_source,harness);
            }
            
            
            /**
             * Register axes with the harness's axis registry.
             * Called by the runner before registering variables.
             * @param harnessRef SAMS harness
             */
            //void registerAxes(SAMS::harness &harnessRef);

            /**
             * Register variables with the harness
             * Called by the runner after registering axes.
             * @param harnessRef SAMS harness
             */
            //void registerVariables(SAMS::harness &harnessRef);
            
            //void registerVariables(SAMS::harness &harnessRef){
            //};


            /**
             * Set default values for control parameters
             */
            void defaultValues(LARE::simulationData &data,LARE_neutral::simulationData &dataNeutral);

            /**
             * Set default values for control parameters
             */
            //void defaultVariables(LARE::simulationData &dataNeutral);

            /**
             * Get the variables needed for LARE3D
             * i.e. convert the raw memory from the variable registry into
             * the volumeArray/lineArray types used by LARE3D
             * @param harnessRef SAMS harness
             * @param data LARE3D simulation data
             */
            void getVariables(SAMS::harness &harnessRef, LARE_neutral::simulationData &dataNeutral){
                //allocate_neutral(harnessRef, dataNeutral);
                //LARE::grid(dataNeutral);
            };

            /**
             * Physics timestep functions
             * This is the predictor step of the LARE3D timestep
             * @param data LARE3D simulation data
             */
            void startOfTimestep(LARE::simulationData &data,LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source, SAMS::controlFunctions &controlFns){
                //get_ac(data,dataNeutral,plasma_source);
                //set_dt_collisional(data,dataNeutral,plasma_source);
                //two_fluid_source(data,dataNeutral);
                //lagrangian_step(data, controlFns);
            };


            /**
             * This is called at the end of the LARE3D timestep
             * @param data LARE3D simulation data
             */
            void endOfTimestep(LARE::simulationData &dataNeutral, LARE_neutral::remapData &remap_dataNeutral){
                //eulerian_remap(data, remap_data);
                //if (data.rke){
                //    energy_correction(data);
                //}
                //eta_calc(data);
            };

            /**
             * Set the timestep based on LARE3D data
             * @note This function is called in response to the control function setTimestep being called
             * by a package. The runner will NOT call this function directly.
             * @param timeData SAMS timeState data
             * @param data LARE3D simulation data
             */
            void calculateTimestep(SAMS::timeState &timeData,LARE::simulationData &data, LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source){
                get_ac(data,dataNeutral,plasma_source);
                get_two_fluid_source(data,dataNeutral,plasma_source);
                set_dt_collisional(data,dataNeutral,plasma_source);
                //printf("two_fluid timestep = %f \n",plasma_source.two_fluid_timestep);
                //set_dt(data);
                //timeData.dt = data.dt<timeData.dt ? data.dt : timeData.dt;
            };

            /**
             * Gather the timestep back after all packages have calculated it
             * @param timeData SAMS timeState data
             * @param data LARE3D simulation data
             */
            void getTimestep(SAMS::timeState &timeData, LARE_neutral::simulationData &dataNeutral){
                //data.dt = timeData.dt;
            };
            
            void halfSplitSourceStart(LARE::simulationData &data,LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source){
                //printf("applying source \n");
                apply_two_fluid_source(data,dataNeutral,plasma_source);
                //data.dt = timeData.dt;
            };
            
            void halfSplitSourceEnd(LARE::simulationData &data,LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source){
                //printf("applying source \n");
                apply_two_fluid_source(data,dataNeutral,plasma_source);
                //data.dt = timeData.dt;
            };
            
            void get_two_fluid_source(LARE::simulationData &data,LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source);
            
            void apply_two_fluid_source(LARE::simulationData &data,LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source);

            void get_ac(LARE::simulationData &data, LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source);
            
            void set_dt_collisional(LARE::simulationData &data,LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source);
            
            void plasma_source_allocate(LARE::simulationData &data,LARE_neutral::simulationData &dataNeutral, data_two_fluid_source &plasma_source);
            
            template<typename T_writer>
            void writeOutputMeshes(writer<T_writer> &writer, LARE::simulationData &data);
        
            template<typename T_writer>
            void registerOutputMeshes(writer<T_writer> &writer, data_two_fluid_source &data);
            
            template<typename T>
            void registerOutputVariables(writer<T> &writer, data_two_fluid_source &plasma_source);

            template<typename T>
            void writeOutputVariables(writer<T> &writer, data_two_fluid_source &plasma_source);

            //void allocate_neutral(SAMS::harness &harness, LARE::simulationData &data);
    };
}


#endif // TWOFLUID_H

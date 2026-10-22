/*
* This is a two-fluid routine for all the multi-fluid physics modules
*/

#ifndef TWOFLUID_H
#define TWOFLUID_H

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include "constants.h"
#include "pp/parallelWrapper.h"
#include "mpiManager.h"
#include "variableDef.h"
#include "harness.h"
#include "runner.h"
#include "io/writerProto.h"
#include <memory>
#include "petscStiffSolver.h"

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
        LARE::volumeArray source_kinetic_energy; // kinetic energy source term (for diagnostics only)
        
        LARE::volumeArray source_mass_n; // mass source term
        LARE::volumeArray source_v_x_n; // velocity source term
        LARE::volumeArray source_v_y_n; // velocity source term
        LARE::volumeArray source_v_z_n; // velocity source term
        LARE::volumeArray source_energy_n; // energy source term
        LARE::volumeArray source_kinetic_energy_n; // kinetic energy source term (for diagnostics only)
        
        LARE::volumeArray gm_ion; //ionisation rate
        LARE::volumeArray gm_rec; //recombination rate
        LARE::volumeArray ion_loss; //ionisation loss term
        LARE::volumeArray ion_heating; //heating term
        
        LARE::volumeArray ac; //coupling coeficient
        LARE::T_dataType two_fluid_timestep; //timestep

        LARE::T_dataType alpha0; // Coupling coefficient
        
        SAMS::T_dataType ref_rec; //Reference recombination rate
        
        LARE::volumeArray4D level_populations; //4D array of level populations per cell
        LARE::volumeArray5D level_rates; //5D array of all the rates per cell
        
        std::string data_path = "./data/hydrogen_atomic_rates.nc";
        LARE::hostLineArray grid_logT;
        LARE::hostVolumeArray hydrogen_excitation_rate;
        
        //pw::portableArrayManager rate_manager; //Rate manager
        int level_offset = 1;                               // atomic level number at array index 0
        std::vector<int> lower_level_map;                   // host: array_index -> level number
        std::vector<int> upper_level_map;                   // host: array_index -> level number
        LARE::hostVolumeArray collisional_excitation_rates; // [n_tsample][n_lower][n_upper]
        LARE::hostPlaneArray collisional_ionisation_rates;  // [n_tsample][n_lower]
        LARE::hostPlaneArray radiative_recombination_rates; // [n_tsample][n_lower]
        LARE::hostPlaneArray radiative_excitation_rates;    // [n_lower][n_upper]
        LARE::hostPlaneArray radiative_de_excitation_rates; // [n_lower][n_upper]
        LARE::hostLineArray radiative_ionisation_rates;     // [n_lower]
        
        //Physical parameters
        SAMS::T_dataType  T0=10000.0;//data.T_reference; //Reference temperature
        SAMS::T_dataType  n0=1.0e16;//data.ne_reference; //Reference electron number density
        SAMS::T_dataType  t_ir=1.0e-5; //Reference recombination timescale (relative to collisional timescale)
        SAMS::T_dataType kb_ev=8.617333e-5; //Kb in eV/K
       
        //Coupling physics
        bool collisions=true;
        bool ion_rec_empirical=true;
        bool ion_rec_nlevel=false;
        
        bool vertex_rates=false;
        LARE::volumeArray rho_p_ac_vertex; // velocity source at vertex
        LARE::volumeArray rho_n_ac_vertex; // velocity source at vertex
        LARE::volumeArray gm_ion_vertex; //
        LARE::volumeArray gm_rec_vertex; // 
        
        bool check_conservation=false;
        bool check_source=false;
        
	bool implicit=true; // flag for using the implicit solvers from PETSc
        SAMS::T_dataType implicit_rtol=1.0e-6;  // relative tolerance of the PETSc time stepper (command line -ts_rtol overrides)
        SAMS::T_dataType implicit_atol=1.0e-9;  // absolute tolerance of the PETSc time stepper (command line -ts_atol overrides)
        bool implicit_limit_dt=false;           // if false the explicit collisional dt limit is not applied when implicit=true
        bool watch_state=true;                  // scan the whole state for NaN/Inf at several points of each step (cheap debugging aid)
        int watch_interval=100;                 // print a one-line trend summary every this many timesteps (needs watch_state)

        bool substepping=false;
        int substep_iter=0;
        int substep_max_nsteps=50; // set a maximum number of step. Not done yet
        SAMS::T_dataType  substep_dt=1.0;
        SAMS::T_dataType  substep_max_speedup=1000.0; //Maximum substeps. The excess decreases the timestep
        SAMS::T_dataType  substep_time=0.0; //to make sure that the tiem is correct after the substeps
        
        bool debug_twofluid_steps=false; //print statements for where the code has gotten to
    };
    
    struct oldData
    {
        LARE::volumeArray rho; // mass source term
        LARE::volumeArray vx; // velocity source term
        LARE::volumeArray vy; // velocity source term
        LARE::volumeArray vz; // velocity source term
        LARE::volumeArray energy; // energy source term
        LARE::volumeArray rho_n; // mass source term
        LARE::volumeArray vx_n; // velocity source term
        LARE::volumeArray vy_n; // velocity source term
        LARE::volumeArray vz_n; // velocity source term
        LARE::volumeArray energy_n; // energy source term
    };
 
    using idealGas = LARE::idealGas;
    using T_indexType = SAMS::T_indexType;
    using T_dataType = SAMS::T_dataType;

    template<typename T_EOS=idealGas>
    class PIP
    {
	private:
	    // ---- PETSc implicit source-term solver -------------------------------------------------
	    // All PETSc code lives in petscStiffSolver.cpp; here we only exchange plain host arrays with it.
	    // State layout (10 variables per cell, over Range(-1,n) in each direction):
	    //   +0 rho  +1 rho_n  +2..4 plasma v  +5..7 neutral v  +8 energy_ion  +9 energy_n

	    // Context handed to the right hand side callback. Refreshed before every solve.
	    struct PETScCtx {
	        PIP *pip = nullptr;
	        typename LARE::LARE3DST<T_EOS>::simulationData *data = nullptr;
	        typename LARE::LARE3DNF<T_EOS>::simulationData *dataNeutral = nullptr;
	        data_two_fluid_source *plasma_source = nullptr;
	    };

	    static constexpr int petsc_stride = 10;

	    std::shared_ptr<SAMS_PETSC::StiffSolver> stiff_solver; // created on first use
	    std::vector<double> u_buf;       // state handed to / returned from the solver
	    std::vector<double> u_start_buf; // copy of the state at the start of a solve, for rollback
	    PETScCtx petsc_ctx;

	    void ensureStiffSolver(LARE::LARE3DST<T_EOS>::simulationData &data, data_two_fluid_source &plasma_source);
	    void implicitSourceSolve(LARE::LARE3DST<T_EOS>::simulationData &data,
	                             LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral,
	                             data_two_fluid_source &plasma_source,
	                             SAMS::T_dataType dt_solve);

	    // Host-array packing in the layout above
	    void PackViewsToArray(LARE::LARE3DST<T_EOS>::simulationData &data,
	                          LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral,
	                          double *u);
	    void UnpackArrayToViews(const double *u,
	                            LARE::LARE3DST<T_EOS>::simulationData &data,
	                            LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral);
	    void PackSourcesToArray(data_two_fluid_source &plasma_source,
	                            LARE::LARE3DST<T_EOS>::simulationData &data,
	                            double *f);

	    // f = F(u): exact two-fluid sources of the trial state u
	    static void RHSAdapter(const double *u, double *f, void *user);

	    // Debugging aid: scan the whole state (ghost layer included) for NaN/Inf and keep a trend line.
	    // `time` < 0 means "not available at this hook". Defined in TwoFluid.cpp.
	    void watchState(LARE::LARE3DST<T_EOS>::simulationData &data,
	                    LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral,
	                    data_two_fluid_source &plasma_source,
	                    const char *label, double time, bool trail);

	public:
	    // Called by the runner at the end of the run; releases the PETSc objects
	    void finalize();

	    // Blocks compilation if the Equation Of State is not idealGas
            static_assert(std::is_same_v<T_EOS, idealGas>);
        
            static constexpr std::string_view name = "PIP";
            
            using dataPack = SAMS::dataPacks::multiPack<data_two_fluid_source,oldData>;
            
            using T_dataType = SAMS::T_dataType;

            void initialize(LARE::LARE3DST<T_EOS>::simulationData &data, LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){
                if (plasma_source.ion_rec_nlevel) {
                    two_fluid_read_rates(plasma_source);
                    two_fluid_test_rates(plasma_source);
                }
                };
            void defaultValues(data_two_fluid_source & plasma_source);
            void allocate(data_two_fluid_source &plasma_source,SAMS::harness &harness);
            void allocate_conserved(oldData &oldData,SAMS::harness &harness);
            void registerVariables(SAMS::harness &harness,data_two_fluid_source &plasma_source);
            
            void registerAxes(SAMS::harness &harness,data_two_fluid_source &plasma_source)
            {
                auto &axisReg = harness.axisRegistry;
                if (plasma_source.ion_rec_nlevel) {axisReg.registerLogicalAxis("species");}
            }
            
            void initialiseSource(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){
                printf("Getting IC for two_fluid rates \n");
                get_ac(data,dataNeutral,plasma_source);
                if (plasma_source.ion_rec_nlevel) {set_reference_recombination(data,plasma_source);}
                get_two_fluid_source(data,dataNeutral,plasma_source);
                if (plasma_source.ion_rec_empirical) {set_bc_heating(data,plasma_source);}
            }
            
            void getVariables(data_two_fluid_source &plasma_source,oldData &oldData, SAMS::harness &harness){
                allocate(plasma_source, harness);
                if (plasma_source.check_conservation) {allocate_conserved(oldData,harness);}
            }
            void beforeStartOfTimestep(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){
                if (plasma_source.debug_twofluid_steps) {printf("beforeStartofTimestep start \n");}
                if (plasma_source.watch_state) { watchState(data,dataNeutral,plasma_source,"start of step (after the previous step's hooks)",-1.0,false); }
                get_ac(data,dataNeutral,plasma_source); //These might not be needed
                get_two_fluid_source(data,dataNeutral,plasma_source);
                if (plasma_source.debug_twofluid_steps) {printf("beforeStartofTimestep end \n");}
                //apply_two_fluid_source(data,dataNeutral,plasma_source);
            };
            
            void applySourceTermsStart(LARE::LARE3DST<T_EOS>::simulationData &data,LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source,oldData &oldData,SAMS::timeState &timeData){
                if (plasma_source.debug_twofluid_steps) {printf("applySourceTermsStart start \n");}
                if (plasma_source.watch_state) { watchState(data,dataNeutral,plasma_source,"before the first source half-step",timeData.time,false); }
                if (plasma_source.check_conservation) {copyState(data, dataNeutral, oldData);};
                if (plasma_source.check_source) {checkSourceConservation(data,dataNeutral,plasma_source);};
                if (plasma_source.implicit) {
                    // Implicit PETSc solve of the first Strang half step
                    plasma_source.substep_dt = 0.5 * timeData.dt; // Strang splitting factor
                    implicitSourceSolve(data, dataNeutral, plasma_source, plasma_source.substep_dt);
                } else if ((plasma_source.substepping) && (plasma_source.two_fluid_timestep<timeData.dt)){
                    //set up the substepping for the two-fluid routines
                    int n_substeps=std::ceil(0.5*timeData.dt/plasma_source.two_fluid_timestep);
                    plasma_source.substep_dt=0.5*timeData.dt/n_substeps; //0.5 from strang split. Not sure if needed elsewhere
                    printf("Beginning subcycle, n, dt, dt_plasma, dt_sub %i, %.12e, %.12e, %.12e \n",n_substeps,timeData.dt, plasma_source.two_fluid_timestep,plasma_source.substep_dt);
                    for (int i = 0; i < n_substeps; i++) {
                        get_ac(data,dataNeutral,plasma_source);
                        get_two_fluid_source(data,dataNeutral,plasma_source);
                        apply_two_fluid_source(data,dataNeutral,plasma_source);
                    }
                } else {
                    //Explicit time integration
                    plasma_source.substep_dt=0.5*data.dt;//0.5 from strang split.
                    apply_two_fluid_source(data,dataNeutral,plasma_source);
                }
                if (plasma_source.debug_twofluid_steps) {printf("applySourceTermsStart end \n");}
            };

            void afterEndOfTimestep(LARE::LARE3DST<T_EOS>::simulationData &data, LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source,oldData &oldData,SAMS::timeState &timeData){
                if (plasma_source.watch_state) { watchState(data,dataNeutral,plasma_source,"after the hydro step, before the second source half-step",timeData.time,false); }
                if (plasma_source.implicit) {
                    // Implicit PETSc solve of the second Strang half step
                    plasma_source.substep_dt = 0.5 * timeData.dt; // Strang splitting factor
                    implicitSourceSolve(data, dataNeutral, plasma_source, plasma_source.substep_dt);
                } else if ((plasma_source.substepping) && (plasma_source.two_fluid_timestep<timeData.dt)){
                    //set up the substepping for the two-fluid routines
                    int n_substeps=std::ceil(0.5*timeData.dt/plasma_source.two_fluid_timestep);
                    plasma_source.substep_dt=0.5*timeData.dt/n_substeps; //0.5 from strang split. Not sure if needed elsewhere
                    printf("Second subcycle, n, dt, dt_plasma, dt_sub %i, %.12e, %.12e, %.12e \n",n_substeps,timeData.dt, plasma_source.two_fluid_timestep,plasma_source.substep_dt);
                    for (int i = 0; i < n_substeps; i++) {
                        get_ac(data,dataNeutral,plasma_source);
                        get_two_fluid_source(data,dataNeutral,plasma_source);
                        apply_two_fluid_source(data,dataNeutral,plasma_source);
                    }
                } else {
                    //Explicit time integration
                    plasma_source.substep_dt=0.5*data.dt;//0.5 from strang split.
                    get_ac(data,dataNeutral,plasma_source);
                    get_two_fluid_source(data,dataNeutral,plasma_source);
                    apply_two_fluid_source(data,dataNeutral,plasma_source);
                }
                if (plasma_source.check_conservation) {checkConservation(data,dataNeutral,oldData);};
                if (plasma_source.watch_state) { watchState(data,dataNeutral,plasma_source,"after the second source half-step",timeData.time,true); }
            };

            void calculateTimestep(SAMS::timeState &timeData,LARE::LARE3DST<T_EOS>::simulationData &data, LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral, data_two_fluid_source &plasma_source){
                //get_ac(data,dataNeutral,plasma_source);
                //get_two_fluid_source(data,dataNeutral,plasma_source);
                if (plasma_source.debug_twofluid_steps) {printf("calculateTimestep start \n");}
                if (plasma_source.implicit && !plasma_source.implicit_limit_dt) {
                    // The implicit solver is not limited by the stiff exchange timescales;
                    // the accuracy of the source step is controlled by the PETSc adaptive time stepper.
                } else {
                set_dt_collisional(data,dataNeutral,plasma_source);
                //printf("two_fluid timestep = %f \n",plasma_source.two_fluid_timestep);
                //set_dt(data);
                if ((plasma_source.substepping) && (plasma_source.two_fluid_timestep<timeData.dt)){
                    //set up the substepping for the two-fluid routines
                    timeData.dt = plasma_source.two_fluid_timestep*plasma_source.substep_max_speedup<timeData.dt ? plasma_source.two_fluid_timestep*plasma_source.substep_max_speedup : timeData.dt;
                    //int n_substeps=std::ceil(timeData.dt/plasma_source.two_fluid_timestep);
                    //plasma_source.substep_dt=timeData.dt/n_substeps;
                    //IS THIS ALWAYS CALCULATED AFTER THE FLUID TIMESTEP?
                } else {
                    timeData.dt = plasma_source.two_fluid_timestep<timeData.dt ? plasma_source.two_fluid_timestep : timeData.dt;
                }
                } // end explicit dt limit
                if (timeData.time<=1.0e-6) {
                    timeData.dt = 1.0e-8<timeData.dt ? 1.0e-8 : timeData.dt;
                }
                if (plasma_source.debug_twofluid_steps) {printf("calculateTimestep end \n");}
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
        void two_fluid_test_rates(const data_two_fluid_source &plasma_source);
        void set_bc_heating(LARE::LARE3DST<T_EOS>::simulationData &data,data_two_fluid_source &plasma_source);
        void set_reference_recombination(LARE::LARE3DST<T_EOS>::simulationData &data,data_two_fluid_source &plasma_source);

        void checkSourceConservation(
                LARE::LARE3DST<T_EOS>::simulationData &data,
                LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral,
                data_two_fluid_source &plasma_source);
        void checkConservation(
                LARE::LARE3DST<T_EOS>::simulationData &data,
                LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral,
                oldData &oldData);
        void copyState(
            LARE::LARE3DST<T_EOS>::simulationData &data,
            LARE::LARE3DNF<T_EOS>::simulationData &dataNeutral,
            oldData &oldData);
            
        void debug_rates(data_two_fluid_source& ps, const char* name)
{

printf("DEBUG: %s\n", name);
    printf("first  = %.16e\n",
           ps.collisional_ionisation_rates(0,0));

    printf("mid1   = %.16e\n",
           ps.collisional_ionisation_rates(50,0));

    printf("mid2   = %.16e\n",
           ps.collisional_ionisation_rates(50,4));

    printf("last   = %.16e\n",
           ps.collisional_ionisation_rates(100,4));
           
   
printf("data ptr = %p\n",
       ps.collisional_ionisation_rates.data());

printf("%p\n",
       &ps.collisional_ionisation_rates(50,4));
       
    }
    };
}

//template class TWOFLUID::PIP<LARE::idealGas>;

#define EOS_DEF(value) template class TWOFLUID::PIP<value>;
EOS_DENSITY_ENERGY
#undef EOS_DEF

#endif // TWOFLUID_H

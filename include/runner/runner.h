#ifndef SAMS_RUNNER_H
#define SAMS_RUNNER_H

#include "runnerUtils.h"
#include "pp/callableTraits.h"
#include "io/writerProto.h"
#include "io/writerHDF5.h"
#include "io/writerSimple.h"
#include "timer.h"
#include "welcome.h"
#include <tuple>
#include <sstream>

namespace SAMS
{
#include "runnerMacros.h"    
/**
 * Struct to hold time stepping state
 */
    struct timeState
    {
        T_dataType time = 0.0;       // Current simulation time
        T_dataType dt = 0.0;   // Current timestep size
        T_indexType step = 0;         // Current timestep number
    };

    /**
     * Class that holds type-erased control functions for simulations
     * This allows a package to call a control function on the runner without knowing the exact type of the runner
     * Performance is not brilliant, so this should NOT be used in performance-critical code
     * Things should only be added to this struct if physics packages directly need to call a function on the runner
     */
    struct controlFunctions
    {private:

        template<typename... Packages>
        friend class runner;
        void* object;
        void(*calculateTimestepFn)(void*)=nullptr;
        bool(*isActiveFn)(void*, const std::string &name)=nullptr;
        void(*activateFn)(void*, const std::string &name)=nullptr;
        void bind(void* obj){
            object = obj;
        }
        void bindCalculateTimestep(void(*calculateTimestepFunc)(void*)){
            calculateTimestepFn = calculateTimestepFunc;
        }
        void bindIsActive(bool(*isActiveFunc)(void*, const std::string &name)){
            isActiveFn = isActiveFunc;
        }
        void bindActivate(void(*activateFunc)(void*, const std::string &name)){
            activateFn = activateFunc;
        }
    public:
    /**
     * Trigger the set timestep function on the runner
     */
        void calculateTimestep(){
            calculateTimestepFn(object);
        }

    /**
     * Query whether a package is active on the runner by name
     */
        bool isPackageActive(const std::string& name){
            return isActiveFn(object, name);
        }

    /**
     * Activate a package on the runner by name
     */        
        void activatePackage(const std::string& name){
            activateFn(object, name);
        }
    };
    
    /**
     * Runner class that runs a set of simulations
     */
    template<typename... Packages>
    class runner
    {
        public:
        /** Types of requested simulations. Must come first */
        using T_packages = std::tuple<Packages...>;
        harness harnessData;
        /** Internal runner data that might be requested by simulations
         * Add new types here if they are internal state shared by multiple simulations
        */
        using T_runnerData = std::tuple<harness&, timeState, controlFunctions>;
        /**Data packs requested by simulations */
        using T_dataPacks = dataPackTuple_t<Packages...>;
        using T_combined = tupleUnion_t<T_packages, T_runnerData, T_dataPacks>;
        T_combined runnerData = initializeTuple<>(harnessData);

        struct internalPackageInfo{
            int level = -1;
            bool isCoreSimulation = false;
            bool timeSimulation = false;
            std::string name = "";
        };

        std::map<std::string, internalPackageInfo> simulationInfoMap;
        std::array<bool, sizeof...(Packages)> simulationActiveFlags; //Active isn't in the map for performance reasons
        std::vector<std::string> outputVariables;
        std::map<std::string, std::vector<std::string>> outputSets;

        FULL_CALL_X(runnerInteraction); //The one chance that a simulation has to interact directly with the runner.

        /**
         * @fn void callCore_initialize()
         * Core initialization function
         * This is called by the runner at the start of the simulation
         * It calls the initialize method on all packages if those packages implement it
         */
        CALL_X(initialize);//Initialize packages

        FULL_CALL_X(checkActivation); //Report whether this package can validly be activated
        /**
         * @fn void callCore_registerDeckElements()
         * Core function to register deck elements for all packages
         * This is called by the runner after initializing the harness but before reading the input deck
         * calls the registerDeckElements method on all packages if those packages implement it
         * @note Currently, we don't have an input deck, but this is for future use
         */
         /**
         * @fn void callCore_registerDeckElements()
         * Function to register input deck elements for all packages
         * @note This is called by the runner after initializing the harness but before reading the input deck
         */
        FULL_CALL_X(registerDeckElements); //Register input deck elements

        /**
         * 
         */
        FULL_CALL_X(registerAxes);//Register axes
        FULL_CALL_X(registerVariables); //Register variables
        FULL_CALL_X(defaultValues); //Set default parameter values
        FULL_CALL_X(controlVariables); //Setup control variables
        FULL_CALL_X(setDomain); //Set the domain for simulations

        FULL_CALL_X(getVariables); //Get variable objects from the variable registry


        FULL_CALL_X(initialConditions); //Set initial conditions
        FULL_CALL_X(setBoundaryConditions); //Set boundary conditions
        FULL_CALL_X(defaultVariables); //Set default variable values

        FULL_CALL_X(startOfTimestep); //Actions to perform at the start of each timestep
        FULL_CALL_X(halfTimestep); //Actions to perform at half timestep
        FULL_CALL_X(endOfTimestep); //Actions to perform at end of timestep
        CALL_X(calculateTimestep); //Set the timestep for simulations
        CALL_X(getTimestep);
        /**
         * Set the timestep for the simulations
         * @note This is called in response to a package calling controlFunctions::setTimestep(), not directly by the runner
         */
        void calculateTimestep(){
            timeState& tData = std::get<timeState>(runnerData);
            //Packages should compare their suggested timesteps and set the smallest one
            tData.dt = std::numeric_limits<T_dataType>::max();
            callCore_calculateTimestep();
            #ifdef USE_MPI
            SAMS::harness& h = std::get<harness&>(runnerData);
            MPI_Allreduce(MPI_IN_PLACE, &tData.dt, 1, MPI_DOUBLE, MPI_MIN, h.MPIManager.getGlobalComm());
            #endif
            //Allow the simulations to gather the updated timestep
            callCore_getTimestep();
        }

        CALL_X(queryTerminate); //Query whether to terminate the simulation
        bool queryTerminate(){
            bool terminate = false;
            callCore_queryTerminate(terminate);
#if defined(USE_MPI) && defined(ALLOW_RANK_INCONSISTENCY)
            SAMS::harness& h = std::get<harness&>(runnerData);
            int localTerm = terminate ? 1 : 0;
            MPI_Allreduce(MPI_IN_PLACE, &localTerm, 1, MPI_INT, MPI_LOR, h.MPIManager.getGlobalComm());
            terminate = (localTerm == 1);
#endif
            return terminate;
        }

        CALL_X(queryOutput); //Query whether output is needed this timestep
        bool queryOutput(){
            bool outputNow = false;
            callCore_queryOutput(outputNow);
#if defined(USE_MPI) && defined(ALLOW_RANK_INCONSISTENCY)
            SAMS::harness& h = std::get<harness&>(runnerData);
            int localOut = outputNow ? 1 : 0;
            MPI_Allreduce(MPI_IN_PLACE, &localOut, 1, MPI_INT, MPI_LOR, h.MPIManager.getGlobalComm());
            outputNow = (localOut == 1);
#endif
            return outputNow;
        }

        CALL_X(registerOutputMeshes); //Register meshes
        template<typename T>
        void registerOutputMeshes(writer<T>& w){
            callCore_registerOutputMeshes<false, true,true,0,T>(w);
        }

        CALL_X(registerOutputVariables); //Register variables
        template<typename T>
        void registerOutputVariables(writer<T>& w){
            callCore_registerOutputVariables<false, true,true,0,T>(w);
        } 

        CALL_X(writeOutputMeshes); //Write output
        template<typename T>
        void writeOutputMeshes(writer<T>& w){
            callCore_writeOutputMeshes<false,true,true,0,T>(w);
        }

        CALL_X(writeOutputVariables); //Write variables
        template<typename T>
        void writeOutputVariables(writer<T>& w){
            callCore_writeOutputVariables<false,true,true,0,T>(w);
        }

        FULL_CALL_X(steer); //Computational steering

        CALL_X(finalize); //Finalize simulations

        HAS_X_PARAM_WITH_TYPE(name, std::string); //Package name parameter
        HAS_X_PARAM_WITH_TYPE(coreSimulation, bool); //Is this a core simulation
        HAS_X_PARAM_WITH_TYPE(timeSimulation, bool); //Should this simulation be timed

        public:

        /**
         * Populate a single simulation info struct
         */
        template<int level=0>
        internalPackageInfo buildPackageInfo(){
            internalPackageInfo info;
            info.name = hasParamType_name<std::tuple_element_t<level, T_combined>>::value;
            info.level = level;
            if constexpr (std::is_void_v<typename hasParamType_coreSimulation<std::tuple_element_t<level, T_combined>>::type> == false){
                info.isCoreSimulation = hasParamType_coreSimulation<std::tuple_element_t<level, T_combined>>::value;
            } else {
                info.isCoreSimulation = false;
            }
            if constexpr (std::is_void_v<typename hasParamType_timeSimulation<std::tuple_element_t<level, T_combined>>::type> == false){
                info.timeSimulation = hasParamType_timeSimulation<std::tuple_element_t<level, T_combined>>::value;
            } else {
                info.timeSimulation = false;
            }
            return info;
        }

        /**
         * Build the simulation info map
         */
        template<int level=0>
        void buildInfo(){
            if constexpr (std::is_void_v<typename hasParamType_name<std::tuple_element_t<level, T_combined>>::type> == false){
                std::string name = static_cast<std::string>(hasParamType_name<std::tuple_element_t<level, T_combined>>::value);
                simulationInfoMap[name] = buildPackageInfo<level>();
            } else {
                std::stringstream ss;
                ss << "ERROR: Package at index " << level << " does not have a name parameter defined.\n";
                abort(ss.str(), false);
            }
            if constexpr (level < sizeof...(Packages)-1){
                buildInfo<level+1>();
            }
        }

        /**
         * Function to initialize the combined tuple of simulations, runner data and data packs
         */
        template<int level=0>
        static auto initializeTuple(harness& h){
           constexpr std::size_t N = std::tuple_size_v<T_combined>;
            using T_current = std::tuple_element_t<level, T_combined>;

            //Case where the tuple element is just default constructible
            if constexpr (std::is_default_constructible_v<T_current>) {
                if constexpr (level + 1 < N) {
                    return std::tuple_cat(
                        std::make_tuple(T_current{}),
                        initializeTuple<level + 1>(h)
                    );
                } else {
                    return std::make_tuple(T_current{});
                }
                //Case where the tuple element is a reference to the harness
            } else if constexpr (std::is_reference_v<T_current>) {
                if constexpr (level + 1 < N) {
                    return std::tuple_cat(
                        std::tuple<T_current>(h),
                        initializeTuple<level + 1>(h)
                    );
                } else {
                    return std::tuple<T_current>(h);
                }
                //Case where the tuple element is constructible from a harness reference
            } else if constexpr (std::is_constructible_v<T_current, harness&>) {
                if constexpr (level + 1 < N) {
                    return std::tuple_cat(
                        std::make_tuple(T_current{h}),
                        initializeTuple<level + 1>(h)
                    );
                } else {
                    return std::make_tuple(T_current{h});
                }
            } else {
                static_assert(portableWrapper::alwaysFalse<T_current>::value,
                              "Error: Package or data pack type is not default constructible or constructible from a harness reference.");
            }
        }

        /**
         * Static function to call setDt on a runner instance
         * Used in type-erased control function binding
         * @param obj Pointer to the runner instance
         */
        void static callCalculateTimestep(void* obj){
            runner* r = static_cast<runner*>(obj);
            r->calculateTimestep();
        }

        /**
         * Static function to call isPackageActive on a runner instance
         * Used in type-erased control function binding
         * @param obj Pointer to the runner instance
         * @param name Name of the package to query
         * @return Whether the package is active
         */
        bool static callIsPackageActive(void* obj, const std::string& name){
            runner* r = static_cast<runner*>(obj);
            return r->isPackageActive(name);
        }

        /**
         * Static function to call activatePackage on a runner instance
         * Used in type-erased control function binding
         * @param obj Pointer to the runner instance
         * @param name Name of the package to activate
         */
        static void callActivatePackage(void* obj, const std::string& name){
            runner* r = static_cast<runner*>(obj);
            r->activatePackage(name);
        }

        public:

        /**
         * Abort the runner, dealing with MPI as needed
         */
        void abort(const std::string& message = "", bool localError = false){
            getHarness().abort(message, localError);
        }

        /**
         * Is T a simulation type managed by this runner
         * @tparam T The type to check
         */
        template<typename T>
        constexpr bool isPackageType(){
            constexpr int64_t index = tupleTypeIndex_v<T, T_combined>;
            if constexpr (index < 0 || index >= sizeof...(Packages)){
                return false;
            } else {
                return true;
            }
        }

        /**
         * Is T a runner data type managed by this runner
         * @tparam T The type to check
         */
        template<typename T>
        constexpr bool isRunnerDataType(){
            constexpr int64_t index = tupleTypeIndex_v<T, T_combined>;
            if constexpr (index < 0 || index < sizeof...(Packages) || index >= sizeof...(Packages) + std::tuple_size_v<T_runnerData>){
                return false;
            } else {
                return true;;
            }
        }

        /**
         * is T a data pack type managed by this runner
         * @tparam T The type to check
         */
        template<typename T>
        constexpr bool isDataPackType(){
            constexpr int64_t index = tupleTypeIndex_v<T, T_combined>;
            if constexpr (index < 0 || index >= sizeof...(Packages) + std::tuple_size_v<T_runnerData>){
                return false;
            } else {
                return true;
            }
        }

        /**
         * Get a physics package from the runner
         * @tparam T_sim The simulation type
         * @return A reference to the simulation
         */
        template<typename T_sim>
        T_sim& getPackage()
        {
            static_assert(isPackageType<T_sim>(), "Error: T_sim is not a package type managed by this runner.");
            return std::get<T_sim>(runnerData);
        }

        /**
         * Get the harness from the runner
         * @return A reference to the harness
         */
        harness& getHarness()
        {
            return std::get<harness&>(runnerData);
        }

        /**
         * Get a data pack from the runner
         * @tparam T_dataPack The data pack type
         * @return A reference to the data pack
         */
        template<typename T_dataPack>
        T_dataPack& getDataPack()
        {
            static_assert(isDataPackType<T_dataPack>(), "Error: T_dataPack is not a data pack type managed by this runner.");
            return std::get<T_dataPack>(runnerData);
        }


        // Activate a simulation by name
        void activatePackage(const std::string& simName)
        {
            auto it = simulationInfoMap.find(simName);
            if (it != simulationInfoMap.end()){
                simulationActiveFlags[it->second.level] = true;
            } else {
                std::stringstream ss;
                ss << "ERROR: Package with name " << simName << " not found in runner.\n";
                abort(ss.str(), false);
            }
        }

        bool isPackageActive(const std::string& simName){
            auto it = simulationInfoMap.find(simName);
            if (it != simulationInfoMap.end()){
                return simulationActiveFlags[it->second.level];
            } else {
                std::stringstream ss;
                ss << "ERROR: Package with name " << simName << " not found in runner.\n";
                throw std::runtime_error(ss.str());
            }
        }

        void decomposeAndAllocate()
        {
            harness& h = getHarness();
            auto& mpi = h.MPIManager;
            auto& varReg = h.variableRegistry;
            mpi.decomposeAllAxes();
            //Allocate variables
            varReg.allocateAll();
        }


        //Iniitialize all active simulations
        void pkgInit()
        {
            buildInfo();
            static_assert(sizeof...(Packages) > 0, "Error: You cannot compile SAMS with zero packages.");
            int coreCount = 0;
            bool haveActive = false;
            for (const auto& [name, info] : simulationInfoMap){
                if (simulationActiveFlags[info.level]){
                    haveActive = true;
                    if (info.isCoreSimulation){
                        coreCount++;
                    }
                }
            }
            if (!haveActive){
                abort("Error: No active simulations in runner. At least one simulation must be activated before initialization.", false);
            }
            if (coreCount == 0){
                abort("Error: No core simulations activated. At least one core simulation must be activated before initialization.", false);
            }
            if (coreCount > 1){
                SAMS::cerr << "Warning: Multiple core simulations activated. This may be an error depending on your use case." << std::endl;
            }
            //Now loop over and print the active simulations
            SAMS::cout << "Active packages in runner:" << std::endl;
            for (const auto& [name, info] : simulationInfoMap){
                if (simulationActiveFlags[info.level]){
                    SAMS::cout << " - " << colorSet(VT100Green) << name ;
                    if (info.isCoreSimulation){
                        SAMS::cout << " (Core Simulation)";
                    }
                    SAMS::cout << colorSet(VT100Reset) << std::endl;
                }
            }
            callCore_initialize();
        }

        void writeOutput(){
            static SAMS::T_sizeType outputCount = 0;
            //Create writer
    #if defined(USE_HDF5)
            HDF5File writer;
    #else
            simpleFile writer;
    #endif
            //Convert step number to a 5-digit padded string
            std::stringstream ss;
            ss << std::setw(5) << std::setfill('0') << outputCount;

            std::string Name = "diagnostics_" + ss.str();
            writer.openFile(Name.c_str());
            registerOutputMeshes(writer);
            registerOutputVariables(writer);
            writeOutputMeshes(writer);
            writeOutputVariables(writer);
            writer.closeFile();
            outputCount++;
        }

        //CODE HERE AND BELOW IS TO ACTUALLY RUN THE PACAKGES


        //Initialise the runner
        void initialize(int &argc, char** &argv)
        {
            //Initialize the harness
            getHarness().initialize(argc, argv);
            //This Must be changed so that there is more control over decomposition
            getHarness().MPIManager.autoDecomposition({false,false,false});
            //Bind the control functions
            auto& ctrlFuncs = SAMS::getItemFromTuple<controlFunctions>(runnerData);
            ctrlFuncs.bind(this);
            ctrlFuncs.bindCalculateTimestep(&runner::callCalculateTimestep);
            ctrlFuncs.bindIsActive(&runner::callIsPackageActive);
            ctrlFuncs.bindActivate(&runner::callActivatePackage);
            //Turn off all simulations
            std::fill(simulationActiveFlags.begin(), simulationActiveFlags.end(), false);
            buildInfo();
        }

        void initializePackages(){
            runnerInteraction<runner>(*this); //Allow user to interact with the runner before initialization (e.g. to activate simulations)
            pkgInit(); //Initialize active packages
            registerAxes(); //Tell packages to register axes
            registerVariables(); //Tell packages to register variables (using axes)
            defaultValues(); //Set default parameter values
            controlVariables(); //Set specific parameter values
            setDomain(); //Set the domain for simulations
            decomposeAndAllocate(); //MPI decompose and allocate variables
            getVariables(); //Get the actual memory for variables
            defaultVariables(); //Set default variable values
            initialConditions(); //Set initial conditions
            setBoundaryConditions(); //Attach boundary conditions
            writeOutput(); //Write initial output
        }

        void runPackages(){
            for (;;){
                size_t step = std::get<timeState>(runnerData).step;
                if (step%10 == 0){
                    SAMS::cout << "Starting timestep " << step << " at time " << std::get<timeState>(runnerData).time <<  ", dt = " << std::get<timeState>(runnerData).dt << std::endl;
                }
                startOfTimestep(); //Start of timestep (predictor)
                halfTimestep(); //Half timestep (correction)
                auto& tData = std::get<timeState>(runnerData);
                tData.step++;
                tData.time += tData.dt;
                endOfTimestep(); //End of timestep (remap for LARE3D)
                if (queryOutput()){
                    writeOutput(); //If ANY package says to output, do so
                }                    
                if (queryTerminate()){// If ANY package says to terminate, do so
                    break;
                }
                steer(); //Computational steering
            }
            writeOutput(); //Final output
            SAMS::cout << "Simulation complete. Time stepping ended at step " << std::get<timeState>(runnerData).step << " and time " << std::get<timeState>(runnerData).time << std::endl;
        }

        void finalizePackages(){
            callCore_finalize();
        }

        //User now activates simulations as needed

        void finalize(){
            printTimer_initialize();
            printTimer_registerAxes();
            printTimer_registerVariables();
            printTimer_controlVariables();
            printTimer_setDomain();
            printTimer_getVariables();
            printTimer_initialConditions();
            printTimer_startOfTimestep();
            printTimer_halfTimestep();
            printTimer_endOfTimestep();
            printTimer_calculateTimestep();

            harness& h = getHarness();
            h.finalize();
        }

    };
    
} // namespace SAMS


#endif // SAMS_RUNNER_H
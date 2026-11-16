#ifndef SAMS_RUNNER_H
#define SAMS_RUNNER_H

#include "KVS/kvs.h"
#include "runnerUtils.h"
#include "commonNames.h"
#include "pp/callableTraits.h"
#include "io/writerProto.h"
#include "io/writers.h"
#include "io/readers.h"
#include "io/simpleHelpers.h"
#ifdef USE_DECK_PARSER
#include "EISpp/eis.h"
#include "deckParserHelpers.h"
#endif
#include "timer.h"
#include "welcome.h"
#include <tuple>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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
 * Struct to hold the locations of the files of the run
 */
    struct fileInformation
    {
        std::string deckDirectory = ".";   // Directory of the input deck. Relative paths in the deck are relative to it
        std::string outputDirectory = "."; // Directory that the output is written to
        std::string restartPath;           // Path of the dump that the run restarts from. Empty if it does not restart
    };

#ifdef USE_DECK_PARSER
    //Capability bit of deck expressions that use the time
    inline constexpr eis::bitmask capTime = 1;
    //Capability bit of deck expressions that use field variables
    inline constexpr eis::bitmask capVariables = 2;
#endif

    /**
     * How a package was activated
     */
    enum class activationSource { none, defaultActivated, deck, package };

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
        void(*deactivateFn)(void*, const std::string &name)=nullptr;
        activationSource(*sourceFn)(void*, const std::string &name)=nullptr;
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
        void bindDeactivate(void(*deactivateFunc)(void*, const std::string &name)){
            deactivateFn = deactivateFunc;
        }
        void bindSource(activationSource(*sourceFunc)(void*, const std::string &name)){
            sourceFn = sourceFunc;
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

    /**
     * Deactivate a package on the runner by name
     */
        void deactivatePackage(const std::string& name){
            deactivateFn(object, name);
        }

    /**
     * How a package on the runner was activated, by name
     */
        activationSource getActivationSource(const std::string& name){
            return sourceFn(object, name);
        }
    };

    /**
     * Handle to an output that a package has registered. Only the output registry makes valid handles
     */
    class outputHandle
    {
        friend class outputRegistry;
        friend class outputLevel;
        friend class outputRequest;
        size_t index = std::numeric_limits<size_t>::max();
        explicit outputHandle(size_t i) : index(i) {}
    public:
        /**
         * A handle that is not to any output, so that a package can hold one before it registers the output
         */
        outputHandle() = default;

        bool operator==(const outputHandle &other) const { return index == other.index; }
        bool operator!=(const outputHandle &other) const { return index != other.index; }
    };

    /**
     * The outputs that packages have registered, which the deck can select
     */
    class outputRegistry
    {
        /**
         * What a package registers about an output
         */
        struct outputInfo
        {
            std::string name;
            std::string description;
            bool includedByDefault;
            bool neededForRestart;
        };

        std::vector<outputInfo> outputs;

        /**
         * A named collection of outputs, which can include other collections
         */
        struct outputCollection
        {
            std::string name;
            //The names of the outputs that the collection names itself
            std::vector<std::string> outputs;
            //The names of the collections that the collection includes
            std::vector<std::string> includes;
            //Whether each registered output is in the collection, once the includes are resolved
            std::vector<bool> members;
        };

        std::vector<outputCollection> collections;

        /**
         * Find a collection by name
         * @throws std::invalid_argument if no collection has that name
         */
        size_t findCollection(const std::string &name) const
        {
            for (size_t i = 0; i < collections.size(); ++i){
                if (collections[i].name == name) return i;
            }
            throw std::invalid_argument("There is no collection called " + name);
        }

        /**
         * Resolve the members of a collection and of every collection that it includes
         * @param i The index of the collection
         * @param state For each collection, 0 if it is not resolved, 1 if it is being resolved and 2 if it is resolved
         * @throws std::invalid_argument if a collection names an output or a collection that does not exist, or includes itself
         */
        void resolveCollection(size_t i, std::vector<int> &state)
        {
            if (state[i] == 2) return;
            if (state[i] == 1) throw std::invalid_argument("Collection " + collections[i].name + " includes itself");
            state[i] = 1;
            outputCollection &collection = collections[i];
            collection.members.assign(outputs.size(), false);
            for (const std::string &outputName : collection.outputs){
                collection.members[getHandle(outputName).index] = true;
            }
            for (const std::string &includeName : collection.includes){
                const size_t j = findCollection(includeName);
                resolveCollection(j, state);
                for (size_t k = 0; k < outputs.size(); ++k){
                    if (collections[j].members[k]) collection.members[k] = true;
                }
            }
            state[i] = 2;
        }

    public:
        /**
         * Register an output
         * @param name The name of the output, which the deck uses
         * @param description A description of the output
         * @param includedByDefault Whether the output is written when the deck does not say which outputs to write
         * @param neededForRestart Whether the output is always written to restarts
         * @return The handle of the output. If an output with the same name is already registered, it is that output, with the settings that it was registered with
         */
        outputHandle registerOutput(const std::string &name, const std::string &description = "", bool includedByDefault = true, bool neededForRestart = true)
        {
            for (size_t i = 0; i < outputs.size(); ++i){
                if (outputs[i].name == name){
                    SAMS::debug1 << "Warning: output " << name << " is already registered. The existing registration is used" << std::endl;
                    return outputHandle(i);
                }
            }
            outputs.push_back({name, description, includedByDefault, neededForRestart});
            return outputHandle(outputs.size() - 1);
        }

        /**
         * Get the handle of a registered output by name
         * @param name The name of the output
         * @return The handle of the output
         * @throws std::invalid_argument if no output has that name
         */
        outputHandle getHandle(const std::string &name) const
        {
            for (size_t i = 0; i < outputs.size(); ++i){
                if (outputs[i].name == name) return outputHandle(i);
            }
            throw std::invalid_argument("There is no output called " + name);
        }

        /**
         * Add a collection of outputs. A collection that is already defined is merged with the new definition, with a warning
         * @param name The name of the collection
         * @param outputNames The names of the outputs in the collection
         * @param includes The names of the collections that the collection includes
         */
        void addCollection(const std::string &name, const std::vector<std::string> &outputNames, const std::vector<std::string> &includes = {})
        {
            for (outputCollection &collection : collections){
                if (collection.name == name){
                    SAMS::cerr << "Warning: collection " << name << " is defined more than once. The definitions are merged" << std::endl;
                    collection.outputs.insert(collection.outputs.end(), outputNames.begin(), outputNames.end());
                    collection.includes.insert(collection.includes.end(), includes.begin(), includes.end());
                    return;
                }
            }
            collections.push_back({name, outputNames, includes, {}});
        }

        /**
         * Add a collection of outputs by their handles. A collection that is already defined is merged with the new definition, with a warning
         * @param name The name of the collection
         * @param outputHandles The handles of the outputs in the collection
         * @param includes The names of the collections that the collection includes
         */
        void addCollection(const std::string &name, const std::vector<outputHandle> &outputHandles, const std::vector<std::string> &includes = {})
        {
            std::vector<std::string> outputNames;
            for (const outputHandle &handle : outputHandles){
                outputNames.push_back(outputs[handle.index].name);
            }
            addCollection(name, outputNames, includes);
        }

        /**
         * Add the built in collections and resolve the members of every collection. The built in collections are
         * default, of the outputs that are included by default, and restart, of the outputs that are needed for a restart
         * @throws std::invalid_argument if a collection names an output or a collection that does not exist, or includes itself
         */
        void resolveCollections()
        {
            std::vector<std::string> defaultNames, restartNames;
            for (const outputInfo &output : outputs){
                if (output.includedByDefault) defaultNames.push_back(output.name);
                if (output.neededForRestart) restartNames.push_back(output.name);
            }
            addCollection("default", defaultNames);
            addCollection("restart", restartNames);
            std::vector<int> state(collections.size(), 0);
            for (size_t i = 0; i < collections.size(); ++i){
                resolveCollection(i, state);
            }
        }

        /**
         * Whether an output is registered with a name
         */
        bool hasOutput(const std::string &name) const
        {
            for (const outputInfo &output : outputs){
                if (output.name == name) return true;
            }
            return false;
        }

        /**
         * The number of registered outputs
         */
        size_t outputCount() const { return outputs.size(); }

        /**
         * The number of collections
         */
        size_t collectionCount() const { return collections.size(); }

        /**
         * Get the index of a collection by name
         * @throws std::invalid_argument if no collection has that name
         */
        size_t collectionIndex(const std::string &name) const { return findCollection(name); }

        /**
         * The name of a collection
         */
        const std::string &collectionName(size_t i) const { return collections[i].name; }

        /**
         * Whether each registered output is in a collection. Only valid once the collections are resolved
         */
        const std::vector<bool> &collectionMembers(size_t i) const { return collections[i].members; }
    };

    /**
     * The outputs that a dump writes, which are the members of the collections that are active for it
     */
    class outputLevel
    {
        std::vector<bool> members;
        std::vector<std::string> activeCollections;
    public:
        /**
         * @param members Whether each registered output is written
         * @param activeCollections The names of the collections that are active for the dump
         */
        outputLevel(const std::vector<bool> &members, const std::vector<std::string> &activeCollections) : members(members), activeCollections(activeCollections) {}

        /**
         * Whether the dump writes an output
         * @param output The handle of the output
         */
        bool includes(const outputHandle &output) const
        {
            return output.index < members.size() && members[output.index];
        }

        /**
         * Whether a collection is active for the dump
         * @param collection The name of the collection
         */
        bool isActive(const std::string &collection) const
        {
            return std::find(activeCollections.begin(), activeCollections.end(), collection) != activeCollections.end();
        }
    };

    /**
     * Whether packages request the end of the run, and the reasons that they give
     */
    class terminationRequest
    {
        bool requested = false;
        std::vector<std::string> reasons;
    public:
        /**
         * Request the end of the run
         * @param reason Why the run ends
         */
        void terminate(const std::string &reason)
        {
            requested = true;
            reasons.push_back(reason);
        }

        /**
         * Whether the end of the run has been requested
         */
        bool isRequested() const { return requested; }

        /**
         * The reasons that packages gave for ending the run
         */
        const std::vector<std::string> &getReasons() const { return reasons; }

        /**
         * The flag itself, for packages that take a bool and give no reason
         */
        operator bool&() { return requested; }
    };

    /**
     * The collections of outputs that packages activate on a step
     */
    class outputRequest
    {
    public:
        /**
         * What is written to one named dump
         */
        struct dumpProperties
        {
            //The indices of the collections in the dump
            std::vector<size_t> collections;
            //The indices of the single outputs in the dump
            std::vector<size_t> outputs;
            //Whether time coalescence is enabled for the dump
            bool timeCoalescence = true;
            //The writerName of the writer of the dump. Empty for the default writer
            std::string writerName;
        };

    private:
        const outputRegistry &registry;
        //The named dumps by dump name. Same on every rank
        std::map<std::string, dumpProperties> dumps;
        //Whether this is the request after the last step
        bool final;
    public:
        /**
         * @param registry The registry of outputs, which holds the collections
         * @param final Whether this is the request after the last step
         */
        outputRequest(const outputRegistry &registry, bool final = false) : registry(registry), final(final) {}

        /**
         * Whether this is the request after the last step
         */
        bool isFinal() const { return final; }

        /**
         * Disable time coalescence for a named dump
         */
        void disableTimeCoalescence(const std::string &dumpName) { dumps[dumpName].timeCoalescence = false; }

        /**
         * Set the writer of a named dump. The first writer that is set is kept, and a writer that is not available leaves the default writer
         * @param dumpName Relative to the output directory, without extension
         * @param writerName The writerName of the writer
         */
        void setWriter(const std::string &dumpName, const std::string &writerName)
        {
            bool available = false;
#define WRITER_DEF(param, T_writer) available |= (writerName == T_writer::writerName);
            WRITER_TYPES(0)
#undef WRITER_DEF
            if (!available){
                SAMS::cerr << "Warning: there is no writer " << writerName << ", so " << dumpName << " uses the default writer" << std::endl;
                return;
            }
            std::string &current = dumps[dumpName].writerName;
            if (current.empty()) current = writerName;
            else if (current != writerName) SAMS::cerr << "Warning: the writer of " << dumpName << " is already " << current << ", so " << writerName << " is not used" << std::endl;
        }

        /**
         * Activate a single output or a collection in a named dump. An output takes precedence over a collection with the same name
         * @param dumpName Relative to the output directory, without extension
         */
        void activate(const std::string &name, const std::string &dumpName)
        {
            if (registry.hasOutput(name)) dumps[dumpName].outputs.push_back(registry.getHandle(name).index);
            else dumps[dumpName].collections.push_back(registry.collectionIndex(name));
        }

        /**
         * Activate a collection in a named dump
         * @param dumpName Relative to the output directory, without extension
         */
        void activateCollection(const std::string &collection, const std::string &dumpName)
        {
            dumps[dumpName].collections.push_back(registry.collectionIndex(collection));
        }

        /**
         * Activate a single output in a named dump
         * @param dumpName Relative to the output directory, without extension
         */
        void activateVariables(const std::string &output, const std::string &dumpName)
        {
            dumps[dumpName].outputs.push_back(registry.getHandle(output).index);
        }

        /**
         * Activate a single output in a named dump
         * @param dumpName Relative to the output directory, without extension
         */
        void activate(const outputHandle &output, const std::string &dumpName)
        {
            dumps[dumpName].outputs.push_back(output.index);
        }

        const std::map<std::string, dumpProperties> &namedDumps() const { return dumps; }
    };
    
    /**
     * Functor that gives the value of an axis read from a restart dump at a core index
     */
    template<typename T>
    struct restartAxisValues
    {
        //The values of the axis, from its first element
        const T* values;
        //The core index of the first element
        SIGNED_INDEX_TYPE start;

        FUNCTORMETHODPREFIX INLINE T operator()(SIGNED_INDEX_TYPE i) const
        {
            return values[i - start];
        }
    };

    /**
     * Runner class that runs a set of simulations
     */
    template<typename... Packages>
    class runner
    {
        HAS_X_PARAM_WITH_TYPE(runnerOrder, uint8_t); //Order that the hooks of a package are called in, lower first

        /** The order of a package, which is 128 if it does not give one */
        template<typename T>
        static constexpr uint8_t packageOrder = std::is_void_v<typename hasParamType_runnerOrder<T>::type> ? uint8_t(128) : uint8_t(hasParamType_runnerOrder<T>::value);

        public:
        /** Types of requested simulations, in the order of their runnerOrder. Must come first */
        using T_packages = reorderTuple_t<std::tuple<Packages...>, sortedIndices<sizeof...(Packages)>({packageOrder<Packages>...})>;
        harness harnessData;
        timeState timeStateData;
#ifdef USE_DECK_PARSER
        /** The parser that evaluates expressions in the deck */
        eis::parser<T_dataType> parserData;
#endif
        /** Internal runner data that might be requested by simulations
         * Add new types here if they are internal state shared by multiple simulations
        */
#ifdef USE_DECK_PARSER
        using T_runnerData = std::tuple<harness&, timeState&, controlFunctions, SAMS::KVS, outputRegistry, fileInformation, eis::parser<T_dataType>&>;
#else
        using T_runnerData = std::tuple<harness&, timeState&, controlFunctions, SAMS::KVS, outputRegistry, fileInformation>;
#endif
        /**Data packs requested by simulations */
        using T_dataPacks = dataPackTuple_t<Packages...>;
        /** Packages and data packs are held by pointer, and only allocated if they are used */
        using T_combined = tupleUnion_t<toPointerTuple_t<T_packages>, T_runnerData, toPointerTuple_t<T_dataPacks>>;
#ifdef USE_DECK_PARSER
        T_combined runnerData = initializeTuple(harnessData, timeStateData, parserData);
#else
        T_combined runnerData = initializeTuple(harnessData, timeStateData);
#endif

        struct internalPackageInfo{
            int level = -1;
            bool isCoreSimulation = false;
            bool timeSimulation = false;
            std::string name = "";
        };

        std::map<std::string, internalPackageInfo> simulationInfoMap;
        std::array<bool, sizeof...(Packages)> simulationActiveFlags; //Active isn't in the map for performance reasons
        std::array<activationSource, sizeof...(Packages)> activationSources; //How each package was activated
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
        FULL_CALL_X(defaultDeck); //Set the deck keys that the deck does not set

        /**
         * 
         */
        FULL_CALL_X(registerAxes);//Register axes
        FULL_CALL_X(registerVariables); //Register variables
        FULL_CALL_X(registerOutputVariables); //Register the outputs that packages can write, which the deck selects from
        FULL_CALL_X(defaultValues); //Set default parameter values
        FULL_CALL_X(controlVariables); //Setup control variables
        FULL_CALL_X(setDomain); //Set the domain for simulations
        FULL_CALL_X(registerSecondaryAxes); //Register axes that depend on the decomposition of the main axes
        FULL_CALL_X(registerSecondaryVariables); //Register variables that use the secondary axes

        FULL_CALL_X(getVariables); //Get variable objects from the variable registry


        FULL_CALL_X(initialConditions); //Set initial conditions
        FULL_CALL_X(setBoundaryConditions); //Set boundary conditions
        FULL_CALL_X(defaultVariables); //Set default variable values
		FULL_CALL_X(initializeBoundaryConditions);
        FULL_CALL_X(repopulateDeck); //Put the parameters back into the deck

        FULL_CALL_X(beforeStartOfTimestep); //Actions just before the step starts
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
            timeState& tData = std::get<timeState&>(runnerData);
            //Packages should compare their suggested timesteps and set the smallest one
            tData.dt = std::numeric_limits<T_dataType>::max();
            callCore_calculateTimestep();
            #ifdef USE_MPI
            SAMS::harness& h = std::get<harness&>(runnerData);
            MPI_Allreduce(MPI_IN_PLACE, &tData.dt, 1, MPI_DOUBLE, MPI_MIN, h.MPIManager.getComm());
            #endif
            //Allow the simulations to gather the updated timestep
            callCore_getTimestep();
        }

        /** Whether packages have requested the end of the run, and why */
        terminationRequest termination;

        CALL_X(queryTerminate); //Query whether to terminate the simulation
        bool queryTerminate(){
            callCore_queryTerminate(termination);
#if defined(USE_MPI) && defined(ALLOW_RANK_INCONSISTENCY)
            SAMS::harness& h = std::get<harness&>(runnerData);
            int localTerm = termination.isRequested() ? 1 : 0;
            MPI_Allreduce(MPI_IN_PLACE, &localTerm, 1, MPI_INT, MPI_LOR, h.MPIManager.getComm());
            static_cast<bool&>(termination) = (localTerm == 1);
#endif
            return termination.isRequested();
        }

        /**
         * The number of active packages that can end the run
         */
        template<int level=0>
        int countTerminators(){
            int count = 0;
            if constexpr (hasMethod_queryTerminate<level>::value) count = simulationActiveFlags[level] ? 1 : 0;
            if constexpr (level < sizeof...(Packages)-1) return count + countTerminators<level+1>();
            else return count;
        }

        CALL_X(queryOutput); //Query which collections of outputs are needed this timestep
        outputRequest queryOutput(bool final = false){
            outputRequest request(std::get<outputRegistry>(runnerData), final);
            callCore_queryOutput(request);
//#if defined(USE_MPI) && defined(ALLOW_RANK_INCONSISTENCY)
            //A collection is written if any package on any rank activates it
//            SAMS::harness& h = std::get<harness&>(runnerData);
//            MPI_Allreduce(MPI_IN_PLACE, request.flags().data(), static_cast<int>(request.flags().size()), MPI_UNSIGNED_CHAR, MPI_MAX, h.MPIManager.getComm());
//#endif
            return request;
        }

        CALL_X(registerDumpMeshes); //Register meshes
        template<typename T>
        void registerDumpMeshes(SAMS::IO::writer<T>& w, SAMS::IO::registryWriter<T>& rw, const outputLevel &level){
            callCore_registerDumpMeshes<false, true,true,0,T>(w, rw, level);
        }

        CALL_X(registerDumpVariables); //Register variables
        template<typename T>
        void registerDumpVariables(SAMS::IO::writer<T>& w, SAMS::IO::registryWriter<T>& rw, const outputLevel &level){
            callCore_registerDumpVariables<false, true,true,0,T>(w, rw, level);
        }

        CALL_X(writeDumpMeshes); //Write output
        template<typename T>
        void writeDumpMeshes(SAMS::IO::writer<T>& w, SAMS::IO::registryWriter<T>& rw, const outputLevel &level){
            callCore_writeDumpMeshes<false,true,true,0,T>(w, rw, level);
        }

        CALL_X(writeDumpVariables); //Write variables
        template<typename T>
        void writeDumpVariables(SAMS::IO::writer<T>& w, SAMS::IO::registryWriter<T>& rw, const outputLevel &level){
            callCore_writeDumpVariables<false,true,true,0,T>(w, rw, level);
        }

        CALL_X(restart); //Read the state of the simulation from a restart dump in place of the initial conditions
        template<typename T>
        void restart(SAMS::IO::reader<T>& r, SAMS::IO::registryReader<T>& rr){
            callCore_restart<false,true,true,0,T>(r, rr);
        }

        FULL_CALL_X(steer); //Computational steering

        CALL_X(finalize); //Finalize simulations

        HAS_X_PARAM_WITH_TYPE(name, std::string); //Package name parameter
        HAS_X_PARAM_WITH_TYPE(coreSimulation, bool); //Is this a core simulation
        HAS_X_PARAM_WITH_TYPE(timeSimulation, bool); //Should this simulation be timed
        HAS_X_PARAM_WITH_TYPE(defaultActivate, bool); //Is this package active unless the deck deactivates it
        HAS_X_PARAM_WITH_TYPE(description, std::string_view); //One sentence description of the package

        public:

        /**
         * Populate a single simulation info struct
         */
        template<int level=0>
        internalPackageInfo buildPackageInfo(){
            internalPackageInfo info;
            info.name = hasParamType_name<std::tuple_element_t<level, T_packages>>::value;
            info.level = level;
            if constexpr (std::is_void_v<typename hasParamType_coreSimulation<std::tuple_element_t<level, T_packages>>::type> == false){
                info.isCoreSimulation = hasParamType_coreSimulation<std::tuple_element_t<level, T_packages>>::value;
            } else {
                info.isCoreSimulation = false;
            }
            if constexpr (std::is_void_v<typename hasParamType_timeSimulation<std::tuple_element_t<level, T_packages>>::type> == false){
                info.timeSimulation = hasParamType_timeSimulation<std::tuple_element_t<level, T_packages>>::value;
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
            if constexpr (std::is_void_v<typename hasParamType_name<std::tuple_element_t<level, T_packages>>::type> == false){
                std::string name = static_cast<std::string>(hasParamType_name<std::tuple_element_t<level, T_packages>>::value);
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
         * Print the active packages, in the order that their hooks are called
         */
        template<int level=0>
        void printActive(){
            if (simulationActiveFlags[level]){
                internalPackageInfo info = buildPackageInfo<level>();
                const char *color = VT100Green;
                if (activationSources[level] == activationSource::defaultActivated) color = VT100Yellow;
                else if (activationSources[level] == activationSource::package) color = VT100Blue;
                SAMS::cout << " - " << colorSet(color) << info.name ;
                if (info.isCoreSimulation){
                    SAMS::cout << " (Core Simulation)";
                }
                SAMS::cout << colorSet(VT100Reset) << std::endl;
            }
            if constexpr (level < sizeof...(Packages)-1){
                printActive<level+1>();
            }
        }

        /**
         * Print every package with its description. Core packages are green and default packages yellow
         */
        template<int level=0>
        void printPackages(){
            using T_package = std::tuple_element_t<level, T_packages>;
            internalPackageInfo info = buildPackageInfo<level>();
            bool isDefault = false;
            if constexpr (!std::is_void_v<typename hasParamType_defaultActivate<T_package>::type>){
                isDefault = hasParamType_defaultActivate<T_package>::value;
            }
            SAMS::cout << " - ";
            if (info.isCoreSimulation) SAMS::cout << colorSet(VT100Green);
            else if (isDefault) SAMS::cout << colorSet(VT100Yellow);
            SAMS::cout << info.name << colorSet(VT100Reset);
            if constexpr (!std::is_void_v<typename hasParamType_description<T_package>::type>){
                SAMS::cout << " : " << hasParamType_description<T_package>::value;
            }
            SAMS::cout << std::endl;
            if constexpr (level < sizeof...(Packages)-1){
                printPackages<level+1>();
            }
        }

        /**
         * Function to initialize the combined tuple of simulations, runner data and data packs
         */
        template<int level=0>
#ifdef USE_DECK_PARSER
        static decltype(auto) inline __attribute__((always_inline)) initializeTupleElement(harness& h, timeState& t, eis::parser<T_dataType>& p){
#else
        static decltype(auto) inline __attribute__((always_inline)) initializeTupleElement(harness& h, timeState& t){
#endif
            using T_current = std::tuple_element_t<level, T_combined>;

            //Case where the tuple element is just default constructible
            if constexpr (std::is_default_constructible_v<T_current>) {
                return T_current{};
            //Case where the tuple element is a reference to the harness
            } else if constexpr (std::is_reference_v<T_current> && std::is_same_v<std::remove_reference_t<T_current>,harness>) {
                return h;
            //Case where the tuple element is a reference to the timestate
            } else if constexpr (std::is_reference_v<T_current> && std::is_same_v<std::remove_reference_t<T_current>,timeState>) {
                return t;
#ifdef USE_DECK_PARSER
            //Case where the tuple element is a reference to the parser
            } else if constexpr (std::is_reference_v<T_current> && std::is_same_v<std::remove_reference_t<T_current>,eis::parser<T_dataType>>) {
                return p;
#endif
            //Case where the tuple element is constructible from a harness reference
            } else if constexpr (std::is_constructible_v<T_current, harness&>) {
                return T_current{h};
            //Case where the tuple element is constructible from a timestate reference
            } else if constexpr (std::is_constructible_v<T_current, timeState&>) {
                return T_current{t};
            } else {
                static_assert(portableWrapper::alwaysFalse<T_current>::value,
                              "Error: Package or data pack type is not default constructible or constructible from a harness reference.");
            }
        }

#ifdef USE_DECK_PARSER
        template<std::size_t... Is>
        static auto inline __attribute__((always_inline)) initializeTupleImpl(harness& h, timeState& t, eis::parser<T_dataType>& p, std::index_sequence<Is...>) {
            return T_combined{initializeTupleElement<Is>(h, t, p)...};
        }

        static auto inline __attribute__((always_inline)) initializeTuple(harness& h, timeState& t, eis::parser<T_dataType>& p) {
            constexpr std::size_t N = std::tuple_size_v<T_combined>;
            return initializeTupleImpl(h, t, p, std::make_index_sequence<N>{});
        }
#else
        template<std::size_t... Is>
        static auto inline __attribute__((always_inline)) initializeTupleImpl(harness& h, timeState& t, std::index_sequence<Is...>) {
            return T_combined{initializeTupleElement<Is>(h, t)...};
        }

        static auto inline __attribute__((always_inline)) initializeTuple(harness& h, timeState& t) {
            constexpr std::size_t N = std::tuple_size_v<T_combined>;
            return initializeTupleImpl(h, t, std::make_index_sequence<N>{});
        }
#endif

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

        /**
         * Static function to call deactivatePackage on a runner instance
         * Used in type-erased control function binding
         * @param obj Pointer to the runner instance
         * @param name Name of the package to deactivate
         */
        static void callDeactivatePackage(void* obj, const std::string& name){
            runner* r = static_cast<runner*>(obj);
            r->deactivatePackage(name);
        }

        /**
         * Static function to call getActivationSource on a runner instance
         * Used in type-erased control function binding
         * @param obj Pointer to the runner instance
         * @param name Name of the package to query
         * @return How the package was activated
         */
        static activationSource callGetActivationSource(void* obj, const std::string& name){
            runner* r = static_cast<runner*>(obj);
            return r->getActivationSource(name);
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
            return tupleTypeIndex_v<T, T_packages> >= 0;
        }

        /**
         * Is T a runner data type managed by this runner
         * @tparam T The type to check
         */
        template<typename T>
        constexpr bool isRunnerDataType(){
            return tupleTypeIndex_v<T, T_runnerData> >= 0;
        }

        /**
         * is T a data pack type managed by this runner
         * @tparam T The type to check
         */
        template<typename T>
        constexpr bool isDataPackType(){
            return tupleTypeIndex_v<T, T_dataPacks> >= 0;
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
            return *std::get<T_sim*>(runnerData);
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
         * Get the key value store from the runner
         * @return A reference to the key value store
         */
        KVS& getKVS()
        {
            return std::get<SAMS::KVS>(runnerData);
        }

        /** Whether the input deck is in EPOCH format. Otherwise it is JSON */
        bool inputIsEPOCH = true;

        /** The output format, which is the writerName of the writer that writes the output */
        std::string outputFormat;

        /** Percentage of a package's total time below which its phases are collapsed into Other in the timing output */
        T_dataType timerThreshold = 10.0;

        /** The number of steps between the progress messages. 0 for none */
        size_t progressSteps = 100;

        /**
         * Set the output directory for the run
         * @param dir The output directory, with or without trailing slash
         */
        void setOutputDir(const std::string& dir){
            std::get<fileInformation>(runnerData).outputDirectory = dir;
        }

        /**
         * Set the dump that the run restarts from
         * @param dump The dump, relative to the output directory and with its extension
         */
        void setRestartDump(const std::string& dump){
            getKVS().setContentPath(deckKeys::restart, dump);
        }

        /**
         * Read the input deck from a directory, which is also the output directory. The deck is input.deck in EPOCH format,
         * or input.json if there is no input.deck
         * @param dir The directory
         * @param deckName The name of the deck in place of input.deck, or empty. JSON if it ends in .json, otherwise EPOCH
         */
        void readInputDeck(const std::string& dir, const std::string& deckName){
            setOutputDir(dir);
            KVS& kvs = getKVS();
            //Rank 0 reads the deck and broadcasts it
#ifdef USE_MPI
            SAMSMPI::comm comm = getHarness().MPIManager.getComm();
            auto exists = [comm](const std::string &name){ return IO::existsOnRoot(comm, name); };
            auto read = [comm](const std::string &name){ return IO::readOnRoot(comm, name); };
#else
            auto exists = [](const std::string &name){ return IO::existsOnRoot(name); };
            auto read = [](const std::string &name){ return IO::readOnRoot(name); };
#endif
            try {
                std::string fileName = deckName;
                if (fileName.empty()) {
                    if (exists(dir + "/input.deck")) fileName = "input.deck";
                    else if (exists(dir + "/input.json")) fileName = "input.json";
                    else abort("No input file selected! Place a suitable input.deck or input.json file in the output directory", false);
                }
                inputIsEPOCH = std::filesystem::path(fileName).extension() != ".json";
                std::get<fileInformation>(runnerData).deckDirectory = (std::filesystem::path(dir) / fileName).parent_path().string();
                const std::string text = read(dir + "/" + fileName);
                if (inputIsEPOCH) kvs = KVSConversion::fromEPOCHString(text);
                else kvs = KVSConversion::fromJSONString(text);
            } catch (const std::exception& e) {
                abort(std::string("Error reading input deck: ") + e.what(), false);
            }
        }

        /**
         * Use a deck that only holds /SAMS/packages in place of an input deck
         * @param dir The output directory
         * @param packages The value of /SAMS/packages in deck format. Brackets are added if they are not there
         */
        void readPackageList(const std::string& dir, const std::string& packages){
            setOutputDir(dir);
            std::get<fileInformation>(runnerData).deckDirectory = dir;
            const size_t start = packages.find_first_not_of(" \t");
            const size_t end = packages.find_last_not_of(" \t");
            std::string value = start == std::string::npos ? std::string() : packages.substr(start, end - start + 1);
            if (value.empty() || value[0] != '(') value = "(" + value + ")";
            try {
                getKVS().merge(KVSConversion::fromEPOCHString("begin:SAMS\n  packages = " + value + "\nend:SAMS\n"));
            } catch (const std::exception& e) {
                abort(std::string("Error reading --packages: ") + e.what(), false);
            }
            inputIsEPOCH = true;
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
            return *std::get<T_dataPack*>(runnerData);
        }


        // Activate a simulation by name
        void activatePackage(const std::string& simName, activationSource source = activationSource::package)
        {
            auto it = simulationInfoMap.find(simName);
            if (it != simulationInfoMap.end()){
                simulationActiveFlags[it->second.level] = true;
                activationSources[it->second.level] = source;
            } else {
                std::stringstream ss;
                ss << "ERROR: Package with name " << simName << " not found in runner.\n";
                abort(ss.str(), false);
            }
        }

        // Deactivate a simulation by name
        void deactivatePackage(const std::string& simName)
        {
            auto it = simulationInfoMap.find(simName);
            if (it != simulationInfoMap.end()){
                simulationActiveFlags[it->second.level] = false;
                activationSources[it->second.level] = activationSource::none;
                //A package deactivated after it was initialized is finalized and freed
                if (packagesInitialized) finalizeAndRelease(it->second.level);
            } else {
                std::stringstream ss;
                ss << "ERROR: Package with name " << simName << " not found in runner.\n";
                abort(ss.str(), false);
            }
        }

        // Deactivate every package that is active by default, as + does in /SAMS/packages
        void deactivatePackage()
        {
            for (const auto &[name, info] : simulationInfoMap){
                if (activationSources[info.level] == activationSource::defaultActivated) deactivatePackage(name);
            }
        }

        /** Whether the packages have been initialized */
        bool packagesInitialized = false;

        /**
         * Finalize and free the package at a level
         */
        template<int level=0>
        void finalizeAndRelease(int target){
            if (level == target){
                callCore_finalize<false, true, false, level>();
                using T_package = std::tuple_element_t<level, T_packages>;
                T_package *&slot = std::get<T_package*>(runnerData);
                delete slot;
                slot = nullptr;
            }
            if constexpr (level < sizeof...(Packages)-1){
                finalizeAndRelease<level+1>(target);
            }
        }

        // How a simulation was activated, by name
        activationSource getActivationSource(const std::string& simName)
        {
            return activationSources[simulationInfoMap.at(simName).level];
        }

        /**
         * Activate the packages that are active unless the deck deactivates them
         */
        template<int level=0>
        void activateDefaults(){
            if constexpr (!std::is_void_v<typename hasParamType_defaultActivate<std::tuple_element_t<level, T_packages>>::type>){
                if (hasParamType_defaultActivate<std::tuple_element_t<level, T_packages>>::value){
                    simulationActiveFlags[level] = true;
                    activationSources[level] = activationSource::defaultActivated;
                }
            }
            if constexpr (level < sizeof...(Packages)-1){
                activateDefaults<level+1>();
            }
        }

        /**
         * The names of the active packages, in the order that their hooks are called. Default packages have a + prefix
         */
        template<int level=0>
        void activeNames(KVS::Array &names){
            if (simulationActiveFlags[level]) names.push_back((activationSources[level] == activationSource::defaultActivated ? "+" : "") + std::string(hasParamType_name<std::tuple_element_t<level, T_packages>>::value));
            if constexpr (level < sizeof...(Packages)-1){
                activeNames<level+1>(names);
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
            auto& varReg = h.variableRegistry;
            h.axisRegistry.decomposeAllAxes();
            //Now that the main axes are decomposed, packages can register axes that depend on that, and decompose them
            registerSecondaryAxes();
            h.axisRegistry.decomposeAllAxes();
            registerSecondaryVariables();
            //Allocate variables
            varReg.allocateAll();
        }

        /**
         * Allocate a package or data pack. A package that has constructorParameters is constructed from those, taken from the runner
         * data by type. Otherwise it is default constructed if possible, or constructed from the harness or the time state
         */
        template<typename T>
        T* allocateRunnerElement()
        {
            if constexpr (tupleTypeIndex_v<T, T_packages> >= 0 && !std::is_void_v<getConstructorParametersType_t<T>>) {
                auto parameters = getCallTupleElements<getConstructorParametersType_t<T>>(runnerData);
                return std::apply([](auto&&... arguments){ return new T(std::forward<decltype(arguments)>(arguments)...); }, parameters);
            } else if constexpr (std::is_default_constructible_v<T>) {
                return new T{};
            } else if constexpr (std::is_constructible_v<T, harness&>) {
                return new T{getHarness()};
            } else if constexpr (std::is_constructible_v<T, timeState&>) {
                return new T{std::get<timeState&>(runnerData)};
            } else {
                static_assert(portableWrapper::alwaysFalse<T>::value,
                              "Error: Package or data pack type is not default constructible or constructible from a harness reference.");
            }
        }

        /**
         * Allocate the data packs that the package at a level uses. A data pack that is also runner data already exists,
         * and anything that is already allocated is left as it is
         */
        template<int level, int pack=0>
        void inline __attribute__((always_inline)) allocateDataPacks()
        {
            using T_packs = uniqueTuple_t<getDataPackType_t<std::tuple_element_t<level, T_packages>>>;
            if constexpr (pack < std::tuple_size_v<T_packs>){
                using T_pack = std::tuple_element_t<pack, T_packs>;
                if constexpr (tupleTypeIndex_v<T_pack, T_runnerData> < 0){
                    T_pack *&slot = std::get<T_pack*>(runnerData);
                    if (!slot) slot = allocateRunnerElement<T_pack>();
                }
                allocateDataPacks<level, pack+1>();
            }
        }

        /**
         * Allocate the active packages and the data packs that they use. The data packs are allocated first, and anything that is already allocated is left as it is.
         * Packages that are allocated but not active are freed. Their data packs are not
         */
        template<bool packages=false, int level=0>
        void inline __attribute__((always_inline)) allocateActive()
        {
            if (simulationActiveFlags[level]){
                if constexpr (packages){
                    using T_package = std::tuple_element_t<level, T_packages>;
                    T_package *&slot = std::get<T_package*>(runnerData);
                    if (!slot) slot = allocateRunnerElement<T_package>();
                } else {
                    allocateDataPacks<level>();
                }
            } else if constexpr (packages){
                using T_package = std::tuple_element_t<level, T_packages>;
                T_package *&slot = std::get<T_package*>(runnerData);
                delete slot;
                slot = nullptr;
            }
            if constexpr (level < sizeof...(Packages)-1){
                allocateActive<packages, level+1>();
            } else if constexpr (!packages){
                //All of the data packs are allocated, so allocate the packages
                allocateActive<true>();
            }
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
            allocateActive();
            callCore_initialize();
            packagesInitialized = true;
        }

        /** The number of dumps that have been written with each suffix */
        //std::map<std::string, SAMS::T_sizeType> outputCounts;

        /** The version of the part of restart dumps that the runner writes */
        static constexpr int restartVersion = 1;

#ifdef USE_DECK_PARSER
        /**
         * Evaluate the text of a deck value as an expression with the parser of the runner. Used by the KVS when a number is asked for and the value is a string
         * @param parserPtr The parser
         * @param text The text of the value
         * @return The value of the expression
         * @throws std::runtime_error if the expression cannot be evaluated
         */
        static double evaluateDeckNumber(void* parserPtr, const std::string &text){
            eis::parser<T_dataType> &parser = *static_cast<eis::parser<T_dataType>*>(parserPtr);
            std::vector<T_dataType> result;
            eis::error errcode = eis::err_none;
            parser.evaluate(text, result, errcode);
            if (errcode != eis::err_none) {
                throw std::runtime_error("Could not evaluate " + text + "\n" + parserErrors(parser));
            }
            return static_cast<double>(result[0]);
        }
#endif

        /** The dump that the run restarts from, without the extension that its format adds. Empty if the run does not restart */
        std::string restartName;
        /** The format of the dump that the run restarts from, which is the readerName of the reader that reads it */
        std::string restartFormat;
        /** The reader of the dump that the run restarts from, which is a reader of its format. It is open from when the axes are read until the restart phase is done */
        void* restartReader = nullptr;
        /** One writer of each output format that has been used, by writerName. They persist for the run */
        std::map<std::string_view, std::shared_ptr<void>> writers;

        /**
         * If the deck gives a dump to restart from, merge the deck of the dump into the deck, with the keys of the deck replacing those of the dump.
         * The dump is /SAMS/restart in the output directory. Its format is hdf5 if it ends in .h5, adios2 if it ends in .bp, and simple if it is a directory
         */
        void readRestartDeck(){
            const std::string restartKey = getKVS().getString(deckKeys::restart, "");
            if (restartKey.empty()) return;
            fileInformation &files = std::get<fileInformation>(runnerData);
            const std::filesystem::path dump = std::filesystem::path(files.outputDirectory) / restartKey;
            files.restartPath = dump.string();
            if (dump.extension() == ".h5") {
                restartFormat = "hdf5";
                restartName = (dump.parent_path() / dump.stem()).string();
            } else if (dump.extension() == ".bp") {
                restartFormat = "adios2";
                restartName = (dump.parent_path() / dump.stem()).string();
            } else if (std::filesystem::is_directory(dump)) {
                restartFormat = "simple";
                restartName = dump.string();
            } else {
                abort("Invalid /SAMS/restart: " + restartKey + " is not a dump", false);
            }
#define READER_DEF(param, T_reader) \
            if (restartFormat == T_reader::readerName) readRestartDeckWith<T_reader>();
            READER_TYPES(0)
#undef READER_DEF
        }

        /**
         * Merge the deck of the dump that the run restarts from into the deck
         * @tparam T_reader The type of the reader of the dump
         */
        template<typename T_reader>
        void readRestartDeckWith(){
            T_reader reader(getHarness());
            reader.openFile(restartName.c_str());
            getKVS().merge(KVSConversion::fromEPOCHString(reader.readDeck()));
            reader.closeFile();
        }

        /**
         * Open the dump that the run restarts from and set each axis in the axis registry from it, in place of setting the domain.
         * The dump stays open until the restart phase is done
         */
        void readRestartAxes(){
#define READER_DEF(param, T_reader) \
            if (restartFormat == T_reader::readerName) readRestartAxesWith<T_reader>();
            READER_TYPES(0)
#undef READER_DEF
        }

        /**
         * Open the dump that the run restarts from and set each axis in the axis registry from it
         * @tparam T_reader The type of the reader of the dump
         */
        template<typename T_reader>
        void readRestartAxesWith(){
            T_reader *reader = new T_reader(getHarness());
            restartReader = reader;
            reader->openFile(restartName.c_str());
            auto &axisReg = getHarness().axisRegistry;
            for (const auto &entry : axisReg.getAxisMap()){
                //A logical axis only has cell centres, and every other axis is set from its edges
                const bool logical = entry.second.logicalAxis;
                const std::string itemName = std::string("Axes/") + (logical ? "c" : "b") + "/" + entry.first;
                const SAMS::IO::itemInfo info = reader->axisInfo(itemName.c_str());
                //The axis in the dump holds its ghost cells, so the rest are its cells, or its edges, which are one more
                const COUNT_TYPE cells = static_cast<COUNT_TYPE>(info.sizes[0] - info.lowerGhosts - info.upperGhosts) - (logical ? 0 : 1);
                const SAMS::staggerType stagger = logical ? SAMS::staggerType::CENTRED : SAMS::staggerType::HALF_CELL;
                //The axis is read with the type that it has in the dump
                switch (static_cast<SAMS::typeID>(info.type.ID)){
                    case SAMS::typeID::TYPE_DOUBLE: setRestartAxis<double>(*reader, entry.first, itemName, info, cells, stagger); break;
                    case SAMS::typeID::TYPE_FLOAT: setRestartAxis<float>(*reader, entry.first, itemName, info, cells, stagger); break;
                    case SAMS::typeID::TYPE_INT32: setRestartAxis<int32_t>(*reader, entry.first, itemName, info, cells, stagger); break;
                    case SAMS::typeID::TYPE_INT64: setRestartAxis<int64_t>(*reader, entry.first, itemName, info, cells, stagger); break;
                    case SAMS::typeID::TYPE_UINT32: setRestartAxis<uint32_t>(*reader, entry.first, itemName, info, cells, stagger); break;
                    case SAMS::typeID::TYPE_UINT64: setRestartAxis<uint64_t>(*reader, entry.first, itemName, info, cells, stagger); break;
                    case SAMS::typeID::TYPE_INT8: setRestartAxis<int8_t>(*reader, entry.first, itemName, info, cells, stagger); break;
                    case SAMS::typeID::TYPE_INT16: setRestartAxis<int16_t>(*reader, entry.first, itemName, info, cells, stagger); break;
                    case SAMS::typeID::TYPE_UINT8: setRestartAxis<uint8_t>(*reader, entry.first, itemName, info, cells, stagger); break;
                    case SAMS::typeID::TYPE_UINT16: setRestartAxis<uint16_t>(*reader, entry.first, itemName, info, cells, stagger); break;
                    default: throw std::invalid_argument("Axis " + itemName + " in the restart dump has a type that cannot be read");
                }
            }
        }

        /**
         * Set an axis in the axis registry from an axis in the dump that the run restarts from
         * @tparam T_axis The type of the axis
         * @tparam T_reader The type of the reader of the dump
         * @param reader The reader of the dump
         * @param name The name of the axis in the axis registry
         * @param itemName The name of the axis in the dump
         * @param info The description of the axis in the dump
         * @param cells The number of cells of the axis
         * @param stagger The staggering of the axis in the dump
         */
        template<typename T_axis, typename T_reader>
        void setRestartAxis(T_reader &reader, const std::string &name, const std::string &itemName, const SAMS::IO::itemInfo &info, COUNT_TYPE cells, SAMS::staggerType stagger){
            std::vector<T_axis> values(info.sizes[0]);
            reader.readAxis(itemName.c_str(), values.data());
            getHarness().axisRegistry.setDomainValues(name, cells, restartAxisValues<T_axis>{values.data(), info.starts[0]}, stagger);
        }

        /**
         * Read the state of the runner from a restart dump
         * @tparam T_reader The type of the reader of the dump
         * @param reader The reader of the dump
         */
        template<typename T_reader>
        void readSimulationState(T_reader &reader){
            timeState &tData = std::get<timeState&>(runnerData);
            tData.time = reader.template readKeyValue<double>("simulation/time");
            tData.step = static_cast<T_indexType>(reader.template readKeyValue<IO::stepType>("simulation/step"));
            tData.dt = reader.template readKeyValue<double>("simulation/dt");
            //The restart dump is the last dump of its step, so the counts of the other suffixes already count their dumps of the step
            //const outputRegistry &outputs = std::get<outputRegistry>(runnerData);
            //for (size_t i = 0; i < outputs.collectionCount(); ++i){
            //    const std::string &suffix = outputs.collectionSuffix(i);
            //    const std::string key = "simulation/output_count_" + (suffix.empty() ? std::string("main") : suffix);
                //A suffix that had no dumps before the restart has no count
            //    if (!reader.hasKeyValue(key.c_str())) continue;
            //    outputCounts[suffix] = static_cast<SAMS::T_sizeType>(std::stoull(reader.readKeyValue(key.c_str()))) + (suffix == "restart" ? 1 : 0);
            //}
            //Each writer reads its own state
#define WRITER_DEF(param, T_writer) getWriter<T_writer>().restart(reader);
            WRITER_TYPES(0)
#undef WRITER_DEF
        }

        /**
         * Read the state of the simulation from the dump that the run restarts from, in place of setting the initial conditions, then close the dump
         */
        void readRestart(){
#define READER_DEF(param, T_reader) \
            if (restartFormat == T_reader::readerName) readRestartWith<T_reader>();
            READER_TYPES(0)
#undef READER_DEF
        }

        /**
         * Read the state of the simulation from the dump that the run restarts from, then close the dump
         * @tparam T_reader The type of the reader of the dump
         */
        template<typename T_reader>
        void readRestartWith(){
            T_reader *reader = static_cast<T_reader*>(restartReader);
            readSimulationState(*reader);
            SAMS::IO::registryReader regReader(getHarness(), *reader);
            restart(*reader, regReader);
            reader->closeFile();
            delete reader;
            restartReader = nullptr;
        }

        /**
         * Write the dumps of the active collections. Unnamed collections give one dump per suffix, named collections one dump per name
         */
        void writeOutput(const outputRequest &request){
            //const outputRegistry &outputs = std::get<outputRegistry>(runnerData);
            //std::vector<std::string> suffixes;
            //for (size_t i = 0; i < outputs.collectionCount(); ++i){
            //    if (request.isActive(i) && std::find(suffixes.begin(), suffixes.end(), outputs.collectionSuffix(i)) == suffixes.end()){
            //        suffixes.push_back(outputs.collectionSuffix(i));
            //    }
            //}
            //for (const std::string &suffix : suffixes){
                //The dump writes the members of every active collection with the suffix
            //    std::vector<size_t> collections;
            //    for (size_t i = 0; i < outputs.collectionCount(); ++i){
            //        if (request.isActive(i) && outputs.collectionSuffix(i) == suffix) collections.push_back(i);
            //    }
            //    SAMS::T_sizeType &outputCount = outputCounts[suffix];
            //    IO::dumpInfo info;
            //    info.suffix = suffix;
            //    info.number = outputCount;
            //    writeCollections(info, collections);
            //    outputCount++;
            //}
            //Named dumps. Restart dumps are written last, so that the dump numbers that they hold count every dump of the step
            const size_t restartIndex = std::get<outputRegistry>(runnerData).collectionIndex("restart");
            for (int pass = 0; pass < 2; ++pass){
                for (const auto &[name, dump] : request.namedDumps()){
                    const bool isRestart = std::find(dump.collections.begin(), dump.collections.end(), restartIndex) != dump.collections.end();
                    if (isRestart != (pass == 1)) continue;
                    IO::dumpInfo info;
                    info.name = name;
                    info.attemptTimeCoalescence = dump.timeCoalescence;
                    writeCollections(info, dump.collections, dump.outputs, dump.writerName);
                }
            }
        }

        /**
         * Write one dump of a set of collections
         * @param info The suffix and number, or the name, of the dump. The directory, whether it is a restart and the time are set here
         * @param singleOutputs The indices of single outputs written as well as the collections
         * @param writerName The writerName of the writer of the dump. Empty for the default writer
         */
        void writeCollections(IO::dumpInfo info, const std::vector<size_t> &collections, const std::vector<size_t> &singleOutputs, const std::string &writerName){
            const outputRegistry &outputs = std::get<outputRegistry>(runnerData);
            std::vector<bool> members(outputs.outputCount(), false);
            std::vector<std::string> activeCollections;
            for (size_t i : collections){
                activeCollections.push_back(outputs.collectionName(i));
                const std::vector<bool> &collectionMembers = outputs.collectionMembers(i);
                for (size_t k = 0; k < members.size(); ++k){
                    if (collectionMembers[k]) members[k] = true;
                }
            }
            for (size_t k : singleOutputs) members[k] = true;
            const outputLevel level(members, activeCollections);
            info.directory = std::get<fileInformation>(runnerData).outputDirectory;
            info.restart = level.isActive("restart");
            info.time = std::get<timeState&>(runnerData).time;
            info.step = std::get<timeState&>(runnerData).step;
            const std::string &format = writerName.empty() ? outputFormat : writerName;
#define WRITER_DEF(param, T_writer) \
            if (format == T_writer::writerName) writeOutputWith<T_writer>(info, level);
            WRITER_TYPES(0)
#undef WRITER_DEF
        }

        /**
         * Write the state of the simulation, and the version of the runner's part of the restart, to a restart dump
         * @tparam T_writer The type of the writer
         * @param writer The writer of the restart dump
         */
        template<typename T_writer>
        void writeSimulationState(T_writer &writer){
            const timeState &tData = std::get<timeState&>(runnerData);
            writer.template setKeyValue<int64_t>("simulation/restart_version", restartVersion);
            writer.template setKeyValue<double>("simulation/time", tData.time);
            writer.template setKeyValue<IO::stepType>("simulation/step", tData.step);
            writer.template setKeyValue<double>("simulation/dt", tData.dt);
            //The number of dumps that have been written with each suffix. The main diagnostics files have the empty suffix, which is written as main
            //for (const auto &count : outputCounts){
            //    writer.setKeyValue(("simulation/output_count_" + (count.first.empty() ? std::string("main") : count.first)).c_str(), std::to_string(count.second).c_str());
            //}
            //Every writer that has been used writes its own state
#define WRITER_DEF(param, T_used) \
            if (auto it = writers.find(T_used::writerName); it != writers.end()) static_cast<T_used*>(it->second.get())->writeRestart(writer);
            WRITER_TYPES(0)
#undef WRITER_DEF
        }

        /**
         * The writer of a format. It is created when it is first used and persists for the run
         * @tparam T_writer The type of the writer
         */
        template<typename T_writer>
        T_writer &getWriter(){
            std::shared_ptr<void> &slot = writers[T_writer::writerName];
            if (!slot) slot = std::make_shared<T_writer>(getHarness());
            return *static_cast<T_writer*>(slot.get());
        }

        /**
         * Write a dump with a given writer
         * @tparam T_writer The type of the writer
         * @param info The dump
         * @param level The outputs that the dump writes
         */
        template<typename T_writer>
        void writeOutputWith(const IO::dumpInfo &info, const outputLevel &level){
            T_writer &writer = getWriter<T_writer>();
            writer.openFile(info);
            if (level.isActive("restart")) writeSimulationState(writer);
            if (level.isActive("restart")) writer.writeDeck(getKVS());
            //Writes the meshes and variables that packages register from the registries
            SAMS::IO::registryWriter regWriter(getHarness(), writer);
            registerDumpMeshes(writer, regWriter, level);
            registerDumpVariables(writer, regWriter, level);
            //Restart dumps hold every axis, so that every mesh can be rebuilt
            if (level.isActive("restart")) regWriter.registerAllAxes();
            writeDumpMeshes(writer, regWriter, level);
            regWriter.writeMeshes();
            writeDumpVariables(writer, regWriter, level);
            regWriter.writeVariables();
            writer.closeFile();
        }

        //CODE HERE AND BELOW IS TO ACTUALLY RUN THE PACAKGES


        //Initialise the runner
        void initialize(int &argc, char** &argv)
        {
            //Initialize the harness
            getHarness().initialize(argc, argv);
            //Bind the control functions
            auto& ctrlFuncs = SAMS::getItemFromTuple<controlFunctions>(runnerData);
            ctrlFuncs.bind(this);
            ctrlFuncs.bindCalculateTimestep(&runner::callCalculateTimestep);
            ctrlFuncs.bindIsActive(&runner::callIsPackageActive);
            ctrlFuncs.bindActivate(&runner::callActivatePackage);
            ctrlFuncs.bindDeactivate(&runner::callDeactivatePackage);
            ctrlFuncs.bindSource(&runner::callGetActivationSource);
            //Errors in the key value store that cannot be recovered from abort the run
            getKVS().fatalErrorHandler = [this](const std::string &message){ abort(message, false); };
            //Turn off all simulations
            std::fill(simulationActiveFlags.begin(), simulationActiveFlags.end(), false);
            std::fill(activationSources.begin(), activationSources.end(), activationSource::none);
            buildInfo();
        }

        /**
         * Add the collections of outputs from the deck, which are collection blocks in the diagnostics block, and resolve every collection.
         * Aborts if a collection is not valid
         */
        void readOutputCollections(){
            KVS& kvs = getKVS();
            outputRegistry &outputs = std::get<outputRegistry>(runnerData);
            try {
                KVS::Element diagnostics = kvs.getPath("/diagnostics", KVS::Element());
                if (diagnostics.isType(KVS::Type::Ordered)){
                    for (KVS::Element &block : diagnostics.get<KVS::Ordered&>().getAllByKey("collection")){
                        const std::string name = block.getPath("name").asString();
                        std::vector<std::string> outputNames;
                        KVS::Element outputList = block.getPath("outputs", KVS::Array());
                        if (!outputList.isType(KVS::Type::Array)){
                            throw std::invalid_argument("outputs of collection " + name + " must be an array");
                        }
                        for (size_t i = 0; i < outputList.size(); i++){
                            outputNames.push_back(outputList[i].asString());
                        }
                        //Included collections are include keys, which in the EPOCH deck are directives
                        std::vector<std::string> includes;
                        for (KVS::Element &include : block.get<KVS::Ordered&>().getAllByKey("include")){
                            includes.push_back(include.asString());
                        }
                        outputs.addCollection(name, outputNames, includes);
                    }
                }
                outputs.resolveCollections();
            } catch (const std::exception &e) {
                abort(std::string("Invalid collection of outputs: ") + e.what(), false);
            }
        }

        /**
         * Activate the packages that the deck requests, and write the active packages back to the deck
         */
        void activateRequestedPackages(){
            //A restart uses the deck of the dump that it restarts from, with the keys of the input deck replacing those of the dump
            readRestartDeck();
            //Activate packages requested in the KVS
            auto& kvs = getKVS();
            KVS::Element packages = kvs.getPath("/SAMS/packages", KVS::Array());
            //Entries in /SAMS/packages
            // - Name : activate
            // - ~Name : deactivate
            // - +Name : activate as a default. Any + entry replaces the built in defaults
            // - + : no defaults
            std::vector<std::string> activate, deactivate, defaults;
            bool haveDefaults = false;
            if (packages.isType(KVS::Type::Array) && packages.size() > 0){
                for (size_t i = 0; i < packages.size(); i++){
                    std::string pkgName;
                    try {
                        pkgName = packages[i].get<std::string>();
                    } catch (const std::exception &e) {
                        abort(std::string("Invalid /SAMS/packages: ") + e.what(), false);
                    }
                    if (!pkgName.empty() && pkgName[0] == '~') deactivate.push_back(pkgName.substr(1));
                    else if (!pkgName.empty() && pkgName[0] == '+'){
                        haveDefaults = true;
                        if (pkgName.size() > 1) defaults.push_back(pkgName.substr(1));
                    }
                    else activate.push_back(pkgName);
                }
            } else {
                SAMS::cerr << "Warning: No packages specified in KVS. Only the packages that are active by default will be activated." << std::endl;
            }
            if (haveDefaults){
                for (const std::string &name : defaults) activatePackage(name, activationSource::defaultActivated);
            } else {
                activateDefaults();
            }
            for (const std::string &name : activate) activatePackage(name, activationSource::deck);
            for (const std::string &name : deactivate) deactivatePackage(name);
            //Written back in the same form
            KVS::Array activeList;
            activeNames(activeList);
            if (std::find(activationSources.begin(), activationSources.end(), activationSource::defaultActivated) == activationSources.end()) activeList.push_back("+");
            for (const std::string &name : deactivate) activeList.push_back("~" + name);
            kvs.setContentPath("/SAMS/packages", activeList);
        }

#ifdef USE_DECK_PARSER
        /**
         * Set up the parser of the runner, which evaluates numbers in the deck
         */
        void initializeDeckParser(){
            //Numbers in the deck can be expressions, which are evaluated by the parser of the runner. Nothing before this reads a number from the deck
            eis::parser<T_dataType> &parser = parserData;
            eis::error parserError = eis::err_none;
            //The physical constants are in SI
            eis::parser_init_options parserOptions;
            parserOptions.physics = eis::physics_si;
            parser.init(parserError, parserOptions);
            //The time of the simulation is read when an expression is evaluated, so it is always the current time
            eis::parser_variable_options timeOptions;
            timeOptions.cap_bits = capTime;
            if (parserError == eis::err_none) parser.add_pointer_variable("time", &timeStateData.time, parserError, timeOptions);
            //The domain is not set until later, so its values are deferred until then
            for (const char *axis : {"x", "y", "z"}){
                if (parserError == eis::err_none) parser.add_function(std::string("domain.n") + axis, parserError, {.expected_params = 0});
                if (parserError == eis::err_none) parser.add_function(std::string("domain.") + axis + "_min", parserError, {.expected_params = 0});
                if (parserError == eis::err_none) parser.add_function(std::string("domain.") + axis + "_max", parserError, {.expected_params = 0});
            }
            parser.include_namespace("domain");
            if (parserError != eis::err_none) {
                abort("Could not set up the parser for the deck\n" + parserErrors(parser), false);
            }
            KVS::stringToNumberContext = &parser;
            KVS::stringToNumber = &runner::evaluateDeckNumber;
        }

        /**
         * Give the deferred domain values of the parser the number of cells and the boundaries of each axis
         */
        void addDomainParserVariables(){
            eis::parser<T_dataType> &parser = parserData;
            eis::error parserError = eis::err_none;
            auto &axisReg = getHarness().axisRegistry;
            for (const auto &[axisName, axis] : {std::pair{"X", "x"}, std::pair{"Y", "y"}, std::pair{"Z", "z"}}){
                //The values of an axis that does not exist stay deferred
                if (axisReg.getAxisMap().find(axisName) == axisReg.getAxisMap().end()) continue;
                const dimension &dim = axisReg.getDimension(axisName);
                auto &axisData = axisReg.getAxis(axisName);
                if (parserError == eis::err_none) parser.add_functor(std::string("domain.n") + axis,
                    [&dim](eis::count_type, const T_dataType*, void*, eis::parser<T_dataType>*, eis::status&, eis::error&){
                        return static_cast<T_dataType>(dim.getDomainCells());
                    }, parserError, {.expected_params = 0});
                if (parserError == eis::err_none) parser.add_functor(std::string("domain.") + axis + "_min",
                    [&axisData](eis::count_type, const T_dataType*, void*, eis::parser<T_dataType>*, eis::status&, eis::error&){
                        return axisData.getDomainRange()[0];
                    }, parserError, {.expected_params = 0});
                if (parserError == eis::err_none) parser.add_functor(std::string("domain.") + axis + "_max",
                    [&axisData](eis::count_type, const T_dataType*, void*, eis::parser<T_dataType>*, eis::status&, eis::error&){
                        return axisData.getDomainRange()[1];
                    }, parserError, {.expected_params = 0});
            }
            if (parserError != eis::err_none) {
                abort("Could not add the domain to the parser for the deck\n" + parserErrors(parser), false);
            }
        }
#endif

        /**
         * Read the settings of the runner from the deck
         */
        void readRunnerSettings(){
            KVS& kvs = getKVS();
            //The output format must be that of one of the writers that are available
            outputFormat = kvs.getString(deckKeys::diagnosticsFormat);
            timerThreshold = kvs.getReal("/SAMS/timer_threshold", 10.0);
            progressSteps = static_cast<size_t>(kvs.getInt("/SAMS/progress_steps", 100));
            bool formatAvailable = false;
            std::string availableFormats;
#define WRITER_DEF(param, T_writer) \
            formatAvailable |= (outputFormat == T_writer::writerName); \
            availableFormats += (availableFormats.empty() ? "" : ", ") + std::string(T_writer::writerName);
            WRITER_TYPES(0)
#undef WRITER_DEF
            if (!formatAvailable){
                abort("Invalid /diagnostics/format: " + outputFormat + ". The available formats are " + availableFormats, false);
            }
        }

        /**
         * Set the MPI decomposition from the deck
         */
        void setMPIDecomposition(){
            KVS& kvs = getKVS();
            //MPI decomposition, with the periodicity in each MPI direction from the deck. Directions that are not given are not periodic
            std::array<bool, MAX_RANK> periodic{};
            KVS::Element periodicKey = kvs.getPath("/domain/MPI_periodic", KVS::Array());
            if (!periodicKey.isType(KVS::Type::Array) || periodicKey.size() > MAX_RANK){
                abort("Invalid /domain/MPI_periodic: must be an array of at most " + std::to_string(MAX_RANK) + " booleans", false);
            }
            for (size_t i = 0; i < periodicKey.size(); i++){
                try {
                    periodic[i] = periodicKey[i].asBool();
                } catch (const std::exception &e) {
                    abort(std::string("Invalid /domain/MPI_periodic: ") + e.what(), false);
                }
            }
            //The decomposition is automatic unless the deck gives the number of ranks in each MPI direction. Directions that are not given, or are 0, are chosen automatically
            KVS::Element decompositionKey = kvs.getPath("/domain/MPI_decomposition", KVS::Element());
            if (decompositionKey.isType(KVS::Type::None)){
                getHarness().MPIManager.autoDecomposition(periodic);
            } else {
                std::array<int, MPI_DECOMPOSITION_RANK> decomposition{};
                if (!decompositionKey.isType(KVS::Type::Array) || decompositionKey.size() > MPI_DECOMPOSITION_RANK){
                    abort("Invalid /domain/MPI_decomposition: must be an array of at most " + std::to_string(MPI_DECOMPOSITION_RANK) + " integers", false);
                }
                for (size_t i = 0; i < decompositionKey.size(); i++){
                    try {
                        decomposition[i] = static_cast<int>(decompositionKey[i].asInt());
                    } catch (const std::exception &e) {
                        abort(std::string("Invalid /domain/MPI_decomposition: ") + e.what(), false);
                    }
                    if (decomposition[i] < 0){
                        abort("Invalid /domain/MPI_decomposition: the number of ranks in a direction cannot be negative", false);
                    }
                }
                //Check the request against the number of ranks, so that it fails with a clear message rather than an MPI error
                const int ranks = getHarness().MPIManager.getSize();
                int fixedRanks = 1;
                bool anyAutomatic = false;
                for (int count : decomposition){
                    if (count == 0){
                        anyAutomatic = true;
                    } else {
                        fixedRanks *= count;
                    }
                }
                if (anyAutomatic ? (ranks % fixedRanks != 0) : (fixedRanks > ranks)){
                    abort("Invalid /domain/MPI_decomposition: it does not fit the " + std::to_string(ranks) + " ranks available", false);
                }
                getHarness().MPIManager.setDecomposition(decomposition, periodic);
            }
        }

        /**
         * Print the MPI information, the output directory and the active packages, which finish the run information
         */
        void printRunInformation(){
            getHarness().MPIManager.printMPIInformation();
            SAMS::cout << "Output directory: " << std::filesystem::weakly_canonical(std::get<fileInformation>(runnerData).outputDirectory).string() << "\n";
            //The description of the problem, if the deck gives one
            KVS::Element description = getKVS().getPath("/SAMS/description", KVS::Element());
            if (!description.isType(KVS::Type::None)) SAMS::cout << "Description: " << description.asString() << "\n";
            SAMS::cout << "Active packages in runner:" << std::endl;
            printActive();
            SAMS::finishWelcomeMessage();
        }

        /**
         * Report the parts of the deck that have never been read, and write the deck that the run uses
         */
        void writeRanDeck(){
            KVS& kvs = getKVS();
            //Report the parts of the deck that have never been read
            std::stringstream unused;
            kvs.checkUnused(unused);
            if (!unused.str().empty()){
                SAMS::cerr << "\n******************************\n**** " << colorSet(VT100Red) << "UNUSED DECK ELEMENTS" << colorSet(VT100Reset) << " ****\n******************************\n\n"
                           << unused.str() << "\n******************************\n" << std::endl;
            }
            //Write the deck that the run uses, with the changes from the packages, in the format of the input deck
            if (getHarness().MPIManager.getRank() == 0){
                try {
                    const std::string &outputDirectory = std::get<fileInformation>(runnerData).outputDirectory;
                    if (inputIsEPOCH){
                        KVSConversion::toEPOCHFile(kvs, outputDirectory + "/ran.deck");
                    } else {
                        KVSConversion::toJSONFile(kvs, outputDirectory + "/ran.json", 4);
                    }
                } catch (const std::exception &e) {
                    abort(std::string("Error writing the deck that the run uses: ") + e.what(), true);
                }
            }
#ifdef USE_MPI
            MPI_Barrier(getHarness().MPIManager.getComm());
#endif
        }

        void initializePackages(){
            activateRequestedPackages();
#ifdef USE_DECK_PARSER
            initializeDeckParser();
#endif
            auto& kvs = getKVS();
            allocateActive(); //Allocate the active packages and their data packs
            runnerInteraction<runner>(*this); //Allow user to interact with the runner before initialization (e.g. to activate simulations)
            pkgInit(); //Initialize active packages
            //The output format is ADIOS2, then HDF5, then simple, whichever is available first. Packages can change it in defaultDeck
#if defined(USE_ADIOS2)
            kvs.setDefaultPath(deckKeys::diagnosticsFormat, "adios2");
#elif defined(USE_HDF5)
            kvs.setDefaultPath(deckKeys::diagnosticsFormat, "hdf5");
#else
            kvs.setDefaultPath(deckKeys::diagnosticsFormat, "simple");
#endif
            defaultDeck(); //Set the deck keys that the deck does not set
#ifdef USE_DECK_PARSER
            //The deck is complete
            try {
                addDeckConstants(kvs, parserData);
            } catch (const std::exception &e) {
                abort(e.what(), false);
            }
#endif
            readRunnerSettings();
            setMPIDecomposition();
            printRunInformation();
            registerAxes(); //Tell packages to register axes
            registerVariables(); //Tell packages to register variables (using axes)
            registerOutputVariables(); //Tell packages to register the outputs that they can write
            readOutputCollections(); //Add the collections of outputs from the deck, and resolve every collection
            defaultValues(); //Set default parameter values
            controlVariables(); //Set specific parameter values
            if (countTerminators() == 0) abort("No active package can end the run", false);
            if (restartName.empty()) setDomain(); //Set the domain for simulations
            else readRestartAxes(); //Set the axes from the dump that the run restarts from
#ifdef USE_DECK_PARSER
            addDomainParserVariables(); //The domain is now known
#endif
            decomposeAndAllocate(); //MPI decompose and allocate variables
            getVariables(); //Get the actual memory for variables
            defaultVariables(); //Set default variable values
            if (restartName.empty()) initialConditions(); //Set initial conditions
            else readRestart(); //Read the state of the simulation from the dump that the run restarts from
            setBoundaryConditions(); //Attach boundary conditions
						initializeBoundaryConditions();
            repopulateDeck(); //Put the parameters back into the deck
            writeRanDeck();
            //Write the collections that ANY package activates at the start
            writeOutput(queryOutput());
        }

        void runPackages(){
            for (;;){
                size_t step = std::get<timeState&>(runnerData).step;
                if (progressSteps > 0 && step%progressSteps == 0){
                    SAMS::cout << "Starting timestep " << step << " at time " << std::get<timeState&>(runnerData).time <<  ", dt = " << std::get<timeState&>(runnerData).dt << std::endl;
                }
                beforeStartOfTimestep(); //Set timestep value
                calculateTimestep(); //Set timestep value
                startOfTimestep(); //Start of timestep (predictor)
                halfTimestep(); //Half timestep (correction)
                auto& tData = std::get<timeState&>(runnerData);
                tData.step++;
                tData.time += tData.dt;
                endOfTimestep(); //End of timestep (remap for LARE3D)
                writeOutput(queryOutput()); //Write the collections that ANY package activates
                if (queryTerminate()){// If ANY package says to terminate, do so
                    break;
                }
                steer(); //Computational steering
            }
            writeOutput(queryOutput(true)); //Final output
            SAMS::cout << "Simulation complete. Time stepping ended at step " << std::get<timeState&>(runnerData).step << " and time " << std::get<timeState&>(runnerData).time << std::endl;
            SAMS::cout << "Reasons for ending:" << std::endl;
            for (const std::string &reason : termination.getReasons()) SAMS::cout << " - " << reason << std::endl;
            if (termination.getReasons().empty()) SAMS::cout << " - No reason given" << std::endl;
        }

        void finalizePackages(){
            callCore_finalize();
        }

        //User now activates simulations as needed

        /**
         * Print one line of the timing output, with the time and its percentage of the package's total
         */
        void printTimerLine(std::string_view label, double time, double total){
            SAMS::cout << "  " << std::left << std::setw(30) << label << std::right << std::setw(12) << std::fixed << std::setprecision(4) << time << " s"
                << std::setw(8) << std::setprecision(1) << 100.0 * time / total << " %" << std::endl;
        }

        /**
         * Print the times of every timed phase of each active package that is timed, with a blank line between packages
         * @param first Whether no package has been printed yet
         */
        template<int level=0>
        void printTimers(bool first = true){
            internalPackageInfo info = buildPackageInfo<level>();
            if (simulationActiveFlags[level] && info.timeSimulation){
                if (first) SAMS::cout << "Timings (seconds)" << std::endl;
                else SAMS::cout << std::endl;
                first = false;
                SAMS::cout << colorSet(VT100Green) << info.name << colorSet(VT100Reset) << std::endl;
                SAMS::cout << std::string(info.name.size(), '-') << std::endl;
                std::vector<std::pair<std::string_view, double>> times;
                printTimer_runnerInteraction<level, false>(times);
                printTimer_initialize<level, false>(times);
                printTimer_checkActivation<level, false>(times);
                printTimer_registerDeckElements<level, false>(times);
                printTimer_defaultDeck<level, false>(times);
                printTimer_registerAxes<level, false>(times);
                printTimer_registerVariables<level, false>(times);
                printTimer_registerOutputVariables<level, false>(times);
                printTimer_defaultValues<level, false>(times);
                printTimer_controlVariables<level, false>(times);
                printTimer_setDomain<level, false>(times);
                printTimer_registerSecondaryAxes<level, false>(times);
                printTimer_registerSecondaryVariables<level, false>(times);
                printTimer_getVariables<level, false>(times);
                printTimer_initialConditions<level, false>(times);
                printTimer_setBoundaryConditions<level, false>(times);
                printTimer_defaultVariables<level, false>(times);
                printTimer_initializeBoundaryConditions<level, false>(times);
                printTimer_repopulateDeck<level, false>(times);
                printTimer_beforeStartOfTimestep<level, false>(times);
                printTimer_startOfTimestep<level, false>(times);
                printTimer_halfTimestep<level, false>(times);
                printTimer_endOfTimestep<level, false>(times);
                printTimer_calculateTimestep<level, false>(times);
                printTimer_getTimestep<level, false>(times);
                printTimer_queryTerminate<level, false>(times);
                printTimer_queryOutput<level, false>(times);
                printTimer_registerDumpMeshes<level, false>(times);
                printTimer_registerDumpVariables<level, false>(times);
                printTimer_writeDumpMeshes<level, false>(times);
                printTimer_writeDumpVariables<level, false>(times);
                printTimer_restart<level, false>(times);
                printTimer_steer<level, false>(times);
                printTimer_finalize<level, false>(times);
                double total = 0.0;
                for (const auto &entry : times) total += entry.second;
                //Phases below the threshold percentage of the package's total are collapsed into Other
                double other = 0.0;
                for (const auto &[label, time] : times){
                    if (time < timerThreshold / 100.0 * total) other += time;
                    else printTimerLine(label, time, total);
                }
                if (other > 0.0) printTimerLine("Other", other, total);
                SAMS::cout << colorSet(VT100Yellow);
                printTimerLine("Total", total, total);
                SAMS::cout << colorSet(VT100Reset);
            }
            if constexpr (level < sizeof...(Packages)-1){
                printTimers<level+1>(first);
            }
        }

        void finalize(){
#ifdef USE_TIMER
            printTimers();
#endif
            //Delete the writers, which may flush their files
            writers.clear();

            //Delete the packages and then the data packs. The rest of the runner data is not allocated by the runner
            auto release = [](auto &element){
                if constexpr (std::is_pointer_v<std::remove_reference_t<decltype(element)>>){
                    delete element;
                    element = nullptr;
                }
            };
            std::apply([&](auto&... elements){ (release(elements), ...); }, runnerData);

            harness& h = getHarness();
            h.finalize();
        }

    };
    
} // namespace SAMS


#endif // SAMS_RUNNER_H

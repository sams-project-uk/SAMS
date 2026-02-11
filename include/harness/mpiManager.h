#ifndef SAMS_MPI_MANAGER_H
#define SAMS_MPI_MANAGER_H
#include <iostream>
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "axisRegistry.h"
#include "dimension.h"
#include "typeRegistry.h"
#include "mpiDefaultTypes.h"
#include <map>
#include <unordered_map>
#include <string>

namespace SAMS
{
    template<int Dims>
    class MPIManager;

    namespace MPI{

        namespace{

            #ifdef USE_MPI
            //Create a map from MPI_THREAD_* to a string for error messages
            inline static std::map<int, std::string> threadLevelMap = {
                {MPI_THREAD_SINGLE, "MPI_THREAD_SINGLE"},
                {MPI_THREAD_FUNNELED, "MPI_THREAD_FUNNELED"},
                {MPI_THREAD_SERIALIZED, "MPI_THREAD_SERIALIZED"},
                {MPI_THREAD_MULTIPLE, "MPI_THREAD_MULTIPLE"}
            };

            #endif

        }


        inline void initialize([[maybe_unused]] int *argc, [[maybe_unused]] char ***argv)
        {
            #ifdef USE_MPI
            int rank = 0;
            //Create a map from MPI_THREAD_* to a string for error messages
            std::map<int, std::string> threadLevelMap = {
                {MPI_THREAD_SINGLE, "MPI_THREAD_SINGLE"},
                {MPI_THREAD_FUNNELED, "MPI_THREAD_FUNNELED"},
                {MPI_THREAD_SERIALIZED, "MPI_THREAD_SERIALIZED"},
                {MPI_THREAD_MULTIPLE, "MPI_THREAD_MULTIPLE"}
            };

            int requested = MPI_THREAD_FUNNELED;
            int provided;
            MPI_Init_thread(argc, argv, requested, &provided);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (provided < requested)
            {
                if (rank == 0){
                    std::cerr << "Error: The MPI library does not provide the required threading level" << std::endl;
                    std::cerr << "Requested: " << threadLevelMap[requested] << ", Provided: " << threadLevelMap[provided] << std::endl;
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (provided > requested)
            {
                if (rank == 0){
                    std::cerr << "Warning: The MPI library provides a higher threading level than required" << std::endl;
                    std::cerr << "Requested: " << threadLevelMap[requested] << ", Provided: " << threadLevelMap[provided] << std::endl;
                    if (provided == MPI_THREAD_MULTIPLE) {
                        std::cerr << "MPI_THREAD_MULTIPLE is often poorly performing and SAMS does not take advantage of it." << std::endl;
                    }
                }
            }
#ifdef MPI_ERRORS_RETURN
            // Set error handler to return errors
            MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
#else
            // Explicitly set to fatal errors
            MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_FATAL);
#endif
            //Set the rank for the SAMS output streams
            //Only rank 0 will output anything
            cout.setRank(rank);
            cerr.setRank(rank);
            debug1.setRank(rank);
            debug2.setRank(rank);
            debug3.setRank(rank);
            //These output from all ranks, and have already been set to do so
            debugAll1.setRank(rank);
            debugAll2.setRank(rank);
            debugAll3.setRank(rank);
            #endif
        }

        inline void initialize([[maybe_unused]] int &argc, [[maybe_unused]] char **&argv)
        {
            initialize(&argc, &argv);

        }

        inline void finalize(){
            #ifdef USE_MPI
            MPI_Finalize();
            #endif
        }


    }

    template <int i_Dims = 3>
    class MPIManager
    {
    public:
    static const int Dims = i_Dims;
    private:

        axisRegistry &ar;

        #ifdef USE_MPI
          //Create a map from an integer to an MPI combiner type string for naming purposes
          inline static std::map<int, std::string> mpiCombinerMap = {
              {MPI_COMBINER_NAMED, "MPI_COMBINER_NAMED"},
              {MPI_COMBINER_DUP, "MPI_COMBINER_DUP"},
              {MPI_COMBINER_CONTIGUOUS, "MPI_COMBINER_CONTIGUOUS"},
              {MPI_COMBINER_VECTOR, "MPI_COMBINER_VECTOR"},
              {MPI_COMBINER_HVECTOR, "MPI_COMBINER_HVECTOR"},
              {MPI_COMBINER_INDEXED, "MPI_COMBINER_INDEXED"},
              {MPI_COMBINER_HINDEXED, "MPI_COMBINER_HINDEXED"},
              {MPI_COMBINER_STRUCT, "MPI_COMBINER_STRUCT"},
              {MPI_COMBINER_SUBARRAY, "MPI_COMBINER_SUBARRAY"},
              {MPI_COMBINER_DARRAY, "MPI_COMBINER_DARRAY"},
              {MPI_COMBINER_F90_REAL, "MPI_COMBINER_F90_REAL"},
              {MPI_COMBINER_F90_COMPLEX, "MPI_COMBINER_F90_COMPLEX"},
              {MPI_COMBINER_F90_INTEGER, "MPI_COMBINER_F90_INTEGER"},
              {MPI_COMBINER_RESIZED, "MPI_COMBINER_RESIZED"}
          };
          #endif

#ifdef MPI_ERRORS_RETURN
#define checkMPIError(fn) checkMPIErrorCore(fn, __FILE__, __LINE__)
        inline void checkMPIErrorCore(int errcode, const char *file, int line)
        {
            if (errcode != MPI_SUCCESS)
            {
                char errorString[MPI_MAX_ERROR_STRING];
                int lengthOfErrorString;
                MPI_Error_string(errcode, errorString, &lengthOfErrorString);
                std::cerr << "MPI error at " << file << ":" << line << " - " << std::string(errorString, lengthOfErrorString) << " on rank " << rank << std::endl;
                MPI_Abort(rootComm, errcode);
            }
        }
#else
#define checkMPIError(fn) fn
#endif

        int rank = 0;
        int size = 1;
        MPI_Comm comm = MPI_COMM_WORLD;
        MPI_Comm rootComm = MPI_COMM_WORLD;
        std::array<int, Dims> dims;
        std::array<int, MAX_RANK> coords;
        std::array<int, MAX_RANK> periods;

        std::array<int, 2 * Dims> neighbors;

        /**
         * Initialise the MPIManager with default values but a custom communicator
         */
        void defaultInit(MPI_Comm customComm)
        {
           //Set up neighbours to MPI_PROC_NULL by default
            for (int i = 0; i < 2 * Dims; i++) {
                neighbors[i] = MPI_PROC_NULL;
            }

            //Now set dims, coords and periods to default values
            for (int i = 0; i < Dims; i++) {
                dims[i] = 1;
            }

            for (int i = 0; i < MAX_RANK; i++) {
                coords[i] = 0;
                periods[i] = 0;
            }

            rootComm = customComm;
            comm = customComm;

#ifdef USE_MPI
            int init;
            checkMPIError(MPI_Initialized(&init));
            if (!init) throw std::runtime_error("Error: MPI environment not initialized before MPIManager initialization\n");
            checkMPIError(MPI_Comm_rank(rootComm, &rank));
            checkMPIError(MPI_Comm_size(rootComm, &size));
#ifdef MPI_ERRORS_RETURN
            // Set error handler to return errors
            checkMPIError(MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN));
#else
            // Explicitly set to fatal errors
            checkMPIError(MPI_Comm_set_errhandler(comm, MPI_ERRORS_FATAL));
#endif
#endif
        }

        /**
         * Type to hold an MPI_Datatype along with a reference count for shared usage
         * Deletes the type when the reference count reaches zero or when forced to delete
         */
        struct MPITypeHolder
        {
            COUNT_TYPE refCount=1;
            MPI_Datatype mpiType = MPI_DATATYPE_NULL;
            std::string Name=""; //Name here is just used for debug output

            /*
            * Release the MPI_Datatype
            */
            void releaseType(){
            #ifdef USE_MPI
                if (mpiType != MPI_DATATYPE_NULL){
                    SAMS::debugAll3 << "Releasing MPI type in destructor " << Name << std::endl;
                    MPI_Type_free(&mpiType);
                }
            #endif
            }

            //Delete default constructor and copy operations to avoid accidental copies
            MPITypeHolder() = delete;
            MPITypeHolder(const MPITypeHolder &) = delete;
            MPITypeHolder &operator=(const MPITypeHolder &) = delete;

            //Move constructor to allow "normal" operation in containers
            MPITypeHolder(MPITypeHolder &&other) noexcept
                : refCount(other.refCount), mpiType(other.mpiType), Name(std::move(other.Name))
            {
                other.mpiType = MPI_DATATYPE_NULL;
                other.refCount = 0;
            }
            
            /**
             * Should never come up, but delete move assignment operator just in case
             */
            MPITypeHolder &operator=(MPITypeHolder &&other) = delete;

            /**
             * Constructor from MPI_Datatype
             */
            MPITypeHolder(MPI_Datatype type)
                : mpiType(type) {}

            MPITypeHolder(MPI_Datatype type, const std::string &name)
                : mpiType(type), Name(name) {}

            /**
             * Destructor to free the MPI_Datatype
             */
            ~MPITypeHolder(){
                releaseType();
            }

            /**
             * Decrement the reference count and return the new value
             */
            COUNT_TYPE decRef(){
                refCount--;
                if (refCount == 0){
                    releaseType();
                }
                return refCount;
            }

            /**
             * Increment the reference count
             */
            COUNT_TYPE incRef(){
                refCount++;
                return refCount;
            }

            /**
             * Get the current reference count
             */
            COUNT_TYPE getRefCount() const {
                return refCount;
            }

            /**
             * Conversion operator to MPI_Datatype
             */
            operator MPI_Datatype() const {
                return mpiType;
            }
        };

        /**
         * Cache of created MPI_Datatype objects for subarrays
         */
        std::map<std::string, MPITypeHolder> mpiSubarrayTypeCache;

        /**
         * Get a forward cache entry and return the MPI_Datatype. Increment the reference count if the "incRef" parameter is true
         * @param Name The name of the MPI_Datatype to look up
         * @param incRef Whether to increment the reference count of the found MPI_Datatype (default false)
         * @return The found MPI_Datatype, or MPI_DATATYPE_NULL if not found
         */
        MPI_Datatype getCacheEntry([[maybe_unused]] const std::string &Name, [[maybe_unused]] bool incRef=false)
        {
            #ifdef USE_MPI
            auto it = mpiSubarrayTypeCache.find(Name);
            if (it != mpiSubarrayTypeCache.end())
            {
                SAMS::debugAll3 << "Found MPI type in cache: " << Name << " with refCount " << it->second.getRefCount() << std::endl;
                if (incRef){
                    it->second.incRef();
                }
                return it->second.mpiType;
            }
            #endif
            return MPI_DATATYPE_NULL;
        }

        /**
         * Use MPI_type_get_envelope and MPI_Type_get_contents to get the name of a built in MPI_Datatype
         * @param mpiType The MPI_Datatype to get the name of
         * @return The name of the MPI_Datatype, or empty string if not found
         */
        std::string buildMPIGenericName([[maybe_unused]] MPI_Datatype mpiType)
        {
            #ifdef USE_MPI
            int nints, nadds, ntypes, combiner;
            checkMPIError(MPI_Type_get_envelope(mpiType, &nints, &nadds, &ntypes, &combiner));
            std::vector<int> array_of_integers(nints);
            std::vector<MPI_Aint> array_of_addresses(nadds);
            std::vector<MPI_Datatype> array_of_types(ntypes);
            //If our type is a built in type, just return its name
            if (combiner == MPI_COMBINER_NAMED || (nints==0 && nadds==0 && ntypes==0)){
                //This is a built in type
                char name[MPI_MAX_OBJECT_NAME];
                int resultlen;
                checkMPIError(MPI_Type_get_name(mpiType, name, &resultlen));
                return std::string(name, resultlen);
            }
            //Type is a derived type so generate a contents
            checkMPIError(MPI_Type_get_contents(mpiType, nints, nadds, ntypes,
                                               array_of_integers.data(),
                                               array_of_addresses.data(),
                                               array_of_types.data()));
            //Check if we have a known combiner
            std::string combinerName;
            auto combIt = mpiCombinerMap.find(combiner);
            if (combIt == mpiCombinerMap.end()){
                combinerName = "MPI_COMBINER_UNKNOWN(" + std::to_string(combiner) + ")";
            } else {
                combinerName = combIt->second;
            }
            //Build a name based on the combiner and contents
            std::string Name = "MPI_Generic_type_combiner(" + combinerName + ")_ints(";
            for (int i = 0; i < nints; i++){
                Name += std::to_string(array_of_integers[i]);
                if (i < nints - 1){
                    Name += ",";
                }
            }
            Name += ")_addr(";
            for (int i = 0; i < nadds; i++){
                Name += std::to_string(array_of_addresses[i]);
                if (i < nadds - 1){
                    Name += ",";
                }
            }
            Name += ")_types(";
            for (int i = 0; i < ntypes; i++){
                //Convert MPI_Datatype to type name
                Name += buildMPIGenericName(array_of_types[i]);
                if (i < ntypes - 1){
                    Name += ",";
                }
            }
            Name += ")";
            return Name;
            #else
            return "";
            #endif
        }

        /**
         * Use the built in MPI_Type_set_name to set the name of an MPI_Datatype
         * @param mpiType The MPI_Datatype to name
         * @param name The name to set
         */
        void setMPITypeName([[maybe_unused]] MPI_Datatype mpiType, [[maybe_unused]] const std::string &name)
        {
            #ifdef USE_MPI
            //Check if the string is later than MPI_MAX_OBJECT_NAME
            char truncatedName[MPI_MAX_OBJECT_NAME];
            std::strncpy(truncatedName, name.c_str(), MPI_MAX_OBJECT_NAME-1);
            //Always null terminating is quicker than checking length first
            truncatedName[MPI_MAX_OBJECT_NAME-1] = '\0';
            checkMPIError(MPI_Type_set_name(mpiType, name.c_str()));
            #endif
        }


        /**
         * Create an MPI datatype for a slice of a variable defined without the use of any harness objects
         * @param rank The rank of the variable
         * @param domainLB Array of size rank specifying the global lower bounds of the variable
         * @param domainUB Array of size rank specifying the global upper bounds of the variable
         * @param LB Array of size rank specifying the lower bounds of the slice
         * @param UB Array of size rank specifying the upper bounds of the slice
         * @param baseType The base MPI_Datatype of the variable
         */
        MPI_Datatype createArraySliceType([[maybe_unused]] int rank, [[maybe_unused]] SIGNED_INDEX_TYPE *domainLB, [[maybe_unused]] SIGNED_INDEX_TYPE *domainUB, [[maybe_unused]] SIGNED_INDEX_TYPE *LB, [[maybe_unused]] SIGNED_INDEX_TYPE *UB, [[maybe_unused]] MPI_Datatype baseType){
            #ifdef USE_MPI
            int starts[MAX_RANK], sizes[MAX_RANK], subsizes[MAX_RANK];
            for (int i = 0; i < rank; i++)
            {
                sizes[i] = domainUB[i] - domainLB[i]+1;
                subsizes[i] = UB[i] - LB[i] + 1;
                starts[i] = LB[i] - domainLB[i];
            }
            //Add the type to the cache or get existing type
            return createMPISubarrayType(rank, sizes, subsizes, starts, baseType, MPI_ORDER_C);
            #else
            return MPI_DATATYPE_NULL;
            #endif
        }

        /**
         * Build an MPI datatype for a slice of a variable defined by lower and upper bounds in each dimension
         */
        MPI_Datatype createArraySliceType(int rank, dimension* dims, SIGNED_INDEX_TYPE *LB, SIGNED_INDEX_TYPE *UB, MPI_Datatype baseType){

            SIGNED_INDEX_TYPE domainLB[MAX_RANK], domainUB[MAX_RANK];
            for(int i=0; i<rank; i++){
            domainLB[i] = dims[i].getLocalLB();
            domainUB[i] = dims[i].getLocalUB();
            }
            return createArraySliceType(rank, domainLB, domainUB, LB, UB, baseType);

        }


        /**
         * Delete a specified MPI_Datatype from the caches and free it
         * @param mpiType The MPI_Datatype to delete
         * @param forceDelete If true, delete the type even if the reference count is not
         */
        bool deleteMPIType([[maybe_unused]] MPI_Datatype mpiType, [[maybe_unused]] bool forceDelete = false)
        {
            #ifdef USE_MPI
            try{
                std::string info = buildMPIGenericName(mpiType);
                auto itF = mpiSubarrayTypeCache.find(info);
                MPITypeHolder& typeHolder = itF->second;
                COUNT_TYPE newRefCount = typeHolder.decRef();
                if ((newRefCount == 0) || forceDelete){
                    SAMS::debugAll3 << "Releasing MPI subarray type: " << info << std::endl;
                    //Delete it from the forward cache
                    mpiSubarrayTypeCache.erase(itF);
                    return true;
                } else {
                    SAMS::debugAll3 << "Decremented reference count for MPI type: " << info << " to " << newRefCount << std::endl;
                    return false;
                }
                //MPI_Type_free is called in the destructor of MPITypeHolder
            } catch (const std::runtime_error& e){
                SAMS::debugAll3 << "*** WARNING *** Freeing unregistered MPI type!. Type is freed, but this is probably an error." << std::endl;
                MPI_Type_free(&mpiType); //Just free it directly
                return true;
            }
            #else
            return true;
            #endif
        }


        void registerMPIType([[maybe_unused]] MPI_Datatype mpiType, [[maybe_unused]] const std::string &name)
        {
            #ifdef USE_MPI
            mpiSubarrayTypeCache.emplace(name, MPITypeHolder(mpiType, name));
            setMPITypeName(mpiType, name);
            SAMS::debugAll3 << "Registered new MPI type: " << name << std::endl;
            #endif
        }

        MPI_Datatype checkoutCore([[maybe_unused]] const std::string &typeName)
        {
            #ifdef USE_MPI
            auto it = mpiSubarrayTypeCache.find(typeName);
            if (it != mpiSubarrayTypeCache.end())
            {
                it->second.incRef();
                SAMS::debugAll3 << "Checked out existing MPI type: " << typeName << " with new reference count " << it->second.getRefCount() << std::endl;
                return it->second.mpiType;
            }
            else
            {
                throw std::runtime_error("Error: MPI_Datatype " + typeName + " not found in MPIManager cache");
            }
            #else
            return MPI_DATATYPE_NULL;
            #endif
        }


        /**
         * Class to extract member pointer traits
         * @tparam M The member pointer type
         */
        template<typename M> struct memberPointerTraits;
        /**
         * Specialisation of memberPointerTraits for member pointers
         * @tparam C The class type
         * @tparam M The member type
         * classType is the class that the type is a member of
         * memberType is the type of the member
         */
        template<typename C, typename M>
        struct memberPointerTraits<M C::*>
        {
            using classType  = C;
            using memberType = M;
        };

        template<int level=0, typename T_classIn, int N, auto c, auto... o>
        void buildTypeFromMemberPointers(T_classIn *base, std::array<int, N> &blocklengths, std::array<MPI_Aint, N> &displacements, std::array<MPI_Datatype, N> &types)
        {
            //First set the type from the type registry
            using MemberType = typename memberPointerTraits<decltype(c)>::memberType;
            //Member must be trivially copyable to be used in MPI communication
            static_assert(std::is_trivially_copyable<MemberType>::value, "Error: Member type must be trivially copyable to create MPI datatype");
            types[level] = gettypeRegistry().getMPIType<MemberType>();
            //Get blocklenght by comparing sizeof member type to the size of the MPI type
            size_t memberSize = sizeof(MemberType);
            size_t mpiTypeSize = gettypeRegistry().getSize<MemberType>();
            blocklengths[level] = static_cast<int>(memberSize / mpiTypeSize);
            //Can't use offsetof, but use a nullptr cast to the class type to get the member address
            MPI_Aint memberAddress = static_cast<MPI_Aint>(reinterpret_cast<char*>(&(base->*c)) - reinterpret_cast<char*>(base));
            displacements[level] = memberAddress;

            //Recurse if we have more members
            if constexpr(sizeof...(o) > 0){
                buildTypeFromMemberPointers<level+1, T_classIn, N, o...>(base, blocklengths, displacements, types);
            }
        }


        MPI_Datatype cacheType([[maybe_unused]] MPI_Datatype newType){
            #ifdef USE_MPI
            std::string Name = buildMPIGenericName(newType);
            //If type already in cache, return existing type
            MPI_Datatype existingType = getCacheEntry(Name, true);
            if (existingType != MPI_DATATYPE_NULL) {
                checkMPIError(MPI_Type_free(&newType)); //Free the newly created type as we are returning existing one
                return existingType;
            }
            //Otherwise create new type
            checkMPIError(MPI_Type_commit(&newType));
            registerMPIType(newType, Name);
            return newType;
            #else
            return MPI_DATATYPE_NULL;
            #endif
        }

    public:

        /**
         * Set up the decomposition of the MPI Cartesian communicator
         * @param decomposition Array of size Dims specifying the number of processors in each dimension
         * @param isPeriodic Array of size Dims specifying whether each dimension is periodic (1
         */
        template<std::size_t N = Dims>
        void setDecomposition([[maybe_unused]] const std::array<int, Dims> &decomposition, [[maybe_unused]] const std::array<bool, N> &isPeriodic)
        {
            static_assert(N <= MAX_RANK, "Error: isPeriodic array size exceeds MAX_RANK");
            //Always want periodic domain info
            for (std::size_t i = 0; i < N; i++)
            {
                periods[i] = isPeriodic[i]? 1 : 0;
            }
            #ifdef USE_MPI
            int totalProcs = 1;
            bool anyZero = false;
            bool allZero = true;
            for (int i = 0; i < Dims; i++)
            {
                dims[i] = decomposition[i];
                if (dims[i] == 0){
                    anyZero = true;
                } else {
                    allZero = false;
                }
            }
            if (anyZero){
                if (allZero){
                    SAMS::cout << "Using automatic MPI decomposition : ";
                } else {
                    SAMS::cout << "Using partially automatic MPI decomposition : ";
                }
                checkMPIError(MPI_Dims_create(size, Dims, dims.data()));
            } else {
                SAMS::cout << "Using specified MPI decomposition: " << totalProcs;
            }
            for (int i = 0; i < Dims; i++) totalProcs *= dims[i];
            //Not really necessary but I do hate it when codes pluralise incorrectly
            if (totalProcs > 1){
                SAMS::cout << " processors (";
            } else {
                SAMS::cout << " processor (";
            }
            for (int i = 0; i < Dims; i++) {
                SAMS::cout << dims[i];
                if (i < Dims - 1)
                    SAMS::cout << " x ";
            }
            SAMS::cout << ")" << std::endl;
            if (totalProcs != size)
            {
                if (rank == 0)
                {
                    SAMS::cerr << "Not all processors are used in the decomposition" << std::endl;
                    SAMS::cerr << "Processors available " << size << ", processors used " << totalProcs << std::endl;
                    SAMS::cerr << "Processors in decomposition: " << totalProcs << " (";
                    for (int i = 0; i < Dims; i++)
                    {
                        SAMS::cerr << dims[i];
                        if (i < Dims - 1)
                            SAMS::cerr << " x ";
                    }
                    SAMS::cerr << ")" << std::endl;
                    SAMS::cerr << "Wasted processors : " << size - totalProcs << std::endl;
                }
            }
            // Create the Cartesian communicator
            checkMPIError(MPI_Cart_create(rootComm, Dims, dims.data(), periods.data(), 1, &comm));
            checkMPIError(MPI_Cart_coords(comm, rank, Dims, coords.data()));
            // Set up neighbouring ranks
            for (int i = 0; i < Dims; i++)
            {
                int lower, upper;
                checkMPIError(MPI_Cart_shift(comm, i, 1, &lower, &upper));
                neighbors[2 * i] = lower;
                neighbors[2 * i + 1] = upper;
            }
#else
            for (int i = 0; i < Dims; i++)
            {
                coords[i] = 0;
                neighbors[2 * i] = MPI_PROC_NULL;
                neighbors[2 * i + 1] = MPI_PROC_NULL;
                dims[i] = 1;
            }
            SAMS::cout << "Running without MPI parallelism" << std::endl;
            SAMS::cout << "DO NOT TRY TO RUN THIS VERSION OF SAMS ACROSS NODES!" << std::endl;
#endif
        }


        /**
         * Use MPI_Dims_create to automatically create a decomposition based on the number of processors and whether each dimension is periodic
         * @param isPeriodic Array of size Dims specifying whether each dimension is periodic (1 to MAX_RANK)
         * Note that you can specify periodicity for an non-decomposed dimension
         */
        template<std::size_t N = Dims>
        void autoDecomposition(const std::array<bool,N> &isPeriodic)
        {
            std::array<int,Dims> decomposition;
#ifdef USE_MPI
            for (int i = 0; i < Dims; i++) decomposition[i] = 0; //Let MPI_Dims_create decide
#else
            for (int i = 0; i < Dims; i++) decomposition[i] = 1; //Without MPI just use 1
#endif
            setDecomposition(decomposition, isPeriodic);
        }


        /**
         * Get the rank of the current processor
         */
        int getRank() const
        {
            return rank;
        }

        /**
         * Get the size of the current communicator
         */
        int getSize() const
        {
            return size;
        }

        /**
         * Get the Cartesian communicator
         * Before decomposition this is the communicator passed to the constructor
         * After decomposition this is the Cartesian communicator
         */
        MPI_Comm getComm() const
        {
            return comm;
        }

        /**
         * Get the neighbours of this processor in each dimension
         * @return Array of size Dims*2 containing the ranks of the neighbours in each dimension
         * Neighbour order is: dim0_lower, dim0_upper, dim1_lower, dim1_upper, ...
         */
        std::array<int, Dims * 2> getNeighbors() const
        {
            return neighbors;
        }

        /**
         * Get the size of the Cartesian decomposition in each dimension
         * @return Array of size Dims containing the number of processors in each dimension
         */
        std::array<int, Dims> getDims() const
        {
            return dims;
        }

        /**
         * Get the coordinates of this processor in the Cartesian decomposition
         * @return Array of size Dims containing the coordinates of this processor
         */
        std::array<int, MAX_RANK> getCoords() const
        {
            return coords;
        }

        /**
         * Check if a given axis is at the edge of the decomposition including the effect of periodicity
         */
        bool isEdge(int axis, SAMS::domain::edges edgeType) const
        {
            //Always on an edge if axis is not decomposed
            if (axis > Dims-1){
                return true;
            }
            int neighborIndex = axis * 2 + (edgeType==SAMS::domain::edges::lower ? 0 : 1);
            return neighbors[neighborIndex] == MPI_PROC_NULL;
        }

        /**
         * This detects if you are at the edge of the decomposition WITHOUT considering periodicity
         */
        bool isEdgeNP(int axis, SAMS::domain::edges edgeType) const
        {
            if (edgeType == SAMS::domain::edges::lower) {
                return coords[axis] == 0;
            } else {
                return coords[axis] == dims[axis]-1;
            }
        }

        /**
         * Check if a given axis is periodic
         */
        bool isPeriodic(int axis) const
        {
            return periods[axis] != 0;
        }

        /**
         * Decompose a single axis by setting the local elements for that axis based on the global elements and the MPI decomposition
         * @param axisName The name of the axis to decompose
         */
        void decomposeAxis(const std::string &axisName)
        {
            int axis = ar.getMPIAxis(axisName);
            size_t globalElements = ar.getDomainElements(axisName, staggerType::CENTRED);
            size_t localElements = 0;
            if (axis >= 0)
            {
                // Decompose the axis, distributing any remainder to the first few ranks
                localElements = globalElements / dims[axis];
                size_t remainder = globalElements % dims[axis];
                if (static_cast<std::size_t>(coords[axis]) < remainder)
                {
                    localElements++;
                }
            } else {
                // Axis is not decomposed, so all elements are local
                localElements = globalElements;
            }
            ar.setLocalDomainElements(axisName, localElements, staggerType::CENTRED);

            /*if (axis < 0){
                //Axis is not decomposed so nothing more to do
                return;
            }*/
           #ifdef USE_MPI
            //Now set part of the global axis that is on this processor
            size_t localLB = 0;
            for(int i=0; i<coords[axis]; i++){
                size_t procElements = globalElements / dims[axis];
                size_t remainder = globalElements % dims[axis];
                if (i < remainder)
                {
                    procElements++;
                }
                localLB += procElements;
            }
            #else
            size_t localLB = 0;
            #endif
            size_t localUB = localLB + localElements;
            auto &axisRef = ar.getAxis(axisName);
            axisRef.dim.setGlobalBounds(localLB, localUB);
            //axisRef.dim.setPeriodic(periods[axis] != 0);
        }

        /**
         * Create an MPI subarray type for a specified set of sizes, subsizes and starts. Caches created types to avoid duplication
         * @param rank The rank of the subarray
         * @param sizes Array of sizes of the full array
         * @param subsizes Array of sizes of the subarray
         * @param starts Array of starting indices of the subarray
         * @param baseType The base MPI_Datatype
         * @param layout The layout of the array (default MPI_ORDER_C)
         */
        MPI_Datatype createMPISubarrayType([[maybe_unused]] int rank, [[maybe_unused]] const int* sizes, [[maybe_unused]] const int* subsizes, [[maybe_unused]] const int* starts, [[maybe_unused]] MPI_Datatype baseType, [[maybe_unused]] int layout=MPI_ORDER_C)
        {
            #ifdef USE_MPI
            MPI_Datatype newType;
            checkMPIError(MPI_Type_create_subarray(
                rank,
                sizes,
                subsizes,
                starts,
                layout, 
                baseType,
                &newType));
                return cacheType(newType);
            #else
            return MPI_DATATYPE_NULL;
            #endif
        }

        /**
         * Create an MPI subarray type for a specified set of sizes, subsizes and starts. Caches created types to avoid duplication. Takes std::array inputs rather than raw pointers
         * @param sizes Array of sizes of the full array
         * @param subsizes Array of sizes of the subarray
         * @param starts Array of starting indices of the subarray
         * @param baseType The base MPI_Datatype
         * @param layout The layout of the array (default MPI_ORDER_C)
         */
        template<int N>
        MPI_Datatype createMPISubarrayType([[maybe_unused]] const std::array<int,N> &sizes, [[maybe_unused]] const std::array<int,N> &subsizes, [[maybe_unused]] const std::array<int,N> &starts, [[maybe_unused]] MPI_Datatype baseType, [[maybe_unused]] int layout=MPI_ORDER_C)
        {
            #ifdef USE_MPI
            return createMPISubarrayType(N, sizes.data(), subsizes.data(), starts.data(), baseType, layout);
            #else
            return MPI_DATATYPE_NULL;
            #endif
        }

        /**
         * Build an MPI datatype for a slice of a portableArray variable defined by lower and upper bounds in each dimension
         * @param arrayVar The portableArray variable to create the slice type for
         * @param ranges portableWrapper::range objects defining the lower and upper bounds in each dimension
         */
        template<typename T, int rank, portableWrapper::arrayTags tag, typename... T_Ranges>
        MPI_Datatype createArraySliceType([[maybe_unused]] portableWrapper::portableArray<T, rank, tag> &arrayVar, [[maybe_unused]] T_Ranges... ranges)
        {
            #ifdef USE_MPI
            static_assert(sizeof...(ranges) == rank, "Error: Number of ranges must match array rank");
            if (!arrayVar.isContiguous()){
                throw std::runtime_error("Error: Cannot create MPI datatype for non-contiguous portableArray");
            }
            SIGNED_INDEX_TYPE LB[MAX_RANK], UB[MAX_RANK];
            std::array<portableWrapper::Range, rank> rangeArray = {ranges...};
            for (int i = 0; i < rank; i++)
            {
                LB[i] = rangeArray[i].lower_bound;
                UB[i] = rangeArray[i].upper_bound;
            }
            return createArraySliceType(rank, arrayVar.getLowerBounds(), arrayVar.getUpperBound(), LB, UB, gettypeRegistry().getMPIType<T>());
            #else
            return MPI_DATATYPE_NULL;
            #endif
        }

        /**
         * Create an MPI_Type_create_struct for a specified set of blocklengths, displacements and types
         * @param elements The number of elements in the struct
         * @param array_of_blocklengths Array of blocklengths
         * @param array_of_displacements Array of displacements
         * @param array_of_types Array of MPI_Datatypes
         */
        
        MPI_Datatype createMPIStructType( [[maybe_unused]] int elements, [[maybe_unused]] const int *array_of_blocklengths, [[maybe_unused]] const MPI_Aint *array_of_displacements, [[maybe_unused]] const MPI_Datatype *array_of_types)
        {
            #ifdef USE_MPI
            MPI_Datatype newType;
            checkMPIError(MPI_Type_create_struct(
                elements,
                array_of_blocklengths,
                array_of_displacements,
                array_of_types,
                &newType));
                return cacheType(newType);
            #else
            return MPI_DATATYPE_NULL;
            #endif
        }

        /**
         * Create an MPI_Type_create_struct for a specified set of blocklengths, displacements and types
         */
        template<int N>
        MPI_Datatype createMPIStructType([[maybe_unused]] const std::array<int,N> &array_of_blocklengths, [[maybe_unused]] const std::array<MPI_Aint,N> &array_of_displacements, [[maybe_unused]] const std::array<MPI_Datatype,N> &array_of_types)
        {
            #ifdef USE_MPI
            return createMPIStructType(N, array_of_blocklengths.data(), array_of_displacements.data(), array_of_types.data());
            #else
            return MPI_DATATYPE_NULL;
            #endif
        }

        /**
         * Create an MPI_Type_create_struct for a specified class and its members
         * @tparam T_First The first member pointer
         * @tparam T_Others The other member pointers
         */
        template<auto T_First, auto... T_Others>
        MPI_Datatype createMPIStructType(){
            #ifdef USE_MPI
            constexpr int numMembers = 1 + sizeof...(T_Others);
            std::array<int, numMembers> blocklengths = {};
            std::array<MPI_Aint, numMembers> displacements = {};
            std::array<MPI_Datatype, numMembers> types = {};
            using classType = typename memberPointerTraits<decltype(T_First)>::classType;
            static_assert(std::is_standard_layout<classType>::value, "Error: Class type must be standard layout to create MPI struct type");
            #ifndef STRICT_STANDARDS
            buildTypeFromMemberPointers<0, classType, numMembers, T_First, T_Others...>(nullptr, blocklengths, displacements, types);
            #else
            classType exampleObj; //Per strict standards we can't use nullptr for pointer arithmetic, so create an example object
            buildTypeFromMemberPointers<0, classType, numMembers, T_First, T_Others...>(exampleObj, blocklengths, displacements, types);
            #endif
            return createMPIStructType<numMembers>(blocklengths, displacements, types);
            #else
            return MPI_DATATYPE_NULL;
            #endif
        }

        /**
         * Create an MPI_Type_create_struct for a specified class and it's members, adding the type to the cache
         * @tparam T_First The first member pointer
         * @tparam T_Others The other member pointers
         * @param typeName The name to assign to the registered type (applies only to the registered type, NOT the MPI_Datatype)
         * @return The created MPI_Datatype
         */
        template<auto T_First, auto... T_Others>
        std::tuple<SAMS::typeHandle, MPI_Datatype> createAndRegisterMPIStructType(const std::string &typeName){
            MPI_Datatype newType = createMPIStructType<T_First, T_Others...>();
            SAMS::typeHandle handle = gettypeRegistry().registerType<typename memberPointerTraits<decltype(T_First)>::classType>(typeName, newType);
            return std::make_tuple(handle, newType);
        }

        /**
         * Create an MPI_Type_create_struct for a specified class and it's members, adding the type to the cache. Generate the name using demangling
         * @tparam T_First The first member pointer
         * @tparam T_Others The other member pointers
         * @return The created MPI_Datatype
         */
        template<auto T_First, auto... T_Others>
        std::tuple<SAMS::typeHandle, MPI_Datatype> createAndRegisterMPIStructType(){
            using classType = typename memberPointerTraits<decltype(T_First)>::classType;
            MPI_Datatype newType = createMPIStructType<T_First, T_Others...>();
            SAMS::typeHandle handle = gettypeRegistry().registerType<classType>(newType);
            return std::make_tuple(handle, newType);
        }

        /**
         * Create an MPI_Type_contiguous type for a specified base type and count
         * @param count The number of elements in the contiguous type
         * @param baseType The base MPI_Datatype
         */
        MPI_Datatype createMPIContiguousType([[maybe_unused]] int count, [[maybe_unused]] MPI_Datatype baseType)
        {
            #ifdef USE_MPI
            MPI_Datatype newType;            
            checkMPIError(MPI_Type_contiguous(
                count,
                baseType,
                &newType));
            std::string Name = buildMPIGenericName(newType);
            //If type already in cache, return existing type
            return cacheType(newType);
            #else
            return MPI_DATATYPE_NULL;
            #endif
        }

        /**
         * Create an MPI_Type_vector type for a specified base type, count, blocklength and stride
         * @param count The number of blocks in the vector type
         * @param blocklength The number of elements in each block
         * @param stride The stride between blocks
         * @param baseType The base MPI_Datatype
         */
        MPI_Datatype createMPIVectorType([[maybe_unused]] int count, [[maybe_unused]] int blocklength, [[maybe_unused]] int stride, [[maybe_unused]] MPI_Datatype baseType)
        {
            #ifdef USE_MPI
            MPI_Datatype newType;
            checkMPIError(MPI_Type_vector(
                count,
                blocklength,
                stride,
                baseType,
                &newType));
            return cacheType(newType);
            #else
            return MPI_DATATYPE_NULL;
            #endif
        }

        /**
         * Create an MPI_Type_hvector type for a specified base type, count, blocklength and stride
         * @param count The number of blocks in the hvector type
         * @param blocklength The number of elements in each block
         * @param stride The stride between blocks
         * @param baseType The base MPI_Datatype
         */
        MPI_Datatype createMPIHVectorType([[maybe_unused]] int count, [[maybe_unused]] int blocklength, [[maybe_unused]] MPI_Aint stride, [[maybe_unused]] MPI_Datatype baseType)
        {
            #ifdef USE_MPI
            MPI_Datatype newType;
            checkMPIError(MPI_Type_create_hvector(
                count,
                blocklength,
                stride,
                baseType,
                &newType));
            std::string Name = buildMPIGenericName(newType);
            //If type already in cache, return existing type
            return cacheType(newType);
            #else
            return MPI_DATATYPE_NULL;
            #endif
        }

        /**
         * Add a generic MPI_Datatype to the cache with an automatically generated name
         * @param mpiType The MPI_Datatype to add
         * @result The name assigned to the MPI_Datatype
         */
        std::string cacheMPIType([[maybe_unused]] MPI_Datatype &mpiType)
        {
            #ifdef USE_MPI
            mpiType = cacheType(mpiType);
            return buildMPIGenericName(mpiType);
            #else
            return "MPI not enabled! Seeing this is an error.";
            #endif
        }

        
        /**
         * Get a name for a given MPI_Datatype
         * @param mpiType The MPI_Datatype to get the name for
         * @return The name of the MPI_Datatype
         */
        std::string getMPITypeName([[maybe_unused]] MPI_Datatype mpiType)
        {
            #ifdef USE_MPI
            return buildMPIGenericName(mpiType);
            #else
            return "MPI not enabled! Seeing this is an error.";
            #endif
        }


        /**
         * Checkout the MPI_Datatype for a given type name, incrementing the reference count
         * @param typeName The name of the type to checkout
         * @return The MPI_Datatype
         */
        MPI_Datatype checkoutMPIType([[maybe_unused]] const std::string &typeName)
        {
            #ifdef USE_MPI
            return checkoutCore(typeName);
            #else
            return MPI_DATATYPE_NULL;
            #endif
        }


        /**
         * Checkin the MPI_Datatype, decrementing the reference count and deleting if necessary
         * @param mpiType The MPI_Datatype to checkin
         * @param forceDelete If true, delete the type even if the reference count is not zero
         */
        void checkinMPIType([[maybe_unused]] MPI_Datatype &mpiType, [[maybe_unused]] bool forceDelete = false)
        {
            #ifdef USE_MPI
            bool deleted = deleteMPIType(mpiType, forceDelete);
            if (deleted) mpiType = MPI_DATATYPE_NULL;
            #endif
        }

        /**
         * Create the MPI datatypes for halo exchange for the given variable
         * @param rank The rank of the variable
         * @param dims Array of dimensions for the variable
         * @param mpiSend Array to store the send MPI_Datatypes (size 2*rank)
         * @param mpiRecv Array to store the receive MPI_Datatypes (size 2*rank)
         * @param baseType The base MPI_Datatype of the variable
         */
        void assignVariableMPITypes([[maybe_unused]] int rank, [[maybe_unused]] dimension* dims, [[maybe_unused]] MPI_Datatype* mpiSend, [[maybe_unused]] MPI_Datatype* mpiRecv, [[maybe_unused]] MPI_Datatype baseType)
        {
            #ifdef USE_MPI
            SIGNED_INDEX_TYPE LB[MAX_RANK], UB[MAX_RANK];
            for (int axis = 0; axis < rank; axis++)
            {
                auto &axisRef = ar.getAxis(dims[axis].axisName);
                //For non-decomposed axes, set MPI types to NULL
                if (axisRef.MPIAxisIndex <0){
                    mpiSend[axis*2] = MPI_DATATYPE_NULL;
                    mpiSend[axis*2+1] = MPI_DATATYPE_NULL;
                    mpiRecv[axis*2] = MPI_DATATYPE_NULL;
                    mpiRecv[axis*2+1] = MPI_DATATYPE_NULL;
                    continue;
                }
                //Create types for the send and receive buffers for each dimension
                //Send on Lower edges
                for (int i = 0; i < rank; i++)
                {
                    if (i == axis) {
                        //In the direction being created, only send the last real cells
                        LB[i] = dims[i].getLocalDomainLB();
                        //Note! Sending lower bounds so need to send number matching upper ghosts
                        UB[i] = LB[i] + dims[i].upperGhosts-1;
                    } else {
                        //In other directions, send the whole local domain
                        LB[i] = dims[i].getLocalLB();
                        UB[i] = dims[i].getLocalUB();
                    }
                }
                SAMS::debugAll3 << "Creating lower send type on axis " << axis << std::endl;
                mpiSend[axis*2] = createArraySliceType(rank, dims, LB, UB, baseType);

                //Send on upper edges
                for (int i = 0; i < rank; i++)
                {
                    if  (i == axis) {
                        //In the direction being created send the last real cells
                        //Note we are sending at top of domain so count must be for LOWER ghosts
                        LB[i] = dims[i].getLocalDomainUB() - dims[i].lowerGhosts+1;
                        UB[i] = dims[i].getLocalDomainUB();
                    } else {
                        //In other directions, send the whole local domain
                        LB[i] = dims[i].getLocalLB();
                        UB[i] = dims[i].getLocalUB();
                    }
                }
                mpiSend[axis*2+1] = createArraySliceType(rank, dims, LB, UB, baseType);

                //Receive types
                //Receive on lower edges
                for (int i = 0; i < rank; i++)
                {
                    if (i == axis) {
                        LB[i] = dims[i].getLocalDomainLB() - dims[i].lowerGhosts;
                        UB[i] = dims[i].getLocalDomainLB()-1;
                    } else {
                        //In other directions, receive the whole local domain
                        LB[i] = dims[i].getLocalLB();
                        UB[i] = dims[i].getLocalUB();
                    }
                }
                mpiRecv[axis*2] = createArraySliceType(rank, dims, LB, UB, baseType);

                //Receive on upper edges
                for (int i = 0; i < rank; i++)
                {
                    if  (i == axis) {
                        //In the direction being created, only receive the upper ghost cells
                        //Receiving on the top of the domain so need to receive number matching upper ghosts
                        LB[i] = dims[i].getLocalDomainUB() +1;
                        UB[i] = LB[i] + dims[i].upperGhosts-1;
                    } else {
                        //In other directions, receive the whole local domain
                        LB[i] = dims[i].getLocalLB();
                        UB[i] = dims[i].getLocalUB();
                    }
                }
                mpiRecv[axis*2+1] = createArraySliceType(rank, dims, LB, UB, baseType);
            }
            #else
            //If not using MPI, set all types to MPI_DATATYPE_NULL
            for (int axis = 0; axis < rank; axis++)
            {
                mpiSend[axis*2] = MPI_DATATYPE_NULL;
                mpiSend[axis*2+1] = MPI_DATATYPE_NULL;
                mpiRecv[axis*2] = MPI_DATATYPE_NULL;
                mpiRecv[axis*2+1] = MPI_DATATYPE_NULL;
            }
            #endif
        }

        /**
         * Release all MPI datatypes associated with a variable
         * @param rank The rank of the variable
         * @param mpiSend Array of send MPI_Datatypes (size 2*rank)
         * @param mpiRecv Array of receive MPI_Datatypes (size 2*rank)
         */
        void releaseVariableMPITypes(int rank, MPI_Datatype* mpiSend, MPI_Datatype* mpiRecv)
        {
            for (int axis = 0; axis < rank; axis++)
            {
                if (mpiSend[axis*2] != MPI_DATATYPE_NULL) {
                    deleteMPIType(mpiSend[axis*2]);
                    mpiSend[axis*2] = MPI_DATATYPE_NULL;
                }
                if (mpiSend[axis*2+1] != MPI_DATATYPE_NULL) {
                    deleteMPIType(mpiSend[axis*2+1]);
                    mpiSend[axis*2+1] = MPI_DATATYPE_NULL;
                }
                if (mpiRecv[axis*2] != MPI_DATATYPE_NULL) {
                    deleteMPIType(mpiRecv[axis*2]);
                    mpiRecv[axis*2] = MPI_DATATYPE_NULL;
                }
                if (mpiRecv[axis*2+1] != MPI_DATATYPE_NULL) {
                    deleteMPIType(mpiRecv[axis*2+1]);
                    mpiRecv[axis*2+1] = MPI_DATATYPE_NULL;
                }
            }
        } 

        /**
         * Decompose all registered axes
         */
        void decomposeAllAxes()
        {
            const auto &areg = ar;
            for (const auto &pair : areg.getAxisMap())
            {
                SAMS::debug3 << "Decomposing axis: " << pair.first << std::endl;
                decomposeAxis(pair.first);
            }
        }

        void haloExchange([[maybe_unused]] void* data, [[maybe_unused]] int rank, [[maybe_unused]] MPI_Datatype* mpiSend, [[maybe_unused]] MPI_Datatype* mpiRecv, [[maybe_unused]] int axis, [[maybe_unused]] SAMS::domain::edges edgeType)
        {
#ifdef USE_MPI
            int sendIndex = (edgeType == SAMS::domain::edges::lower ? axis*2 : axis*2+1);
            int recvIndex = (edgeType == SAMS::domain::edges::lower ? axis*2+1 : axis*2);

            //Send to specified neighbor, receive from specified neighbor
            //Use precreated MPI datatypes for the variable slices
            checkMPIError(MPI_Sendrecv(static_cast<char*>(data), 1, mpiSend[sendIndex], neighbors[sendIndex], 0, static_cast<char*>(data), 1, mpiRecv[recvIndex], neighbors[recvIndex], 0, comm, MPI_STATUS_IGNORE));
            SAMS::debugAll3 << "Completed halo exchange on axis " << axis << std::endl;
#endif
        }

        void haloExchange([[maybe_unused]] void* data, [[maybe_unused]] int rank, [[maybe_unused]] MPI_Datatype* mpiSend, [[maybe_unused]] MPI_Datatype* mpiRecv, [[maybe_unused]] int axis)
        {
#ifdef USE_MPI
            haloExchange(data, rank, mpiSend, mpiRecv, axis, SAMS::domain::edges::lower);
            haloExchange(data, rank, mpiSend, mpiRecv, axis, SAMS::domain::edges::upper);
#endif
        }

        /**
         * 
         */
        void haloExchange([[maybe_unused]] void* data, [[maybe_unused]] int rank, [[maybe_unused]] MPI_Datatype* mpiSend, [[maybe_unused]] MPI_Datatype* mpiRecv)
        {
#ifdef USE_MPI
            for (int axis = 0; axis < rank; axis++)
            {
                haloExchange(data, rank, mpiSend, mpiRecv, axis);
            }
            SAMS::debugAll3 << "Completed halo exchange" << std::endl;
#endif
        }

        void abort(const std::string &message, [[maybe_unused]] bool localError = false)
        {
#ifdef USE_MPI
            if (localError) {
                std::cerr << "Error detected on rank " << rank << " : " << message << std::endl;
            } else {
                SAMS::cerr << "Error detected : " << message << std::endl;
            }
            MPI_Abort(comm, 1);
#else
            throw std::runtime_error("Abort called: " + message);
#endif
        }

        void finalize()
        {
            #ifdef USE_MPI
            if (comm != rootComm)
                checkMPIError(MPI_Comm_free(&comm));
            #endif
        }

        /**
         * Create an MPIManager with default communicator MPI_COMM_WORLD
         */
        MPIManager(axisRegistry &axisReg) : ar(axisReg)
        {
            defaultInit(MPI_COMM_WORLD);
        }

        /**
         * Create an MPIManager with a custom communicator
         * Outer code must ensure that the communicator is valid throughout the lifetime of the MPIManager
         * @param customComm The custom MPI communicator to use
         */
        MPIManager(axisRegistry &axisReg, MPI_Comm customComm) : ar(axisReg)
        {
            defaultInit(customComm);
        }

        /*Destructor to free MPI resources
        *Only have to free comm if it is not the rootComm
        *Don't have to free rootComm because we didn't create it
        *Don't have to free types because they are freed in the caches
        */
        ~MPIManager() {
            finalize();
        }
    };

} // namespace SAMS

#endif // SAMS_MPI_MANAGER_H
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

        enum class edges
        {
            LOWER=0,
            UPPER=1
        };

        namespace{

            //Singleton map of MPIManager instances
            template<int Dims>
            inline std::unordered_map<MPI_Comm, MPIManager<Dims>> &getInstanceMap(){
                static std::unordered_map<MPI_Comm, MPIManager<Dims>> instances;
                return instances;
            }

            //Recursive function to release all MPIManager instances for all ranks up to MAX_RANK
            template<int Dims=0>
            inline void releaseAllMPIManagers(){
                getInstanceMap<Dims>().clear();
                if constexpr (Dims<MAX_RANK){
                    releaseAllMPIManagers<Dims+1>();
                }
            }

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

        #ifdef USE_MPI
            //Create a map from MPI_Order_* to a string for error messages
            inline static std::map<int, std::string> mpiOrderMap = {
                {MPI_ORDER_C, "MPI_ORDER_C"},
                {MPI_ORDER_FORTRAN, "MPI_ORDER_FORTRAN"}
            };
        #endif


        inline void initialize(int *argc, char ***argv)
        {
            int rank=0;
            #ifdef USE_MPI
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

        inline void initialize(int &argc, char **&argv)
        {
            initialize(&argc, &argv);

        }

        inline void finalize(){
            releaseAllMPIManagers();
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
         * Type defining the information required to uniquely identify or construct an MPI subarray datatype
         */
        struct MPISubarrayInfo
        {
            MPI_Datatype basetype;
            int rank;
            int starts[MAX_RANK]={};
            int subsizes[MAX_RANK]={};
            int sizes[MAX_RANK]={};
            int layout = MPI_ORDER_C;
            std::string Name="";

            /**
             * Default constructor initializing all values to zero
             */
            MPISubarrayInfo(){
                for (int i = 0; i < MAX_RANK; i++){
                    starts[i] = 0;
                    subsizes[i] = 0;
                    sizes[i] = 0;
                }
            }

            void buildName(){
                #ifdef USE_MPI
                Name = "Subarray_type(" + gettypeRegistry().getTypeName(basetype) + ")" +
                       "_rank(" + std::to_string(rank) + ")" +
                       "_layout(" + SAMS::MPI::mpiOrderMap[layout] + ")_s(";
                for (int i = 0; i < rank; i++){
                    Name += std::to_string(starts[i]);
                    if (i < rank - 1){
                        Name += ",";
                    }
                }
                Name += ")_ss(";
                for (int i = 0; i < rank; i++){
                    Name += std::to_string(subsizes[i]);
                    if (i < rank - 1){
                        Name += ",";
                    }
                }
                Name += ")_sz(";
                for (int i = 0; i < rank; i++){
                    Name += std::to_string(sizes[i]);
                    if (i < rank - 1){
                        Name += ",";
                    }
                }
                Name += ")";
                #else
                Name = "MPI not enabled! Seeing this is an error.";
                #endif
            }

            /**
             * Constructor from arrays
             */
            template<int N>
            MPISubarrayInfo(MPI_Datatype base, const std::array<int, N>& startArr, const std::array<int, N>& subsizeArr, const std::array<int, N>& sizeArr, int order = MPI_ORDER_C)
                : basetype(base), rank(N), layout(order)
            {
                static_assert(N <= MAX_RANK, "Error: Rank exceeds MAX_RANK in MPISubarrayInfo constructor\n");
                for (int i = 0; i < N; i++){
                    starts[i] = startArr[i];
                    subsizes[i] = subsizeArr[i];
                    sizes[i] = sizeArr[i];
                }
                buildName();
            }

            /**
             * Construct from C-style arrays
             */
            MPISubarrayInfo(MPI_Datatype base, const int* startArr, const int* subsizeArr, const int* sizeArr, int nRank,  int order = MPI_ORDER_C)
                : basetype(base), rank(nRank), layout(order)
            {
                if (nRank > MAX_RANK){
                    throw std::runtime_error("Error: Rank exceeds MAX_RANK in MPISubarrayInfo constructor\n");
                }
                for (int i = 0; i < nRank; i++){
                    starts[i] = startArr[i];
                    subsizes[i] = subsizeArr[i];
                    sizes[i] = sizeArr[i];
                }
                buildName();
            }

            /**
             * Comparator for use in map keys
             * @param other The other MPISubarrayInfo to compare against
             * @return true if this is less than other
             */
            bool operator<(const MPISubarrayInfo& other) const
            {
                if (basetype != other.basetype)
                {
                    //This are potentially opaque types, so we compare the raw bytes
                    uint64_t t1, t2;
                    std::size_t n = (sizeof(t1) < sizeof(basetype)) ? sizeof(t1) : sizeof(basetype);
                    std::memcpy(&t1, &basetype, n);
                    std::memcpy(&t2, &other.basetype, n);
                    return t1 < t2;
                }
                if (rank != other.rank)
                    return rank < other.rank;
                if (layout != other.layout)
                    return layout < other.layout;
                for (int i = 0; i < rank; i++)
                {
                    if (starts[i] != other.starts[i])
                        return starts[i] < other.starts[i];
                    if (subsizes[i] != other.subsizes[i])
                        return subsizes[i] < other.subsizes[i];
                    if (sizes[i] != other.sizes[i])
                        return sizes[i] < other.sizes[i];
                }
                return false; //They are equal
            }
        };


        /**
         * Type to hold an MPI_Datatype along with a reference count for shared usage
         */
        struct MPITypeHolder
        {
            COUNT_TYPE refCount=1;
            MPI_Datatype mpiType = MPI_DATATYPE_NULL;
            std::string Name="";

            //Delete default constructor and copy operations to avoid accidental copies
            MPITypeHolder() = delete;
            MPITypeHolder(const MPITypeHolder &) = delete;
            MPITypeHolder &operator=(const MPITypeHolder &) = delete;

            //Move constructor to allow "normal" operation in containers
            MPITypeHolder(MPITypeHolder &&other) noexcept
                : refCount(other.refCount), mpiType(other.mpiType)
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

            MPITypeHolder(MPI_Datatype type, std::string name)
                : mpiType(type), Name(name) {}

            /**
             * Destructor to free the MPI_Datatype
             */
            ~MPITypeHolder(){
                releaseType();
            }

            /*
            * Release the MPI_Datatype
            */
            void releaseType(){
            #ifdef USE_MPI
                if (mpiType != MPI_DATATYPE_NULL){
                    SAMS::debugAll3 << "Releasing MPI type " << Name << std::endl;
                    MPI_Type_free(&mpiType);
                }
            #endif
            }

            /**
             * Increment the reference count
             */
            COUNT_TYPE operator++(){
                return ++refCount;
            }

            /**
             * Post increment the reference count
             */
            COUNT_TYPE operator++(int){
                COUNT_TYPE oldCount = refCount;
                refCount++;
                return oldCount;
            }

            /**
             * Decrement the reference count and release the type if it reaches zero
             */
            COUNT_TYPE operator--(){
                refCount--;
                if (refCount == 0){
                    releaseType();
                }
                return refCount;
            }

            /**
             * Decrement refrence count for post decrement
             */
            COUNT_TYPE operator--(int){
                COUNT_TYPE oldCount = refCount;
                refCount--;
                if (refCount == 0){
                    releaseType();
                }
                return oldCount;
            }

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
         * Because there are no requirements on the type of MPI_Datatype, we have to use another struct to store reverse mapping information
         */
        struct MPIReverseInfo
        {
            MPI_Datatype mpiType = MPI_DATATYPE_NULL;

            //Comparator used for map
            bool operator<(const MPIReverseInfo& other) const
            {
                //Again could be an opaque type, so we compare the raw bytes
                uint64_t t1, t2;
                std::size_t n = (sizeof(t1) < sizeof(mpiType)) ? sizeof(t1) : sizeof(mpiType);
                std::memcpy(&t1, &mpiType, n);
                std::memcpy(&t2, &other.mpiType, n);
                return t1 < t2;
            }
        };

        /**
         * Cache of created MPI_Datatype objects for subarrays
         */
        std::map<MPISubarrayInfo, MPITypeHolder> mpiSubarrayTypeCache;

        /**
         * Reverse cache to find MPISubarrayInfo from MPI_Datatype
         * Used when freeing MPI_Datatype objects
         * Map used rather than unordered_map to avoid
         * having to deal with possible destruction/copying
         * of the MPI_Datatype keys on adding new keys
         * Yes. I know that I've used deleted copy constructors
         * elsewhere to avoid this, but this is simpler for now. 
        */
        std::map<MPIReverseInfo, MPISubarrayInfo> mpiReverseTypeCache;


        /** Function to create an MPI type from an MPISubarrayInfo type 
         * @param info The MPISubarrayInfo structure defining the MPI subarray type
         * @return The created MPI_Datatype
        */
        MPI_Datatype createMPISubarrayType(const MPISubarrayInfo &info)
        {
            #ifdef USE_MPI
            //If type already in cache, return existing type
            auto it = mpiSubarrayTypeCache.find(info);
            if (it != mpiSubarrayTypeCache.end())
            {
                SAMS::debugAll3<< "Found existing MPI subarray type: " << info.Name << std::endl;
                return it->second.mpiType;
            }
            //Otherwise create new type
            MPI_Datatype newType;
            checkMPIError(MPI_Type_create_subarray(
                info.rank,
                info.sizes,
                info.subsizes,
                info.starts,
                info.layout, 
                info.basetype,
                &newType));
            checkMPIError(MPI_Type_commit(&newType));

            //Add to forward cache
            mpiSubarrayTypeCache.emplace(info, newType);
            auto& typeHolder = mpiSubarrayTypeCache.at(info);
            typeHolder.Name = info.Name;
            SAMS::debugAll3 << "Created new MPI subarray type: " << info.Name << std::endl;
            //Also add to reverse cache
            MPIReverseInfo revInfo;
            revInfo.mpiType = newType;
            mpiReverseTypeCache[revInfo] = info;
            return newType;
            #else
            return MPI_DATATYPE_NULL;
            #endif
        }

        /**
         * Delete a specified MPI_Datatype from the caches and free it
         */
        void deleteMPISubarrayType(MPI_Datatype mpiType, bool forceDelete = false)
        {
            MPIReverseInfo revInfo;
            revInfo.mpiType = mpiType;
            auto itR = mpiReverseTypeCache.find(revInfo);
            if (itR != mpiReverseTypeCache.end())
            {
                MPISubarrayInfo& info = itR->second;
                auto itF = mpiSubarrayTypeCache.find(info);
                MPITypeHolder& typeHolder = itF->second;

                //The -- operator has the normal pre and post decrement behaviour
                //So need predecrement here to get the correct value
                COUNT_TYPE newRefCount = --typeHolder;
                if ((newRefCount == 0) || forceDelete){
                    SAMS::debugAll3 << "Releasing MPI subarray type: " << info.Name << std::endl;
                    //Delete it from the forward cache
                    mpiReverseTypeCache.erase(itR);
                    mpiSubarrayTypeCache.erase(itF);
                } else {
                    SAMS::debugAll3 << "Decremented reference count for MPI subarray type: " << info.Name << " to " << newRefCount << std::endl;
                }
                //MPI_Type_free is called in the destructor of MPIReverseInfo
            }
        }

    public:

        /**
         * Set up the decomposition of the MPI Cartesian communicator
         * @param decomposition Array of size Dims specifying the number of processors in each dimension
         * @param isPeriodic Array of size Dims specifying whether each dimension is periodic (1
         */
        template<std::size_t N = Dims>
        void setDecomposition(const std::array<int, Dims> &decomposition, const std::array<bool, N> &isPeriodic)
        {
            static_assert(N <= MAX_RANK, "Error: isPeriodic array size exceeds MAX_RANK");
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
            for (int i = 0; i < N; i++)
            {
                periods[i] = isPeriodic[i]? 1 : 0;
            }
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
         */
        MPI_Comm getComm() const
        {
            return comm;
        }

        /**
         * Get the coordinates of the current processor in the Cartesian communicator
         * @return Array of size Dims containing the coordinates
         */
        std::array<int, Dims * 2> getNeighbors() const
        {
            return neighbors;
        }

        /**
         * Check if a given axis is at the edge of the decomposition including the effect of periodicity
         */
        bool isEdge(int axis, bool isLower) const
        {
            int neighborIndex = axis * 2 + (isLower ? 0 : 1);
            return neighbors[neighborIndex] == MPI_PROC_NULL;
        }

        /**
         * This detects if you are at the edge of the decomposition WITHOUT considering periodicity
         */
        bool isEdgeNP(int axis, bool isLower) const
        {
            if (isLower){
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
            int axis = getaxisRegistry().getMPIAxis(axisName);
            size_t globalElements = getaxisRegistry().getLocalDomainElements(axisName, staggerType::CENTRED);
            size_t localElements = 0;
            if (axis >= 0)
            {
                // Decompose the axis, distributing any remainder to the first few ranks
                localElements = globalElements / dims[axis];
                size_t remainder = globalElements % dims[axis];
                if (coords[axis] < remainder)
                {
                    localElements++;
                }
            } else {
                // Axis is not decomposed, so all elements are local
                localElements = globalElements;
            }
            getaxisRegistry().setLocalDomainElements(axisName, localElements, staggerType::CENTRED);

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
            size_t localUB = localLB + localElements;
            auto &axisRef = getaxisRegistry().getAxis(axisName);
            axisRef.dim.setGlobalBounds(localLB, localUB);
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
        MPI_Datatype variableSliceType(int rank, SIGNED_INDEX_TYPE *domainLB, SIGNED_INDEX_TYPE *domainUB, SIGNED_INDEX_TYPE *LB, SIGNED_INDEX_TYPE *UB, MPI_Datatype baseType){
            #ifdef USE_MPI
            int starts[MAX_RANK], sizes[MAX_RANK], subsizes[MAX_RANK];
            for (int i = 0; i < rank; i++)
            {
                sizes[i] = domainUB[i] - domainLB[i]+1;
                subsizes[i] = UB[i] - LB[i] + 1;
                starts[i] = LB[i] - domainLB[i];
            }

            MPI_Datatype sliceType;

            //Add the type to the cache or get existing type
            sliceType = createMPISubarrayType(MPISubarrayInfo(baseType, starts, subsizes, sizes, rank));

            return sliceType;
            #else
            return MPI_DATATYPE_NULL;
            #endif
        }

        /**
         * Build an MPI datatype for a slice of a variable defined by lower and upper bounds in each dimension
         */
        MPI_Datatype variableSliceType(int rank, dimension* dims, SIGNED_INDEX_TYPE *LB, SIGNED_INDEX_TYPE *UB, MPI_Datatype baseType){

            SIGNED_INDEX_TYPE domainLB[MAX_RANK], domainUB[MAX_RANK];
            for(int i=0; i<rank; i++){
            domainLB[i] = dims[i].getLocalLB();
            domainUB[i] = dims[i].getLocalUB();
            }
            return variableSliceType(rank, domainLB, domainUB, LB, UB, baseType);

        }

        /**
         * Create the MPI datatypes for halo exchange for the given variable
         * @param rank The rank of the variable
         * @param dims Array of dimensions for the variable
         * @param mpiSend Array to store the send MPI_Datatypes (size 2*rank)
         * @param mpiRecv Array to store the receive MPI_Datatypes (size 2*rank)
         * @param baseType The base MPI_Datatype of the variable
         */
        void assignVariableMPITypes(int rank, dimension* dims, MPI_Datatype* mpiSend, MPI_Datatype* mpiRecv, MPI_Datatype baseType)
        {
            #ifdef USE_MPI
            SIGNED_INDEX_TYPE LB[MAX_RANK], UB[MAX_RANK];
            for (int axis = 0; axis < rank; axis++)
            {
                auto &axisRef = getaxisRegistry().getAxis(dims[axis].axisName);
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
                        LB[i] = dims[i].getLocalDomainLB();
                        UB[i] = dims[i].getLocalDomainUB();
                    }
                }
                SAMS::debugAll3 << "Creating lower send type on axis " << axis << std::endl;
                mpiSend[axis*2] = variableSliceType(rank, dims, LB, UB, baseType);

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
                        LB[i] = dims[i].getLocalDomainLB();
                        UB[i] = dims[i].getLocalDomainUB();
                    }
                }
                mpiSend[axis*2+1] = variableSliceType(rank, dims, LB, UB, baseType);

                //Receive types
                //Receive on lower edges
                for (int i = 0; i < rank; i++)
                {
                    if (i == axis) {
                        LB[i] = dims[i].getLocalDomainLB() - dims[i].lowerGhosts;
                        UB[i] = dims[i].getLocalDomainLB()-1;
                    } else {
                        //In other directions, receive the whole local domain
                        LB[i] = dims[i].getLocalDomainLB();
                        UB[i] = dims[i].getLocalDomainUB();
                    }
                }
                mpiRecv[axis*2] = variableSliceType(rank, dims, LB, UB, baseType);

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
                        LB[i] = dims[i].getLocalDomainLB();
                        UB[i] = dims[i].getLocalDomainUB();
                    }
                }
                mpiRecv[axis*2+1] = variableSliceType(rank, dims, LB, UB, baseType);
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
                    deleteMPISubarrayType(mpiSend[axis*2]);
                    mpiSend[axis*2] = MPI_DATATYPE_NULL;
                }
                if (mpiSend[axis*2+1] != MPI_DATATYPE_NULL) {
                    deleteMPISubarrayType(mpiSend[axis*2+1]);
                    mpiSend[axis*2+1] = MPI_DATATYPE_NULL;
                }
                if (mpiRecv[axis*2] != MPI_DATATYPE_NULL) {
                    deleteMPISubarrayType(mpiRecv[axis*2]);
                    mpiRecv[axis*2] = MPI_DATATYPE_NULL;
                }
                if (mpiRecv[axis*2+1] != MPI_DATATYPE_NULL) {
                    deleteMPISubarrayType(mpiRecv[axis*2+1]);
                    mpiRecv[axis*2+1] = MPI_DATATYPE_NULL;
                }
            }
        } 

        /**
         * Decompose all registered axes
         */
        void decomposeAllAxes()
        {
            const auto &areg = getaxisRegistry();
            for (const auto &pair : areg.getAxisMap())
            {
                SAMS::debug3 << "Decomposing axis: " << pair.first << std::endl;
                decomposeAxis(pair.first);
            }
        }

        void haloExchange(void* data, int rank, MPI_Datatype* mpiSend, MPI_Datatype* mpiRecv)
        {
#ifdef USE_MPI
            void* basePtr = data;
            for (int axis = 0; axis < rank; axis++)
            {
                if (mpiSend[axis*2] == MPI_DATATYPE_NULL){
                    //Axis is not decomposed, skip
                    continue;
                }
                if (neighbors[axis*2] != MPI_PROC_NULL){
                    SAMS::debugAll3 << "SendRecv to lower neighbor " << neighbors[axis*2] << " on axis " << axis << std::endl;
                }
                if (neighbors[axis*2+1] != MPI_PROC_NULL){
                    SAMS::debugAll3 << "SendRecv to upper neighbor " << neighbors[axis*2+1] << " on axis " << axis << std::endl;
                }
                //Send to lower neighbor, receive from upper neighbor
                //Use precreated MPI datatypes for the variable slices
                checkMPIError(MPI_Sendrecv(static_cast<char*>(basePtr), 1, mpiSend[axis*2], neighbors[axis*2], 0, static_cast<char*>(basePtr), 1, mpiRecv[axis*2+1], neighbors[axis*2+1], 0, comm, MPI_STATUS_IGNORE));
                //Send to upper neighbor, receive from lower neighbor
                checkMPIError(MPI_Sendrecv(static_cast<char*>(basePtr), 1, mpiSend[axis*2+1], neighbors[axis*2+1], 0, static_cast<char*>(basePtr), 1, mpiRecv[axis*2], neighbors[axis*2], 0, comm, MPI_STATUS_IGNORE));
                SAMS::debugAll3 << "Done axis " << axis << std::endl;
            }
            SAMS::debugAll3 << "Completed halo exchange" << std::endl;
#endif
        }

        /**
         * Create an MPIManager with default communicator MPI_COMM_WORLD
         */
        MPIManager()
        {
            defaultInit(MPI_COMM_WORLD);
        }

        /**
         * Create an MPIManager with a custom communicator
         * Outer code must ensure that the communicator is valid throughout the lifetime of the MPIManager
         * @param customComm The custom MPI communicator to use
         */
        MPIManager(MPI_Comm customComm)
        {
            defaultInit(customComm);
        }

        /*Destructor to free MPI resources
        *Only have to free comm if it is not the rootComm
        *Don't have to free rootComm because we didn't create it
        *Don't have to free types because they are freed in the caches
        */
        ~MPIManager() {
            #ifdef USE_MPI
            if (comm != rootComm)
                checkMPIError(MPI_Comm_free(&comm));
            #endif
        }
    };


    template<int Dims=MPI_DECOMPOSITION_RANK>
    MPIManager<Dims>& getMPIManager(MPI_Comm customComm = MPI_COMM_WORLD)
    {
        auto &instances = MPI::getInstanceMap<Dims>();
        auto it = instances.find(customComm);
        if (it == instances.end()) {
            instances.emplace(customComm, customComm);
        }
        return instances[customComm];
    }


} // namespace SAMS

#endif // SAMS_MPI_MANAGER_H
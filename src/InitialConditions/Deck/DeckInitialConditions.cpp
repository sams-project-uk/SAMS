
#include "DeckInitialConditions.h"
#include "deckVariables.h"
#include "deckHelpers/deckInitialConditionsHelper.h"
#include "deckHelpers/expressionBC.h"
#include "uniformDomain.h"
#include "io/writers.h"
#include "io/readers.h"
#ifdef USE_DECK_PARSER
#include "EISpp/eis.h"
#endif
#include <array>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace LARE
{

    namespace
    {
        using T = SAMS::T_dataType;

        constexpr std::array<const char*, 9> fieldNames = {"rho", "energy_electron", "energy_ion", "vx", "vy", "vz", "bx", "by", "bz"};

        using SAMS::conditionType;
        using SAMS::directionNames;
        using SAMS::edgeText;

#ifdef USE_DECK_PARSER
        using SAMS::expressionBC;
#endif

        /**
         * The deck text of the condition of LARE/dm from that of rho. The type is kept, values become 0 and expression becomes clamp(0)
         */
        std::string dmConditionFromRho(const std::string &text)
        {
            const std::string type = conditionType(text);
            if (type == "expression") return "clamp(0)";
            if ((type == "clamp" || type == "mirror") && text.find('(') != std::string::npos) return type + "(0)";
            return text;
        }

        /**
         * Attach the condition of deck text such as mirror(0.0) to one edge. mirror without a value is a floating mirror
         */
        void attachEdgeCondition(SAMS::harness &harness, SAMS::variableDef &varDef, int dimension, SAMS::domain::edges edge, const std::string &text, const std::string &path,
            [[maybe_unused]] SAMS::timeState &tData, [[maybe_unused]] SAMS::deckHostData<3> &hostData)
        {
            const std::string type = conditionType(text);
            const size_t open = text.find('(');
            const bool needsValue = type == "clamp" || type == "expression";
            if (needsValue && open == std::string::npos) harness.abort("Invalid " + path + ": " + type + " needs a value", false);
            const bool hasValue = needsValue || (type == "mirror" && open != std::string::npos);
            const std::string argument = hasValue ? text.substr(open + 1, text.rfind(')') - open - 1) : "";
            T value = 0.0;
            if ((type == "clamp" || type == "mirror") && hasValue) {
                try {
                    value = SAMS::KVS::Element(argument).asReal();
                } catch (const std::exception &e) {
                    harness.abort("Invalid " + path + ": " + e.what(), false);
                }
            }
            if (type == "expression") {
#ifdef USE_DECK_PARSER
                eis::parser<T> &parser = *hostData.parser;
                eis::stack<T> stack;
                eis::error errcode = eis::err_none;
                parser.tokenize(argument, stack, errcode);
                if (errcode != eis::err_none) {
                    harness.abort("Invalid " + path + ": " + text + "\n" + SAMS::parserErrors(parser), false);
                }
                hostData.addPositions(harness, varDef);
                varDef.addBoundaryCondition(dimension, edge, expressionBC<3>(varDef, harness, tData, hostData, stack));
#else
                harness.abort("Invalid " + path + ": " + text + "\nExpressions need the deck parser", false);
#endif
            }
            else if (type == "periodic") varDef.addBoundaryCondition(dimension, edge, SAMS::simplePeriodicBC<T, 3>(varDef));
            else if (type == "symmetric") varDef.addBoundaryCondition(dimension, edge, SAMS::simpleSymmetricBC<T, 3>(varDef));
            else if (type == "zero_gradient") varDef.addBoundaryCondition(dimension, edge, SAMS::simpleZeroGradientBC<T, 3>(varDef));
            else if (type == "clamp") varDef.addBoundaryCondition(dimension, edge, SAMS::simpleClamp<T, 3>(varDef, value));
            else if (type == "mirror" && hasValue) varDef.addBoundaryCondition(dimension, edge, SAMS::fixedMirrorBC<T, 3>(varDef, value));
            else if (type == "mirror") varDef.addBoundaryCondition(dimension, edge, SAMS::floatingMirrorBC<T, 3>(varDef));
            else harness.abort("Invalid " + path + ": " + text, false);
        }

        /**
         * Attach the conditions of a block of the boundaries block to every edge of a variable
         * @param block Path of the block below boundaries
         */
        void attachBoundaryConditions(const std::string &varName, const std::string &block, SAMS::harness &harness, SAMS::KVS &kvs,
            [[maybe_unused]] SAMS::timeState &tData, [[maybe_unused]] SAMS::deckHostData<3> &hostData)
        {
            SAMS::variableDef &varDef = harness.variableRegistry.getVariable(varName);
            for (int d = 0; d < 3; ++d){
                for (const auto &[suffix, edge] : {std::pair<const char*, SAMS::domain::edges>{"_min", SAMS::domain::edges::lower}, std::pair<const char*, SAMS::domain::edges>{"_max", SAMS::domain::edges::upper}}){
                    std::string path;
                    const std::string text = edgeText(kvs, "/boundaries", block, d, suffix, path);
                    if (text.empty()) harness.abort("No boundary condition for " + varName + " at " + directionNames[d] + suffix, false);
                    attachEdgeCondition(harness, varDef, d, edge, text, path, tData, hostData);
                }
            }
        }
    }

    /**
     * Default deck keys, and LARE3D's parameters as parser variables
     * @param data LARE3D simulation data
     * @param kvs The key value store
     * @param hostData The host data of the deck, which holds the parser
     */
    template<typename T_EOS>
    void LAREDeckInitialConditions<T_EOS>::defaultDeck([[maybe_unused]] typename LARE::LARE3D<T_EOS>::simulationData &data, SAMS::KVS &kvs, [[maybe_unused]] SAMS::deckHostData<3> &hostData)
    {
#ifdef USE_DECK_PARSER
        eis::parser<SAMS::T_dataType> &parser = *hostData.parser;
        //The parameters are read when an expression is evaluated, so they are always the current values
        eis::error errcode = eis::err_none;
        parser.add_pointer_variable("gas_gamma", &data.gas_gamma, errcode);
        parser.add_pointer_variable("mf", &data.mf, errcode);
        parser.add_pointer_variable("t_end", &data.t_end, errcode);
        if (errcode != eis::err_none) parser.print_errors();
#endif

        // Periodic in each direction in which rho is periodic
        SAMS::KVS::Array periodic;
        for (int d = 0; d < 3; ++d){
            std::string path;
            periodic.push_back(conditionType(edgeText(kvs, "/boundaries", "rho", d, "_min", path)) == "periodic");
        }
        kvs.setDefaultPath("/domain/MPI_periodic", periodic);

        // The number of cells and the limits of the domain in x have no defaults
        // y and z default to 2 cells, starting at x_min, with cells that are the same size as in x
        const SAMS::T_dataType x_min = kvs.getReal(SAMS::deckKeys::xMin);
        const SAMS::T_dataType dx = (kvs.getReal(SAMS::deckKeys::xMax) - x_min) / static_cast<SAMS::T_dataType>(kvs.getInt(SAMS::deckKeys::nx));
        kvs.setDefaultPath(SAMS::deckKeys::ny, 2);
        kvs.setDefaultPath(SAMS::deckKeys::nz, 2);
        kvs.setDefaultPath(SAMS::deckKeys::yMin, x_min);
        kvs.setDefaultPath(SAMS::deckKeys::yMax, kvs.getReal(SAMS::deckKeys::yMin) + dx * static_cast<SAMS::T_dataType>(kvs.getInt(SAMS::deckKeys::ny)));
        kvs.setDefaultPath(SAMS::deckKeys::zMin, x_min);
        kvs.setDefaultPath(SAMS::deckKeys::zMax, kvs.getReal(SAMS::deckKeys::zMin) + dx * static_cast<SAMS::T_dataType>(kvs.getInt(SAMS::deckKeys::nz)));
        kvs.setDefaultPath(SAMS::deckKeys::description, "Problem read from a deck. No description was given.");
    }

    /**
     * Read the end time
     * @param data LARE3D simulation data
     * @param kvs The key value store
     */
    template<typename T_EOS>
    void LAREDeckInitialConditions<T_EOS>::controlVariables(typename LARE::LARE3D<T_EOS>::simulationData &data, SAMS::KVS &kvs)
    {
        data.t_end = kvs.getReal(SAMS::deckKeys::endTime);
    }

    /**
     * Set up the simulation domain
     * @param harness SAMS harness
     * @param kvs The key value store
     */
    template<typename T_EOS>
    void LAREDeckInitialConditions<T_EOS>::setDomain(SAMS::harness &harness, SAMS::KVS &kvs)
    {
        SAMS::setUniformDomain(harness, kvs);
    }

    /**
     * Set boundary conditions for the simulation from the boundaries block
     * @param harness SAMS harness
     * @param hostData The host parameters of deck expressions
     * @param kvs The key value store
     * @param tData SAMS time state data
     */
    template<typename T_EOS>
    void LAREDeckInitialConditions<T_EOS>::setBoundaryConditions(SAMS::harness &harness, SAMS::deckHostData<3> &hostData, SAMS::KVS &kvs, SAMS::timeState &tData)
    {
        for (const char *name : fieldNames) hostData.addPlanes(harness, name);
        for (const char *name : fieldNames){
            attachBoundaryConditions(name, name, harness, kvs, tData, hostData);
        }
        attachBoundaryConditions("LARE/vx1", "vx", harness, kvs, tData, hostData);
        attachBoundaryConditions("LARE/vy1", "vy", harness, kvs, tData, hostData);
        attachBoundaryConditions("LARE/vz1", "vz", harness, kvs, tData, hostData);
        if (!kvs.getPath("/boundaries/LARE/dm", SAMS::KVS::Element()).isType(SAMS::KVS::Type::None)) {
            attachBoundaryConditions("LARE/dm", "LARE/dm", harness, kvs, tData, hostData);
            return;
        }
        //Without its own block, dm takes the types of the conditions of rho with no values
        SAMS::variableDef &varDef = harness.variableRegistry.getVariable("LARE/dm");
        for (int d = 0; d < 3; ++d){
            for (const auto &[suffix, edge] : {std::pair<const char*, SAMS::domain::edges>{"_min", SAMS::domain::edges::lower}, std::pair<const char*, SAMS::domain::edges>{"_max", SAMS::domain::edges::upper}}){
                std::string path;
                const std::string text = edgeText(kvs, "/boundaries", "rho", d, suffix, path);
                if (text.empty()) harness.abort("No boundary condition for LARE/dm at " + std::string(directionNames[d]) + suffix, false);
                attachEdgeCondition(harness, varDef, d, edge, dmConditionFromRho(text), path, tData, hostData);
            }
        }
    }

    /**
     * Set the variables from the expressions in the initial_conditions block of the deck
     * @param harnessRef SAMS harness
     * @param data LARE3D simulation data
     * @param kvs The key value store
     * @param files The locations of the files of the run
     * @param hostData The host parameters of deck expressions, which hold the parser
     */
    template<typename T_EOS>
    void LAREDeckInitialConditions<T_EOS>::initialConditions(SAMS::harness &harnessRef, [[maybe_unused]] typename LARE::LARE3D<T_EOS>::simulationData &data, SAMS::KVS &kvs, SAMS::fileInformation &files, SAMS::deckHostData<3> &hostData)
    {
#ifdef USE_DECK_PARSER
        eis::parser<SAMS::T_dataType> &parser = *hostData.parser;
        eis::error errcode = eis::err_none;
        addDeckVariables(parser, errcode);
        if (errcode != eis::err_none) {
            harnessRef.abort("Could not set up the parser for the initial conditions\n" + SAMS::parserErrors(parser), false);
        }
#endif

        const std::array<SAMS::deckVariableName, 9> variables = {{
            {"rho", "rho"}, {"energy_electron", "energy_electron"}, {"energy_ion", "energy_ion"},
            {"vx", "vx"}, {"vy", "vy"}, {"vz", "vz"},
            {"bx", "bx"}, {"by", "by"}, {"bz", "bz"}}};
        SAMS::setInitialConditions(harnessRef, hostData, kvs, files, variables);
    }

    #define EOS_DEF(value) template class LAREDeckInitialConditions<value>;
    EOS_DENSITY_ENERGY
    #undef EOS_DEF

}

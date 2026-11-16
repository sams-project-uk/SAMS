
#include "HLLEDeckInitialConditions.h"
#include "HLLEDeckVariables.h"
#include "deckHelpers/deckInitialConditionsHelper.h"
#include "deckHelpers/expressionBC.h"
#include "uniformDomain.h"
#include "builtInBoundaryConditions.h"
#ifdef USE_DECK_PARSER
#include "EISpp/eis.h"
#endif
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace conservedVariable
{

    namespace
    {
        using T = SAMS::T_dataType;

        constexpr std::array<const char*, 8> fieldNames = {"rho", "mx", "my", "mz", "energy", "bx", "by", "bz"};

        using SAMS::conditionType;
        using SAMS::directionNames;
        using SAMS::edgeText;

#ifdef USE_DECK_PARSER
        using SAMS::expressionBC;
#endif

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

    void deckInitialConditions::defaultDeck([[maybe_unused]] conservedSolver::simulationData &data, SAMS::KVS &kvs, [[maybe_unused]] SAMS::deckHostData<3> &hostData)
    {
#ifdef USE_DECK_PARSER
        eis::parser<SAMS::T_dataType> &parser = *hostData.parser;
        //The parameters are read when an expression is evaluated, so they are always the current values
        eis::error errcode = eis::err_none;
        parser.add_pointer_variable("gas_gamma", &data.gas_gamma, errcode);
        parser.add_pointer_variable("t_end", &t_end, errcode);
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
    }

    void deckInitialConditions::controlVariables(SAMS::KVS &kvs)
    {
        t_end = kvs.getReal(SAMS::deckKeys::endTime);
    }

    void deckInitialConditions::setDomain(SAMS::harness &harness, SAMS::KVS &kvs)
    {
        SAMS::setUniformDomain(harness, kvs);
    }

    void deckInitialConditions::setBoundaryConditions(SAMS::harness &harness, SAMS::deckHostData<3> &hostData, SAMS::KVS &kvs, SAMS::timeState &tData)
    {
        for (const char *name : fieldNames) hostData.addPlanes(harness, name);
        for (const auto &[varName, block] : {std::pair<const char*, const char*>{"rho", "rho"}, {"mx", "px"}, {"my", "py"}, {"mz", "pz"},
                {"energy", "energy"}, {"bx", "bx"}, {"by", "by"}, {"bz", "bz"}}){
            attachBoundaryConditions(varName, block, harness, kvs, tData, hostData);
        }
    }

    void deckInitialConditions::initialConditions(SAMS::harness &harnessRef, SAMS::KVS &kvs, SAMS::fileInformation &files, SAMS::deckHostData<3> &hostData)
    {
#ifdef USE_DECK_PARSER
        eis::parser<SAMS::T_dataType> &parser = *hostData.parser;
        eis::error errcode = eis::err_none;
        addDeckVariables(parser, errcode);
        if (errcode != eis::err_none) {
            harnessRef.abort("Could not set up the parser for the initial conditions\n" + SAMS::parserErrors(parser), false);
        }
#endif

        const std::array<SAMS::deckVariableName, 8> variables = {{
            {"rho", "rho"}, {"px", "mx"}, {"py", "my"}, {"pz", "mz"},
            {"energy", "energy"}, {"bx", "bx"}, {"by", "by"}, {"bz", "bz"}}};
        SAMS::setInitialConditions(harnessRef, hostData, kvs, files, variables);
    }

}

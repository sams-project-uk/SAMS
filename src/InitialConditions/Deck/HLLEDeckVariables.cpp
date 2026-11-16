
#include "HLLEDeckVariables.h"
#include "runner.h"
#ifdef USE_DECK_PARSER

namespace conservedVariable
{

    namespace
    {
        using evaluationPoint = SAMS::deckEvaluationPoint<3>;

        /**
         * The value of a variable at the point
         * @param host_params The host parameters of the parser
         * @param name The name of the variable in the registry
         */
        SAMS::T_dataType variableValue(const void* host_params, std::string_view name)
        {
            const evaluationPoint &point = *static_cast<const evaluationPoint*>(host_params);
            return std::apply(point.hostData.variable(name), point.index);
        }

        /**
         * The value of a variable on its last domain plane at the boundary of a boundary expression. The index normal to the boundary is that of the plane
         * @param host_params The host parameters of the parser
         * @param name The name of the variable in the registry
         */
        SAMS::T_dataType boundaryValue(const void* host_params, std::string_view name)
        {
            const evaluationPoint &point = *static_cast<const evaluationPoint*>(host_params);
            const portableWrapper::hostArray<SAMS::T_dataType, 3> &plane = point.hostData.plane(name, point.boundaryDimension, point.boundaryEdge);
            auto index = point.index;
            portableWrapper::getTupleElement(index, point.boundaryDimension) = plane.getLowerBound(point.boundaryDimension);
            return std::apply(plane, index);
        }

        SAMS::T_dataType getRho(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return variableValue(host_params, "rho");
        }

        SAMS::T_dataType getPx(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return variableValue(host_params, "mx");
        }

        SAMS::T_dataType getPy(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return variableValue(host_params, "my");
        }

        SAMS::T_dataType getPz(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return variableValue(host_params, "mz");
        }

        SAMS::T_dataType getEnergy(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return variableValue(host_params, "energy");
        }

        SAMS::T_dataType getBx(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return variableValue(host_params, "bx");
        }

        SAMS::T_dataType getBy(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return variableValue(host_params, "by");
        }

        SAMS::T_dataType getBz(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return variableValue(host_params, "bz");
        }

        //Getters for the variables on their last domain plane at the boundary of a boundary expression
        SAMS::T_dataType getRhoBoundary(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return boundaryValue(host_params, "rho");
        }

        SAMS::T_dataType getPxBoundary(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return boundaryValue(host_params, "mx");
        }

        SAMS::T_dataType getPyBoundary(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return boundaryValue(host_params, "my");
        }

        SAMS::T_dataType getPzBoundary(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return boundaryValue(host_params, "mz");
        }

        SAMS::T_dataType getEnergyBoundary(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return boundaryValue(host_params, "energy");
        }

        SAMS::T_dataType getBxBoundary(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return boundaryValue(host_params, "bx");
        }

        SAMS::T_dataType getByBoundary(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return boundaryValue(host_params, "by");
        }

        SAMS::T_dataType getBzBoundary(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return boundaryValue(host_params, "bz");
        }
    }

    /**
     * Add the conserved variables to a parser, by their names
     * @param parser The parser
     * @param errcode The error code
     */
    void addDeckVariables(eis::parser<SAMS::T_dataType> &parser, eis::error &errcode)
    {
        parser.add_variable("rho", getRho, errcode);
        parser.add_variable("px", getPx, errcode);
        parser.add_variable("py", getPy, errcode);
        parser.add_variable("pz", getPz, errcode);
        parser.add_variable("energy", getEnergy, errcode);
        parser.add_variable("bx", getBx, errcode);
        parser.add_variable("by", getBy, errcode);
        parser.add_variable("bz", getBz, errcode);
        eis::parser_variable_options boundaryOptions;
        boundaryOptions.cap_bits = SAMS::capVariables;
        parser.add_variable("rhoBoundary", getRhoBoundary, errcode, boundaryOptions);
        parser.add_variable("pxBoundary", getPxBoundary, errcode, boundaryOptions);
        parser.add_variable("pyBoundary", getPyBoundary, errcode, boundaryOptions);
        parser.add_variable("pzBoundary", getPzBoundary, errcode, boundaryOptions);
        parser.add_variable("energyBoundary", getEnergyBoundary, errcode, boundaryOptions);
        parser.add_variable("bxBoundary", getBxBoundary, errcode, boundaryOptions);
        parser.add_variable("byBoundary", getByBoundary, errcode, boundaryOptions);
        parser.add_variable("bzBoundary", getBzBoundary, errcode, boundaryOptions);
    }

}
#endif

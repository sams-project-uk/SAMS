
#include "deckVariables.h"
#include "runner.h"
#ifdef USE_DECK_PARSER

namespace LARE
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

        SAMS::T_dataType getEnergyElectron(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return variableValue(host_params, "energy_electron");
        }

        SAMS::T_dataType getEnergyIon(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return variableValue(host_params, "energy_ion");
        }

        SAMS::T_dataType getVx(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return variableValue(host_params, "vx");
        }

        SAMS::T_dataType getVy(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return variableValue(host_params, "vy");
        }

        SAMS::T_dataType getVz(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return variableValue(host_params, "vz");
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

        SAMS::T_dataType getEnergyElectronBoundary(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return boundaryValue(host_params, "energy_electron");
        }

        SAMS::T_dataType getEnergyIonBoundary(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return boundaryValue(host_params, "energy_ion");
        }

        SAMS::T_dataType getVxBoundary(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return boundaryValue(host_params, "vx");
        }

        SAMS::T_dataType getVyBoundary(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return boundaryValue(host_params, "vy");
        }

        SAMS::T_dataType getVzBoundary(eis::count_type, const SAMS::T_dataType*, void* host_params, eis::parser<SAMS::T_dataType>*, eis::status&, eis::error&)
        {
            if (host_params == nullptr) return 0.0;
            return boundaryValue(host_params, "vz");
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
     * Add the variables to a parser, by their names
     * @param parser The parser
     * @param errcode The error code
     */
    void addDeckVariables(eis::parser<SAMS::T_dataType> &parser, eis::error &errcode)
    {
        parser.add_variable("rho", getRho, errcode);
        parser.add_variable("energy_electron", getEnergyElectron, errcode);
        parser.add_variable("energy_ion", getEnergyIon, errcode);
        parser.add_variable("vx", getVx, errcode);
        parser.add_variable("vy", getVy, errcode);
        parser.add_variable("vz", getVz, errcode);
        parser.add_variable("bx", getBx, errcode);
        parser.add_variable("by", getBy, errcode);
        parser.add_variable("bz", getBz, errcode);
        eis::parser_variable_options boundaryOptions;
        boundaryOptions.cap_bits = SAMS::capVariables;
        parser.add_variable("rhoBoundary", getRhoBoundary, errcode, boundaryOptions);
        parser.add_variable("energy_electronBoundary", getEnergyElectronBoundary, errcode, boundaryOptions);
        parser.add_variable("energy_ionBoundary", getEnergyIonBoundary, errcode, boundaryOptions);
        parser.add_variable("vxBoundary", getVxBoundary, errcode, boundaryOptions);
        parser.add_variable("vyBoundary", getVyBoundary, errcode, boundaryOptions);
        parser.add_variable("vzBoundary", getVzBoundary, errcode, boundaryOptions);
        parser.add_variable("bxBoundary", getBxBoundary, errcode, boundaryOptions);
        parser.add_variable("byBoundary", getByBoundary, errcode, boundaryOptions);
        parser.add_variable("bzBoundary", getBzBoundary, errcode, boundaryOptions);
    }

}
#endif

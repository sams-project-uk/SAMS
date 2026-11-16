#ifndef HLLEDECKVARIABLES_H
#define HLLEDECKVARIABLES_H
#ifdef USE_DECK_PARSER

#include <array>
#include "harness.h"
#include "EISpp/eis.h"
#include "deckHelpers/deckBoundaries.h"

namespace conservedVariable
{

    /**
     * Add the conserved variables to a parser, by their names. Their getters read
     * the deckEvaluationPoint that is the host parameters of the parser
     */
    void addDeckVariables(eis::parser<SAMS::T_dataType> &parser, eis::error &errcode);

}
#endif
#endif //HLLEDECKVARIABLES_H

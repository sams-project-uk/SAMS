#ifndef DECKVARIABLES_H
#define DECKVARIABLES_H
#ifdef USE_DECK_PARSER

#include <array>
#include "harness.h"
#include "EISpp/eis.h"
#include "deckHelpers/deckBoundaries.h"

namespace LARE
{

    /**
     * Add the variables to a parser, by their names. Their getters read
     * the deckEvaluationPoint that is the host parameters of the parser
     * @param parser The parser
     * @param errcode The error code
     */
    void addDeckVariables(eis::parser<SAMS::T_dataType> &parser, eis::error &errcode);

}
#endif
#endif //DECKVARIABLES_H

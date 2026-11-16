#ifndef SAMS_DECKPARSERHELPERS_H
#define SAMS_DECKPARSERHELPERS_H

#ifdef USE_DECK_PARSER
#include "KVS/kvs.h"
#include "EISpp/eis.h"
#include <stdexcept>
#include <string>

namespace SAMS
{

    /**
     * The reports of every error that a parser holds
     */
    template<typename T>
    std::string parserErrors(eis::parser<T> &parser)
    {
        std::string reports;
        for (eis::count_type i = 0; i < parser.get_error_count(); ++i) reports += parser.get_error_report(i);
        return reports;
    }

    /**
     * Add every key of every top level constants block to the parser as a stack variable, in deck order
     * @throws std::runtime_error if a key cannot be tokenised
     */
    template<typename T>
    void addDeckConstants(KVS &kvs, eis::parser<T> &parser)
    {
        if (!kvs.root.isType(KVS::Type::Ordered)) return;
        for (KVS::Element &block : kvs.root.get<KVS::Ordered&>().getAllByKey("constants")){
            for (auto &[name, value] : block.get<KVS::Ordered&>().data){
                eis::stack<T> stack;
                eis::error errcode = eis::err_none;
                parser.tokenize(value.template get<std::string>(), stack, errcode);
                if (errcode == eis::err_none) parser.add_stack_variable(name, stack, errcode);
                if (errcode != eis::err_none){
                    throw std::runtime_error("Invalid constants/" + name + "\n" + parserErrors(parser));
                }
            }
        }
    }

}
#endif
#endif // SAMS_DECKPARSERHELPERS_H

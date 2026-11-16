#ifndef DECKINITIALCONDITIONSHELPER_H
#define DECKINITIALCONDITIONSHELPER_H
#include "harness.h"
#include "runner.h"
#include "deckHelpers/deckBoundaries.h"
#include "deckHelpers/deckUtils.h"
#include "deckHelpers/deckEvaluator.h"
#include "loadVariableFromURI.h"
#include <algorithm>
#include <array>
#include <exception>
#include <string>

namespace SAMS
{
	namespace pw = portableWrapper;

	/**
	 * A variable that the initial_conditions block can set, with its name in the deck and its name in the registry
	 */
	struct deckVariableName
	{
		const char *deckName;
		const char *varName;
	};

	/**
	 * Load a variable from the file in deck text such as file(uri). Relative paths are relative to the deck directory.
	 * The host copy of the variable is refreshed afterwards
	 * @param harnessRef The harness, which holds the variable
	 * @param hostData The host data, which holds the host copy of the variable
	 * @param key The key in the initial_conditions block
	 * @param varName The name of the variable in the registry
	 * @param text The deck text
	 * @param files The locations of the files of the run
	 */
	template <T_smallIndexType rank>
	void loadDeckFile(harness &harnessRef, deckHostData<rank> &hostData, const std::string &key, const std::string &varName, const std::string &text,
		const fileInformation &files)
	{
		const std::string::size_type open = text.find('(');
		const std::string uri = text.substr(open + 1, text.rfind(')') - open - 1);
		try {
			loadVariableFromURI<T_dataType, rank>(harnessRef, varName, uri, files.deckDirectory);
		} catch (const std::exception &e) {
			harnessRef.abort("Could not load /initial_conditions/" + key + ": " + e.what(), false);
		}
		hostData.copyToHost(harnessRef, varName);
	}

	/**
	 * Set variables from the initial_conditions block of the deck. The keys are set in the order of the deck, so an expression
	 * can use the values of variables set by earlier keys. A value of file(uri) is loaded from the file at the URI.
	 * Variables that the block does not set, or all of them if there is no block, are zero
	 * @param harnessRef The harness, which holds the variables
	 * @param hostData The host data, which holds the host copies of the variables while they are set
	 * @param kvs The key value store
	 * @param files The locations of the files of the run
	 * @param variables The variables that can be set
	 */
	template <T_smallIndexType rank, typename T_variables>
	void setInitialConditions(harness &harnessRef, deckHostData<rank> &hostData, KVS &kvs, const fileInformation &files, const T_variables &variables)
	{
		for (const deckVariableName &variable : variables) hostData.addVariable(harnessRef, variable.varName);

		KVS::Element block = kvs.getPath("/initial_conditions", KVS::Element());
#ifdef USE_DECK_PARSER
		eis::stack<T_dataType> stack;
		eis::error errcode = eis::err_none;
#endif
		if (block.isType(KVS::Type::Ordered)){
			for (const auto &entry : block.get<KVS::Ordered&>().data){
				const std::string expression = entry.second.template get<std::string>();
				const auto variable = std::find_if(variables.begin(), variables.end(), [&entry](const deckVariableName &v){ return entry.first == v.deckName; });
				if (variable == variables.end()) harnessRef.abort("Invalid /initial_conditions/" + entry.first + ": not a variable that can be set", false);
				if (conditionType(expression) == "file") {
					loadDeckFile(harnessRef, hostData, entry.first, variable->varName, expression, files);
					continue;
				}
				const variableDef &varDef = harnessRef.variableRegistry.getVariable(variable->varName);
				//The domain points that are set, at the staggering of the variable in each dimension
				std::array<pw::Range, rank> ranges;
				for (T_smallIndexType d = 0; d < rank; ++d){
					const dimension &dim = varDef.getDimension(d);
					ranges[d] = harnessRef.axisRegistry.getLocalDomainRange(dim.axisName, dim.stagger);
				}
				pw::hostArray<T_dataType, rank> &array = hostData.variable(variable->varName);
#ifdef USE_DECK_PARSER
				eis::parser<T_dataType> &parser = *hostData.parser;
				parser.tokenize(expression, stack, errcode);
				if (errcode != eis::err_none) harnessRef.abort("Invalid /initial_conditions/" + entry.first + ": " + expression + "\n" + parserErrors(parser), false);
				deckEvaluator<rank> evaluator(harnessRef, parser, stack, hostData, varDef, "Could not evaluate /initial_conditions/" + entry.first + ": " + expression);
				std::apply([&](auto... range){
					pw::applyKernelHostSerial([&](auto... indices){ array(indices...) = evaluator(indices...); }, range...);
				}, ranges);
#else
				//Without the deck parser a value is a number
				T_dataType value = 0.0;
				try {
					value = KVS::Element(expression).asReal();
				} catch (const std::exception &) {
					harnessRef.abort("Invalid /initial_conditions/" + entry.first + ": " + expression + " is not a number, and expressions need the deck parser", false);
				}
				std::apply([&](auto... range){
					pw::applyKernelHostSerial([&](auto... indices){ array(indices...) = value; }, range...);
				}, ranges);
#endif
			}
		}

		//Copy the variables that are on the device back to it
		hostData.copyToDevice(harnessRef);
		hostData.clearVariables(harnessRef);
	}
}

#endif

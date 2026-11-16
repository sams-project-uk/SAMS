#ifndef DECKEVALUATOR_H
#define DECKEVALUATOR_H
#include "harness.h"
#include "deckHelpers/deckUtils.h"
#include "deckHelpers/deckBoundaries.h"
#include <string>
#include <utility>
#include <vector>
#ifdef USE_DECK_PARSER
#include "deckParserHelpers.h"
#include "EISpp/eis.h"
#endif

namespace SAMS
{
	namespace pw = portableWrapper;

#ifdef USE_DECK_PARSER
	/**
	 * Evaluates a tokenized deck expression at single points, for a loop over the points of an array
	 */
	template <T_smallIndexType rank>
	class deckEvaluator
	{
		harness &harnessRef;
		eis::parser<T_dataType> &parser;
		eis::stack<T_dataType> &stack;
		std::string errorText;
		deckEvaluationPoint<rank> point;
		eis::host_options hostOptions;
		std::vector<T_dataType> result;
		eis::error errcode = eis::err_none;

		template <T_smallIndexType... dims>
		static pw::std_N_ary_tuple_type_t<staggerType, rank> staggersOf(const variableDef &varDef, std::integer_sequence<T_smallIndexType, dims...>)
		{
			return {varDef.getDimension(dims).stagger...};
		}

	public:
		/**
		 * @param harnessRef The harness, which reports errors
		 * @param parser The parser
		 * @param stack The tokenized expression
		 * @param hostData The host data that the getters read
		 * @param varDef The variable that is set, which gives the stagger in each dimension
		 * @param errorText The message given if the expression cannot be evaluated
		 */
		deckEvaluator(harness &harnessRef, eis::parser<T_dataType> &parser, eis::stack<T_dataType> &stack, const deckHostData<rank> &hostData,
			const variableDef &varDef, const std::string &errorText)
			: harnessRef(harnessRef), parser(parser), stack(stack), errorText(errorText),
			point{hostData, {}, staggersOf(varDef, std::make_integer_sequence<T_smallIndexType, rank>{})}
		{
		}

		/**
		 * Make the evaluations those of a boundary expression at an edge
		 * @param direction The dimension normal to the edge
		 * @param edge The edge
		 */
		void setBoundary(T_smallIndexType direction, domain::edges edge)
		{
			point.boundaryDimension = direction;
			point.boundaryEdge = edge;
		}

		/**
		 * The value of the expression at a point
		 * @param indices The index of the point in each dimension
		 */
		template <typename... T_indices>
		T_dataType operator()(T_indices... indices)
		{
			point.index = {indices...};
			hostOptions.host_params = &point;
			parser.evaluate(stack, result, errcode, hostOptions);
			if (errcode != eis::err_none) harnessRef.abort(errorText + "\n" + parserErrors(parser), false);
			return result[0];
		}
	};
#endif
}

#endif

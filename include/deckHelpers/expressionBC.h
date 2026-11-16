#ifndef DECKEXPRESSIONBC_H
#define DECKEXPRESSIONBC_H
#include "harness.h"
#include "builtInBoundaryConditions.h"
#include "deckHelpers/deckBoundaries.h"
#include "deckHelpers/deckEvaluator.h"
#include <array>
#include <tuple>
#ifdef USE_DECK_PARSER
#include "EISpp/eis.h"
#endif

namespace SAMS
{
	namespace pw = portableWrapper;

#ifdef USE_DECK_PARSER
	/**
	 * Clamps the ghost cells of an edge to an expression evaluated at their positions on the host.
	 * Reevaluated when the time changes if the expression uses the time or fields
	 */
	template <T_smallIndexType rank>
	class expressionBC : public singleVariableBC<T_dataType, rank>
	{
		harness *harnessPtr;
		timeState *tData;
		deckHostData<rank> &hostData;
		eis::stack<T_dataType> stack;
		//Per edge, indexed as boundaryRanges
		std::array<pw::portableArray<T_dataType, rank>, 2 * rank> values;
		std::array<T_dataType, 2 * rank> evaluatedAt{};
		std::array<bool, 2 * rank> evaluated{};

		void evaluate(int dimension, domain::edges edge, int index)
		{
			auto &ranges = this->boundaryRanges[index];
			auto &manager = harnessPtr->memoryRegistry.getArrayManager();
			pw::hostArray<T_dataType, rank> host;
			std::apply([&](auto... range){ manager.allocate(host, range...); }, ranges);
			if (!evaluated[index]) std::apply([&](auto... range){ manager.allocate(values[index], range...); }, ranges);

			if (stack.cap_bits & capVariables) hostData.copyToHost(*harnessPtr, dimension, edge, tData->time);
			deckEvaluator<rank> evaluator(*harnessPtr, *hostData.parser, stack, hostData, this->varDef, "Could not evaluate a boundary condition expression");
			evaluator.setBoundary(dimension, edge);
			std::apply([&](auto... range){
				pw::applyKernelHostSerial([&](auto... indices){ host(indices...) = evaluator(indices...); }, range...);
			}, ranges);
			manager.copyData(values[index], host);
			manager.deallocate(host);
			evaluatedAt[index] = tData->time;
			evaluated[index] = true;
		}

	public:
		expressionBC(const variableDef &varDef, harness &harnessRef, timeState &tData, deckHostData<rank> &hostData, const eis::stack<T_dataType> &stack)
			: singleVariableBC<T_dataType, rank>(varDef), harnessPtr(&harnessRef), tData(&tData), hostData(hostData), stack(stack)
		{
		}

		void apply(int dimension, domain::edges edge) override
		{
			const int edgeIndex = (edge == domain::edges::lower) ? 0 : 1;
			const int index = dimension * 2 + edgeIndex;
			if (!evaluated[index] || ((stack.cap_bits & (capTime | capVariables)) && evaluatedAt[index] != tData->time)) evaluate(dimension, edge, index);
			pw::portableArray<T_dataType, rank> slice = std::apply([this](auto... range){
				return this->variable(range...);
			}, this->boundaryRanges[index]);
			pw::assign(slice, values[index]);
		}
	};
#endif
}

#endif

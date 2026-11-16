#ifndef DECKSERVICES_H
#define DECKSERVICES_H
#include "harness.h"
#include "runner.h"
#include "deckHelpers/deckBoundaries.h"
#include <cstdint>
#include <string_view>
#include <tuple>
#ifdef USE_DECK_PARSER
#include "EISpp/eis.h"
#endif

namespace SAMS
{
	/**
	 * Services for packages that are set up or controlled from the deck. Its data pack is the host data of the deck, which
	 * every package that requests it shares. With the deck parser, it gives the parser to the host data and adds the position
	 * of the point to the parser, as x, y and z
	 */
	class deckServices
	{
		inline static constexpr auto nameType = makeName("deckServices");

#ifdef USE_DECK_PARSER
		/**
		 * Getter for the position of the point in one dimension
		 * @tparam axis The dimension
		 */
		template <T_smallIndexType axis>
		static T_dataType getPosition(eis::count_type, const T_dataType*, void* host_params, eis::parser<T_dataType>*, eis::status&, eis::error&)
		{
			if (host_params == nullptr) return 0.0;
			const deckEvaluationPoint<3> &point = *static_cast<const deckEvaluationPoint<3>*>(host_params);
			return point.hostData.position(axis, std::get<axis>(point.staggers), std::get<axis>(point.index));
		}
#endif

	public:
		constexpr static std::string_view name = nameType;

		constexpr static bool timeSimulation = false;

		/**
		 * Run before the packages that use the host data
		 */
		constexpr static uint8_t runnerOrder = 32;

		using dataPack = deckHostData<3>;

#ifdef USE_DECK_PARSER
		/**
		 * Give the parser to the host data, and add the position of the point to the parser
		 * @param hostData The host data of the deck
		 * @param parser The parser of the runner
		 */
		void defaultDeck(deckHostData<3> &hostData, eis::parser<T_dataType> &parser)
		{
			hostData.parser = &parser;
			eis::error errcode = eis::err_none;
			parser.add_variable("x", getPosition<0>, errcode);
			parser.add_variable("y", getPosition<1>, errcode);
			parser.add_variable("z", getPosition<2>, errcode);
			if (errcode != eis::err_none) parser.print_errors();
		}
#endif
	};
}

#endif

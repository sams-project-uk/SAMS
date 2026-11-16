#ifndef DECKBOUNDARIES_H
#define DECKBOUNDARIES_H
#include "harness.h"
#include "deckHelpers/deckUtils.h"
#include <algorithm>
#include <array>
#include <initializer_list>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>
#ifdef USE_DECK_PARSER
#include "EISpp/eis.h"
#endif

namespace SAMS
{
	namespace pw = portableWrapper;

	/**
	 * Host copies of the last domain plane at each edge of variables in the registry, for deck expressions at the boundaries.
	 * Planes keep the indices of their variable, and point at the variable if it is available on the host
	 */
	template <T_smallIndexType rank>
	class edgePlanes
	{
		static constexpr T_smallIndexType edgeCount = static_cast<T_smallIndexType>(domain::edges::EDGE_COUNT);
		template <typename T_element>
		using perEdge = std::array<std::array<T_element, edgeCount>, rank>;
		struct entry
		{
			std::string name;
			//True if the planes are copies of the device slices
			bool copies = false;
			perEdge<pw::acceleratedArray<T_dataType, rank>> deviceSlices;
			perEdge<pw::hostArray<T_dataType, rank>> planes;
		};
		std::vector<entry> entries;
		perEdge<T_dataType> copiedAt{};
		perEdge<bool> copied{};

		static constexpr T_smallIndexType side(domain::edges edge) { return static_cast<T_smallIndexType>(edge); }

	public:
		/**
		 * Add the planes of a variable
		 * @param harnessRef The harness, which holds the variable
		 * @param name The name of the variable in the registry
		 */
		void add(harness &harnessRef, const std::string &name)
		{
			const variableDef &varDef = harnessRef.variableRegistry.getVariable(name);
			auto &manager = harnessRef.memoryRegistry.getArrayManager();
			entry &e = entries.emplace_back();
			e.name = name;
			const bool onDevice = varDef.getMemorySpace() != pw::arrayTags::host;
			pw::hostArray<T_dataType, rank> hostVariable;
			pw::acceleratedArray<T_dataType, rank> deviceVariable;
			if (onDevice) varDef.fillPPArray(deviceVariable);
			else varDef.fillPPArray(hostVariable);
			for (T_smallIndexType direction = 0; direction < rank; ++direction){
				const dimension &normal = varDef.getDimension(direction);
				for (domain::edges edge : {domain::edges::lower, domain::edges::upper}){
					const T_indexType plane = edge == domain::edges::lower ? normal.getLocalDomainLB() : normal.getLocalDomainUB();
					std::array<pw::Range, rank> ranges;
					for (T_smallIndexType d = 0; d < rank; ++d){
						ranges[d] = d == direction ? pw::Range(plane, plane) : varDef.getDimension(d).getLocalRange();
					}
					//Slices start at 0, so they are rebased to the indices of the variable
					auto slice = [&ranges](const auto &variable){
						auto result = std::apply([&variable](auto... range){ return variable(range...); }, ranges);
						std::apply([&result](auto... range){ result.rebase(range...); }, ranges);
						return result;
					};
					pw::hostArray<T_dataType, rank> &hostPlane = e.planes[direction][side(edge)];
					if (onDevice){
						pw::acceleratedArray<T_dataType, rank> &deviceSlice = e.deviceSlices[direction][side(edge)];
						deviceSlice = slice(deviceVariable);
						hostPlane = manager.makeHostAvailable(deviceSlice);
					} else {
						hostPlane = slice(hostVariable);
					}
				}
			}
			e.copies = onDevice && e.planes[0][0].data() != e.deviceSlices[0][0].data();
		}

		/**
		 * Copy the planes of every variable at an edge off the device, once for each time
		 * @param harnessRef The harness, which holds the array manager
		 * @param direction The dimension normal to the edge
		 * @param edge The edge
		 * @param time The time of the values
		 */
		void copyToHost(harness &harnessRef, T_smallIndexType direction, domain::edges edge, T_dataType time)
		{
			if (copied[direction][side(edge)] && copiedAt[direction][side(edge)] == time) return;
			auto &manager = harnessRef.memoryRegistry.getArrayManager();
			for (entry &e : entries){
				if (e.copies) manager.copyData(e.planes[direction][side(edge)], e.deviceSlices[direction][side(edge)]);
			}
			copiedAt[direction][side(edge)] = time;
			copied[direction][side(edge)] = true;
		}

		/**
		 * Copy the planes of every variable at an edge back onto the device
		 * @param harnessRef The harness, which holds the array manager
		 * @param direction The dimension normal to the edge
		 * @param edge The edge
		 */
		void copyToDevice(harness &harnessRef, T_smallIndexType direction, domain::edges edge)
		{
			auto &manager = harnessRef.memoryRegistry.getArrayManager();
			for (entry &e : entries){
				if (e.copies) manager.copyData(e.deviceSlices[direction][side(edge)], e.planes[direction][side(edge)]);
			}
		}

		/**
		 * The plane of a variable at an edge
		 * @param name The name of the variable in the registry
		 * @param direction The dimension normal to the edge
		 * @param edge The edge
		 */
		const pw::hostArray<T_dataType, rank> &get(std::string_view name, T_smallIndexType direction, domain::edges edge) const
		{
			const auto it = std::find_if(entries.begin(), entries.end(), [name](const entry &e){ return e.name == name; });
			return it->planes[direction][side(edge)];
		}
	};

	template <T_smallIndexType rank>
	class deckHostData;

	/**
	 * The host parameters of one evaluation of a deck expression. It holds the point and the edge, and the getters read the
	 * variables from the host data through it
	 */
	template <T_smallIndexType rank>
	struct deckEvaluationPoint
	{
		const deckHostData<rank> &hostData;
		pw::std_N_ary_tuple_type_t<T_indexType, rank> index{};
		//The stagger of the variable being set in each dimension
		pw::std_N_ary_tuple_type_t<staggerType, rank> staggers{};
		//In a boundary expression, the dimension normal to the boundary and its edge
		T_smallIndexType boundaryDimension = 0;
		domain::edges boundaryEdge = domain::edges::lower;
	};

	/**
	 * The host copies of the variables that deck expressions read by name, both whole and on their planes at the boundaries
	 */
	template <T_smallIndexType rank>
	class deckHostData
	{
		deckHostVariables<rank> hostVariables;
		edgePlanes<rank> planes;
		struct axisPositions
		{
			staggerType stagger;
			pw::hostArray<T_dataType, 1> positions;
		};
		//The host positions of the points of each dimension, for each stagger that is used
		std::array<std::vector<axisPositions>, rank> axes;

	public:
#ifdef USE_DECK_PARSER
		//The parser of the runner, which deck expressions are evaluated with
		eis::parser<T_dataType> *parser = nullptr;
#endif

		/**
		 * Add a variable, for expressions that use the variables. It is made available on the host
		 * @param harnessRef The harness, which holds the variable
		 * @param name The name of the variable in the registry
		 * @param copyBack If true the host array is copied from the device, and if false it is set to zero
		 */
		void addVariable(harness &harnessRef, const std::string &name, bool copyBack = false)
		{
			hostVariables.addVariable(harnessRef, name, copyBack);
			addPositions(harnessRef, harnessRef.variableRegistry.getVariable(name));
		}

		/**
		 * Copy the host array of a variable off the device
		 * @param harnessRef The harness, which holds the array manager
		 * @param name The name of the variable in the registry
		 */
		void copyToHost(harness &harnessRef, std::string_view name)
		{
			hostVariables.copyToHost(harnessRef, name);
		}

		/**
		 * Copy the host arrays of the variables back onto the device
		 * @param harnessRef The harness, which holds the array manager
		 */
		void copyToDevice(harness &harnessRef)
		{
			hostVariables.copyToDevice(harnessRef);
		}

		/**
		 * Release the host copies of the variables and remove the variables
		 * @param harnessRef The harness, which holds the array manager
		 */
		void clearVariables(harness &harnessRef)
		{
			hostVariables.clearVariables(harnessRef);
		}

		/**
		 * The host array of a variable
		 * @param name The name of the variable in the registry
		 */
		pw::hostArray<T_dataType, rank> &variable(std::string_view name)
		{
			return hostVariables.variable(name);
		}

		/**
		 * The host array of a variable
		 * @param name The name of the variable in the registry
		 */
		const pw::hostArray<T_dataType, rank> &variable(std::string_view name) const
		{
			return hostVariables.variable(name);
		}

		/**
		 * Add the planes of a variable, for boundary expressions that use the variables at the boundary
		 * @param harnessRef The harness, which holds the variable
		 * @param name The name of the variable in the registry
		 */
		void addPlanes(harness &harnessRef, const std::string &name)
		{
			planes.add(harnessRef, name);
			addPositions(harnessRef, harnessRef.variableRegistry.getVariable(name));
		}

		/**
		 * Copy the planes of every variable at an edge off the device, once for each time
		 * @param harnessRef The harness, which holds the array manager
		 * @param direction The dimension normal to the edge
		 * @param edge The edge
		 * @param time The time of the values
		 */
		void copyToHost(harness &harnessRef, T_smallIndexType direction, domain::edges edge, T_dataType time)
		{
			planes.copyToHost(harnessRef, direction, edge, time);
		}

		/**
		 * The plane of a variable at an edge
		 * @param name The name of the variable in the registry
		 * @param direction The dimension normal to the edge
		 * @param edge The edge
		 */
		const pw::hostArray<T_dataType, rank> &plane(std::string_view name, T_smallIndexType direction, domain::edges edge) const
		{
			return planes.get(name, direction, edge);
		}

		/**
		 * Add the host positions of the points of a variable in each dimension, at its stagger, if they are not already held
		 * @param harnessRef The harness, which holds the axes
		 * @param varDef The variable
		 */
		void addPositions(harness &harnessRef, const variableDef &varDef)
		{
			for (T_smallIndexType d = 0; d < rank; ++d){
				const dimension &dim = varDef.getDimension(d);
				std::vector<axisPositions> &held = axes[d];
				if (std::find_if(held.begin(), held.end(), [&dim](const axisPositions &a){ return a.stagger == dim.stagger; }) != held.end()) continue;
				axisPositions &a = held.emplace_back();
				a.stagger = dim.stagger;
				harnessRef.axisRegistry.fillPPLocalAxis(dim.axisName, a.positions, dim.stagger);
			}
		}

		/**
		 * The position of a point in a dimension
		 * @param direction The dimension
		 * @param stagger The stagger of the point
		 * @param index The index of the point in the dimension
		 */
		T_dataType position(T_smallIndexType direction, staggerType stagger, T_indexType index) const
		{
			const std::vector<axisPositions> &held = axes[direction];
			return std::find_if(held.begin(), held.end(), [stagger](const axisPositions &a){ return a.stagger == stagger; })->positions(index);
		}

	};
}

#endif

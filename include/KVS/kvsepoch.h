#ifndef KVSEPOCH_H
#define KVSEPOCH_H

#include <string>
#include <vector>
#include <regex>
#include <fstream>
#include <sstream>
#include "kvs.h"
#include "kvsutils.h"

/*This block reads and writes the KVS contents to an EPOCH (Warwick Plasma PIC code) (github.com/warwick-plasma/epoch)
	input deck format. It is an extended version of EPOCH's deck compatible with EIS-2 (https://github.com/csbrady-warwick/EIS-2

	ordered data map to blocks (EIS-2 and EPOCH) and sub blocks (EIS-2 only) in the standard
begin:blockname
key = value
key2 = value2
begin:subblockname
key3 = value3
end:subblockname
end:blockname

form. Arrays become (A,B,C) and maps become {key1=value1, key2=value2}.
Ranged values are represented as (value, LB, UB)
*/

namespace SAMS
{

	namespace KVSConversion
	{

		namespace
		{

			inline std::string trimWhitespace(const std::string &str)
			{
				size_t first = str.find_first_not_of(" \t\r");
				size_t last = str.find_last_not_of(" \t\r");
				if (first == std::string::npos || last == std::string::npos)
					return "";
				return str.substr(first, last - first + 1);
			}

			std::string orderedToEPOCH(const std::shared_ptr<KVS::Ordered> ordered, int indentLevel = 0);
			std::string arrayToEPOCH(const std::shared_ptr<KVS::Array> array);
			std::string mapToEPOCH(const std::shared_ptr<KVS::Map> map, int indentLevel = 0);
			std::string mapToInlineEPOCH(const std::shared_ptr<KVS::Map> map);
			std::string rangedToEPOCH(const KVS::Element &element);
			std::string valueToEPOCH(const KVS::Element &element, int indentLevel=0);

			inline std::string valueToEPOCH(const KVS::Element &element, int indentLevel)
			{
				switch (element.type)
				{
					case KVS::Type::Int32:
						return std::to_string(std::get<int32_t>(element.value));
					case KVS::Type::Int64:
						return std::to_string(std::get<int64_t>(element.value));
					case KVS::Type::Float:
						return floatToString(std::get<float>(element.value));
					case KVS::Type::Double:
						return doubleToString(std::get<double>(element.value));
					case KVS::Type::String:
						{
							std::string quote = "";
							try {
								if (element.getMetadata("isEPOCHStringQuoted") == "true")
								{
									quote = "\"";
								}
							}
							catch (const std::out_of_range&)
							{
								// No metadata, treat as normal string
							}
							return quote + std::get<std::string>(element.value) + quote;
						}
					case KVS::Type::Bool:
						return std::get<bool>(element.value) ? "T" : "F";
					case KVS::Type::Int32Range:
					case KVS::Type::Int64Range:
					case KVS::Type::FloatRange:
					case KVS::Type::DoubleRange:
						return rangedToEPOCH(element);
					case KVS::Type::Array:
						return arrayToEPOCH(std::get<std::shared_ptr<KVS::Array>>(element.value));
					case KVS::Type::Map:
						try {
							if (element.getMetadata("isEPOCHInlineMap") == "true")
								return mapToInlineEPOCH(std::get<std::shared_ptr<KVS::Map>>(element.value));
							else {
								return mapToEPOCH(std::get<std::shared_ptr<KVS::Map>>(element.value), indentLevel);
							}
						}
						catch (...)
						{
							return mapToEPOCH(std::get<std::shared_ptr<KVS::Map>>(element.value), indentLevel);
						}
					case KVS::Type::Ordered:
						return orderedToEPOCH(std::get<std::shared_ptr<KVS::Ordered>>(element.value), indentLevel);
					case KVS::Type::None:
						return "NULL";
					default:
						throw std::runtime_error("Unsupported element type for value conversion");
				}
			}

			inline std::string rangedToEPOCH(const KVS::Element &element)
			{
				switch (element.type)
				{
					case KVS::Type::Int32Range:
						{
							const auto &range = std::get<KVS::ranged<int32_t>>(element.value);
							return "(" + std::to_string(range.value) + ", " + std::to_string(range.begin) + ", " + std::to_string(range.end) + ")";
						}
					case KVS::Type::Int64Range:
						{
							const auto &range = std::get<KVS::ranged<int64_t>>(element.value);
							return "(" + std::to_string(range.value) + ", " + std::to_string(range.begin) + ", " + std::to_string(range.end) + ")";
						}
					case KVS::Type::FloatRange:
						{
							const auto &range = std::get<KVS::ranged<float>>(element.value);
							return "(" + floatToString(range.value) + ", " + floatToString(range.begin) + ", " + floatToString(range.end) + ")";
						}
					case KVS::Type::DoubleRange:
						{
							const auto &range = std::get<KVS::ranged<double>>(element.value);
							return "(" + doubleToString(range.value) + ", " + doubleToString(range.begin) + ", " + doubleToString(range.end) + ")";
						}
					default:
						throw std::runtime_error("Unsupported element type for ranged conversion");
				}
			}

			inline std::string arrayToEPOCH(const std::shared_ptr<KVS::Array> array)
			{
				std::string result = "(";
				for (size_t i = 0; i < array->elements.size(); ++i)
				{
					result += valueToEPOCH(array->elements[i]);
					if (i < array->elements.size() - 1)
						result += ", ";
				}
				result += ")";
				return result;
			}

			/**
			 * Convert a comment to EPOCH format. Separate comments are separated by \r, and are written as # comments, except that a comment of several lines, separated by \n, is written as a multi line comment
			 * @param indent The indent of the comment
			 * @param comment The comment
			 */
			inline std::string commentToEPOCH(const std::string &indent, const std::string &comment)
			{
				std::string result;
				std::istringstream comments(comment);
				for (std::string part; std::getline(comments, part, '\r');)
				{
					if (part.find('\n') == std::string::npos)
					{
						result += indent + "# " + part + "\n";
						continue;
					}
					result += indent + "/*\n";
					std::istringstream lines(part);
					for (std::string line; std::getline(lines, line);) result += indent + line + "\n";
					result += indent + "*/\n";
				}
				return result;
			}

			inline std::string keyValueToEPOCH(const KVS::Element &keyElement, const KVS::Element &value, int indentLevel)
			{
				std::string indent(indentLevel * 2, ' ');
				std::string block, key;
				if (keyElement.type == KVS::Type::String)
				{
					key = std::get<std::string>(keyElement.value);
				}
				else
				{
					std::stringstream ss;
					ss << keyElement;
					key = ss.str();
				}
				bool isBlockMap = (value.type == KVS::Type::Map);
				try{
					isBlockMap = isBlockMap && !(value.getMetadata("isEPOCHInlineMap") == "true");
				}
				catch (...)
				{
					isBlockMap = true;
				}
				if (value.type == KVS::Type::Ordered || isBlockMap)
				{
					block += commentToEPOCH(indent, value.hasMetadata("beginComment") ? value.getMetadata("beginComment") : "");
					block += indent + "begin:" + key + "\n";
					block += valueToEPOCH(value, indentLevel + 1);
					//The comment before the end of the block is indented as the contents of the block
					block += commentToEPOCH(indent + "  ", value.hasMetadata("endComment") ? value.getMetadata("endComment") : "");
					block += indent + "end:" + key + "\n";
					//Aesthetically extra space between top level blocks, but not between nested blocks
					if (indentLevel == 0)
						block += "\n";
				}
				else
				{
					block += commentToEPOCH(indent, value.getComment());
					std::string separator =" = ";
					try{
						if (value.metadata.at("isEPOCHDirective") == "true")
						{
							separator = " : ";
						}
					}
					catch (const std::out_of_range&)
					{
						// No metadata, treat as normal key-value pair
					}
					block += indent + key + separator + valueToEPOCH(value) + "\n";
				}
				return block;
			}

			inline std::string mapToEPOCH(const std::shared_ptr<KVS::Map> map, int indentLevel)
			{
				//This is just another block, but where the order is unimportant
				std::string result;
				std::string indent(indentLevel * 2, ' ');
				std::string separator = " = ";
				for (const auto &pair : map->data)
				{
					result += keyValueToEPOCH(pair.first, pair.second, indentLevel);   
				}
				return result;
			}

			inline std::string mapToInlineEPOCH(const std::shared_ptr<KVS::Map> map)
			{
				std::string result = "{";
				size_t count = 0;
				for (const auto &pair : map->data)
				{
					result += valueToEPOCH(pair.first) + "=" + valueToEPOCH(pair.second);
					if (count < map->data.size() - 1)
						result += ", ";
					++count;
				}
				result += "}";
				return result;
			}

			inline std::string orderedToEPOCH(const std::shared_ptr<KVS::Ordered> ordered, int indentLevel)
			{
				std::string indent(indentLevel * 2, ' ');
				std::string block;
				for (const auto &pair : ordered->data)
				{
					const std::string &key = pair.first;
					const KVS::Element &value = pair.second;
					block += keyValueToEPOCH(key, value, indentLevel);
				}
				return block;
			}

			/**
			 * Get a trivial vector of strings representing the lines of an input string
			 * @param str The input string to be split into lines
			 * @return A vector of strings, each representing a line from the input string
			 */
			inline std::vector<std::string> naiveSplitLines(const std::string &str)
			 {
				std::vector<std::string> lines;
				size_t start = 0;
				size_t end = str.find('\n');
				while (end != std::string::npos)
				{
					lines.push_back(str.substr(start, end - start));
					start = end + 1;
					end = str.find('\n', start);
				}
				lines.push_back(str.substr(start));
				return lines;
			}

			/**
			 * Convert an EPOCH deck string to a vector of lines, handling line continuations (lines ending with \) and comments (a # at the start of a line or after whitespace). Leading and trailing whitespace is removed from each line, and lines that are empty after removing comments are discarded.
			 * @param str The EPOCH deck string to be split into lines.
			 * @return A vector of strings, each representing a line from the EPOCH deck, with line continuations and comments handled appropriately.
			 */
			inline std::vector<std::pair<std::string, std::string>> splitLines(const std::string &str)
			{
				std::vector<std::pair<std::string, std::string>> result;
				std::vector<std::string> lines = naiveSplitLines(str);
				for (auto &line : lines)
				{
					line = trimWhitespace(line);
					if (!line.empty())
					{
						result.emplace_back(line, "");
					}
				}

				//Multi line comments, an EIS-2 feature. The text of a comment becomes the comment of the line that it starts on
				size_t index = 0;
				while (index < result.size())
				{
					const size_t commentStart = result[index].first.find("/*");
					if (commentStart == std::string::npos)
					{
						++index;
						continue;
					}
					//Find the end of the comment, on the same line or a later one
					size_t endLine = index;
					size_t commentEnd = result[index].first.find("*/", commentStart + 2);
					while (commentEnd == std::string::npos && endLine + 1 < result.size())
					{
						++endLine;
						commentEnd = result[endLine].first.find("*/");
					}
					std::string comment;
					auto addCommentLine = [&comment](const std::string &text){
						const std::string trimmed = trimWhitespace(text);
						if (!trimmed.empty()) comment += (comment.empty() ? "" : "\n") + trimmed;
					};
					if (endLine == index)
					{
						addCommentLine(result[index].first.substr(commentStart + 2, commentEnd - commentStart - 2));
					}
					else
					{
						addCommentLine(result[index].first.substr(commentStart + 2));
						for (size_t i = index + 1; i < endLine; ++i) addCommentLine(result[i].first);
						addCommentLine(result[endLine].first.substr(0, commentEnd));
					}
					const std::string after = result[endLine].first.substr(commentEnd + 2);
					result[index].first = trimWhitespace(result[index].first.substr(0, commentStart) + " " + after);
					result.erase(result.begin() + index + 1, result.begin() + endLine + 1);
					result[index].second = result[index].second.empty() ? comment : result[index].second + "\r" + comment;
				}


				// Now loop through and find any lines that have a # character and remove the # and everything after it. If the line is empty after this, remove it from the vector.
				// A # only starts a comment at the start of a line or after whitespace, and never inside quotes, so that a # in a value such as a URI is kept
				for (auto it = result.begin(); it != result.end();)
				{
					size_t commentPos = std::string::npos;
					bool inQuotes = false;
					for (size_t i = 0; i < it->first.size(); ++i)
					{
						const char c = it->first[i];
						//An escaped quote does not start or end a quoted string
						if (c == '"' && (i == 0 || it->first[i - 1] != '\\'))
						{
							inQuotes = !inQuotes;
						}
						else if (c == '#' && !inQuotes && (i == 0 || std::isspace(static_cast<unsigned char>(it->first[i - 1]))))
						{
							commentPos = i;
							break;
						}
					}
					if (commentPos != std::string::npos)
					{
						std::string comment = trimWhitespace(it->first.substr(commentPos+1));
						*it = std::make_pair(it->first.substr(0, commentPos), it->second);
						//The comment of the line follows any comment that it already has
						it->second = it->second.empty() ? comment : it->second + "\r" + comment;
					}
					if (it->first.empty())
					{
						//A line that is only a comment after the last line is kept, so that its comment can be given to the deck
						if (std::next(it) == result.end())
						{
							++it;
							continue;
						}
						//A line that is only a comment gives its comment to the next line, before that line's own comment
						std::string comment = it->second;
						it = result.erase(it);
						it->second = it->second.empty() ? comment : comment + "\r" + it->second;
					}
					else
					{
						++it;
					}
				}

				// Find any lines that finish with a \ character and merge them with the next line, removing the \ character. Repeat until there are no more lines that finish with a \ character.
				for (size_t i = 0; i < result.size(); ++i)
				{
					while (!result[i].first.empty() && result[i].first.back() == '\\')
					{
						result[i].first.pop_back();
						if (i + 1 < result.size())
						{
							result[i].first += result[i + 1].first;
							result.erase(result.begin() + i + 1);
						}
					}
				}

				// Final stage is to spin through and find any lines that start with "import", any
				// white space follower by : and then a file name. Any white
				std::regex importRegex(R"(^\s*import\s*:\s*(\S+))");
				for (size_t i = 0; i < result.size(); ++i)
				{
					std::smatch match;
					if (std::regex_search(result[i].first, match, importRegex))
					{
						std::string filename = match[1];
						// Read the file and split it into lines
						std::ifstream file(filename);
						if (!file.is_open())
						{
							throw std::runtime_error("Could not open imported file: " + filename);
						}
						std::stringstream buffer;
						buffer << file.rdbuf();
						std::vector<std::pair<std::string, std::string>> splitResult = splitLines(buffer.str());
						// Insert the imported lines into the main lines vector at the position of the import statement, replacing the import statement
						const std::string importComment = result[i].second;
						result.erase(result.begin() + i);
						// A comment after the last line of the imported file is given to the line after the import, before that line's own comment
						if (!splitResult.empty() && splitResult.back().first.empty() && i < result.size())
						{
							const std::string comment = splitResult.back().second;
							splitResult.pop_back();
							result[i].second = result[i].second.empty() ? comment : comment + "\r" + result[i].second;
						}
						// The comment of the import line is given to the first imported line. If nothing was imported it is given to the line after the import,
						// or kept as the comment after the last line of the deck if there is no line after the import
						if (!importComment.empty())
						{
							if (!splitResult.empty())
							{
								std::string &first = splitResult.front().second;
								first = first.empty() ? importComment : importComment + "\r" + first;
							}
							else if (i < result.size())
							{
								result[i].second = result[i].second.empty() ? importComment : importComment + "\r" + result[i].second;
							}
							else
							{
								splitResult.emplace_back("", importComment);
							}
						}
						result.insert(result.begin() + i, splitResult.begin(), splitResult.end());
						// Move the index back to the position of the first imported line
						i += splitResult.size() - 1;
					}
				}

				return result;
			}

			KVS::Element valueFromEPOCH(const std::string &valueText);
			KVS::Element arrayFromEPOCH(const std::string &valueText);
			KVS::Element mapFromEPOCH(const std::string &valueText);
			KVS::Element orderedFromEPOCH(const std::vector<std::pair<std::string, std::string>> &lines, const std::string &currentBlockName, size_t &index);

			inline KVS::Element valueFromEPOCH(const std::string &valueText)
			{
				// Check if the value starts and ends with a " character, if so it is a string and we should remove the " characters and unescape any escaped characters

				// Regex to match a string value, capturing the contents without the surrounding quotes and handling escaped characters
				std::regex stringRegex(R"DELIM(^\s*"((?:\\.|[^"\\])*)"\s*$)DELIM");
				std::smatch match;
				if (std::regex_search(valueText, match, stringRegex))
				{
					KVS::Element el(match[1]);
					el.setMetadata("isEPOCHStringQuoted", "true");
					return el;
				}

				// Next check for an array value, which starts with ( and ends with )
				std::regex arrayRegex(R"DELIM(^\s*\((.*)\)\s*$)DELIM");
				if (std::regex_search(valueText, match, arrayRegex))
				{
					return arrayFromEPOCH(match[1]);
				}

				//Next check for a map value, which starts with { and ends with }
				std::regex mapRegex(R"DELIM(^\s*\{(.*)\}\s*$)DELIM");
				if (std::regex_search(valueText, match, mapRegex))
				{
					return mapFromEPOCH(match[1]);
				}

				//Unquoted T, F, true and false are booleans
				std::string trimmedValue = trimWhitespace(valueText);
				if (trimmedValue == "T" || trimmedValue == "true")
				{
					return KVS::Element(true);
				}
				if (trimmedValue == "F" || trimmedValue == "false")
				{
					return KVS::Element(false);
				}

				//Now try to parse as an integer, then a float, then assume that it is a string without quotes
				try
				{
					size_t pos;
					int64_t intValue = std::stoll(valueText, &pos);
					//Check that the entire string was parsed to ensure that we don't accidentally parse something like "123abc" as a valid number
					if (pos == valueText.size())
					{
						if (intValue >= std::numeric_limits<int32_t>::min() && intValue <= std::numeric_limits<int32_t>::max())
						{
							return KVS::Element(static_cast<int32_t>(intValue));
						}
						else
						{
							return KVS::Element(intValue);
						}
					}
				}
				catch (...)
				{}

				// Try to parse as a double, which also covers floats. We will check if the entire string was parsed to ensure that we don't accidentally parse something like "1.0abc" as a valid number.
				try
				{
					size_t pos;
					double doubleValue = std::stod(valueText, &pos);
					if (pos == valueText.size())
					{
						//Don't try and create floats from text, just use double
						return KVS::Element(doubleValue);
					}

				}
				catch (...)
				{}

				//Now it is just an unquoted string, we will trim any whitespace from the start and end of the string
				if (!valueText.empty())
				{
					return KVS::Element(trimWhitespace(valueText));
				}

				throw std::runtime_error("Unsupported value format in EPOCH deck: " + valueText);
			}

			inline KVS::Element arrayFromEPOCH(const std::string &valueText)
			{
				// Split the array contents by commas, but ignore commas that are inside parentheses or braces (or inside quotes)
				KVS::Array elements;
				std::string currentElement;
				int parenDepth = 0;
				int braceDepth = 0;
				bool inQuotes = false;
				for (char c : valueText)
				{
					currentElement += c;
					if (c == '"')
					{
						inQuotes = !inQuotes;
					}
					else if (!inQuotes)
					{
						if (c == '(')
							parenDepth++;
						else if (c == ')')
							parenDepth--;
						else if (c == '{')
							braceDepth++;
						else if (c == '}')
							braceDepth--;
						else if (c == ',' && parenDepth == 0 && braceDepth == 0)
						{
							//Pull the comma out of the current element and trim any whitespace from the start and end of the element
							currentElement.pop_back();
							currentElement = trimWhitespace(currentElement);
							elements.push_back(valueFromEPOCH(currentElement));
							currentElement.clear();
							continue;
						}
					}
				}
				if (!currentElement.empty())
				{
					elements.push_back(valueFromEPOCH(currentElement));
				}
				return elements;
			}

			inline KVS::Element mapFromEPOCH(const std::string &valueText)
			{
				// Split the map contents by commas, but ignore commas that are inside parentheses or braces (or inside quotes)
				KVS::Map map;
				std::string currentPair;
				int parenDepth = 0;
				int braceDepth = 0;
				bool inQuotes = false;
				for (char c : valueText)
				{
					currentPair += c;
					if (c == '"')
					{
						inQuotes = !inQuotes;
					}
					else if (!inQuotes)
					{
						if (c == '(')
							parenDepth++;
						else if (c == ')')
							parenDepth--;
						else if (c == '{')
							braceDepth++;
						else if (c == '}')
							braceDepth--;
						else if (c == ',' && parenDepth == 0 && braceDepth == 0)
						{
							//Pull the comma out of the current pair and trim any whitespace from the start and end of the pair
							currentPair.pop_back();
							currentPair = trimWhitespace(currentPair);
							//Now split the pair by the first = character, ignoring any = characters that are inside parentheses or braces or quotes
							size_t equalsPos = currentPair.find('=');
							if (equalsPos != std::string::npos)
							{
								std::string key = trimWhitespace(currentPair.substr(0, equalsPos));
								std::string value = trimWhitespace(currentPair.substr(equalsPos + 1));
								map.upsert(key);
								map[key] = valueFromEPOCH(value);
								currentPair.clear();
							}
						}
					}
				}
				if (!currentPair.empty())
				{
					//Handle the last pair
					currentPair = trimWhitespace(currentPair);
					size_t equalsPos = currentPair.find('=');
					if (equalsPos != std::string::npos)
					{
						std::string key = trimWhitespace(currentPair.substr(0, equalsPos));
						std::string value = trimWhitespace(currentPair.substr(equalsPos + 1));
						auto &keyElement = map.upsert(key);
						keyElement = valueFromEPOCH(value);
					}
				}
				KVS::Element el(map);
				el.setMetadata("isEPOCHInlineMap", "true");
				return el;
			}


			inline KVS::Element orderedFromEPOCH(const std::vector<std::pair<std::string, std::string>> &lines, const std::string &startName, size_t &index)
			{
				KVS::Ordered ordered;
				std::string blockName;
				while (index < lines.size())
				{
					const std::string &line = lines[index].first;
					//An empty line only holds the comment after the last line of the deck
					if (line.empty())
					{
						index++;
						continue;
					}
					if (line.rfind("begin:", 0) == 0)
					{
						blockName = line.substr(6);
						//Trim any whitespace from the block name
						blockName = trimWhitespace(blockName);
						const std::string beginComment = lines[index].second;
						index++;//Skip over start
						KVS::Element &el = ordered.push_back(blockName, orderedFromEPOCH(lines, blockName, index));
						//The recursion stops after the end line, so the end line is the one before index
						const std::string endComment = lines[index-1].second;
						if (!beginComment.empty()) el.setMetadata("beginComment", beginComment);
						if (!endComment.empty()) el.setMetadata("endComment", endComment);
						continue;
					}
					else if (line.rfind("end:", 0) == 0)
					{
						//Check that the block name matches the most recent begin block
						std::string endBlockName = line.substr(4);
						//Trim any whitespace from the end block name
						endBlockName = trimWhitespace(endBlockName);
						if (endBlockName != startName)
						{
							throw std::runtime_error("Mismatched end block: " + endBlockName + " does not match begin block: " + startName);
						}
						index++;//Skip over end
						return ordered;
					}
					else
					{
						// This should be a key=value pair, although it might be key:value instead. We will split on the first = or : character we find, and trim any whitespace from the key and value
						size_t equalsPos = line.find('=');
						size_t colonPos = line.find(':');
						bool colonSeparator = false;
						if (equalsPos == std::string::npos && colonPos == std::string::npos)
						{
							throw std::runtime_error("Invalid line in EPOCH deck: " + line);
						}
						//Since either character may be present in the value, find the first occurrence of either character and split on that
						size_t delimiterPos;
						if (equalsPos != std::string::npos && colonPos != std::string::npos)
						{
							if (equalsPos < colonPos)
							{
								delimiterPos = equalsPos;
							}
							else
							{
								colonSeparator = true;
								delimiterPos = colonPos;
							}
						}
						else if (equalsPos != std::string::npos)
						{
							delimiterPos = equalsPos;
						}
						else if (colonPos != std::string::npos)
						{
							colonSeparator = true;
							delimiterPos = colonPos;
						} else 
						{
							throw std::runtime_error("Invalid line in EPOCH deck: " + line);
						}
						std::string key = trimWhitespace(line.substr(0, delimiterPos));
						std::string valueText = trimWhitespace(line.substr(delimiterPos + 1));
						KVS::Element &el = ordered.push_back(key, valueFromEPOCH(valueText));
						el.setComment(lines[index].second);
						el.setMetadata("isEPOCHDirective", colonSeparator ? "true" : "false");
					}
					index++;
				}
				return ordered;
			}
		}

		/**
		 * Function to split a comment line into multiple comment lines if it exceeds a maximum line length.
		 */
		inline std::string splitCommentLine(const std::string &line, size_t maxLineLength)
		{
			//Count the number of white space characters at the start of the line, we will preserve this indentation in the split lines
			size_t indentLength = 0;
			while (indentLength < line.length() && std::isspace(line[indentLength]))
			{
				indentLength++;
			}
			std::string indent(indentLength, ' ');
			//Strip off the indent and the # character at the start of the line to get the actual comment text, we will add the # character back in when we create the split lines
			std::string commentText = line.substr(indentLength);
			//Loop through the line and split it into multiple lines. We will try to split on spaces, but if there are no spaces we will split on the last character that fits within the max line length. We will also preserve any indentation at the start of the line in the split lines.
			std::vector<std::string> splitLines;
			size_t start = 0;
			while (start < commentText.length())
			{
				size_t end = start + maxLineLength;
				if (end >= commentText.length())
				{
					splitLines.push_back(commentText.substr(start));
					break;
				}
				size_t lastSpace = commentText.rfind(' ', end);
				if (lastSpace != std::string::npos && lastSpace > start)
				{
					splitLines.push_back(commentText.substr(start, lastSpace - start));
					start = lastSpace + 1;
				}
				else
				{
					splitLines.push_back(commentText.substr(start, maxLineLength));
					start += maxLineLength;
				}
			}
			//Now add the indentation to each split line and add a # character at the start of each line to make it a comment
			std::string result;
			for (auto &splitLine : splitLines)
			{
				result += indent + "# " + splitLine + "\n";
			}
			return result;
		}

		inline std::string splitLongKeyLine(const std::string &line, size_t maxLineLength)
		{
			size_t indentLength = 0;
			while (indentLength < line.length() && std::isspace(line[indentLength]))
			{
				indentLength++;
			}
			std::string indent(indentLength, ' ');
			//Strip off the indent at the start of the line to get the actual text, we will add the indent back in when we create the split lines
			std::string text = line.substr(indentLength);
			std::vector<std::string> splitLines;
			size_t start = 0;
			std::string extraIndent = "";
			while (start < text.length())
			{
				size_t end = start + maxLineLength - 1; // -1 to account for the \ character we will add at the end of the line to indicate a line continuation
				if (end >= text.length())
				{
					splitLines.push_back(extraIndent + text.substr(start));
					break;
				}
				//Split on the following characters
				std::string splitChars{",= +-*/ "}; // Characters we can split on if we need to break a long line
				size_t lastSplitChar = text.find_last_of(splitChars, end);
				if (lastSplitChar != std::string::npos && lastSplitChar > start)
				{
					splitLines.push_back(extraIndent + text.substr(start, lastSplitChar - start) + "\\");
					extraIndent = "  "; // Add extra indentation for subsequent lines after the first split line
					//Remember that we want to *keep* the split character unless it is whitespace
					start = lastSplitChar;
					if (std::isspace(text[start]))
					{
						start++;
					}
				}
				else
				{
					splitLines.push_back(text.substr(start, maxLineLength) + "\\");
					start += maxLineLength;
				}
			}
			std::string result;
			for (auto &splitLine : splitLines)
			{
				result += indent + splitLine + "\n";
			}
			return result;
		}

		inline std::string fixLongLines(const std::string &str, size_t maxLineLength)
		{
			std::vector<std::string> lines = naiveSplitLines(str);
			std::string result;
			for (const auto &line : lines)
			{
				if (line.length() <= maxLineLength)
				{
					result += line + "\n";
				}
				else
				{
					//Check if this is a comment line or a key-value line, we will split them differently
					//If the first non-whitespace character is a #, then it is a comment line
					size_t firstNonWhitespace = line.find_first_not_of(" \t");
					if (firstNonWhitespace != std::string::npos && line[firstNonWhitespace] == '#')
					{
						result += splitCommentLine(line, maxLineLength);
					} else
					{
						result += splitLongKeyLine(line, maxLineLength);
					}
				}
			}
			return result;
		}

		inline std::string toEPOCHString(const KVS &kvs, size_t maxLineLength = 0)
		{
			if (kvs.root.type != KVS::Type::Ordered && kvs.root.type != KVS::Type::Map)
				throw std::runtime_error("Root element must be an ordered collection or a map to convert to EPOCH format");

			std::string result;

			if (kvs.root.type == KVS::Type::Map)
			{
				result = mapToEPOCH(std::get<std::shared_ptr<KVS::Map>>(kvs.root.value), 0);
			}
			else if (kvs.root.type == KVS::Type::Ordered)
			{
				result = orderedToEPOCH(std::get<std::shared_ptr<KVS::Ordered>>(kvs.root.value), 0);
			}
			//A comment after the last line of the deck is written at the end
			if (kvs.root.hasMetadata("endComment"))
			{
				//The blank line after the last top level block is not wanted before the comment
				if (result.size() >= 2 && result.compare(result.size() - 2, 2, "\n\n") == 0) result.pop_back();
				result += commentToEPOCH("", kvs.root.getMetadata("endComment"));
			}
			if (maxLineLength > 0)
			{
				result = fixLongLines(result, maxLineLength);
			}
			return result;
		}

		inline void toEPOCHFile(const KVS &kvs, const std::string &filename)
		{
			std::ofstream file(filename);
			if (!file.is_open())
			{
				throw std::runtime_error("Could not open file for writing: " + filename);
			}
			file << toEPOCHString(kvs);
			file.close();
		}

		inline KVS fromEPOCHString(const std::string &epochString)
		{
			std::vector<std::pair<std::string, std::string>> splitResult = splitLines(epochString);
			std::vector<std::string> lines;
			for (const auto &linePair : splitResult)
			{
				lines.push_back(linePair.first);
			}
			size_t index = 0;
			KVS kvs{orderedFromEPOCH(splitResult, "", index)};
			//A comment after the last line of the deck is kept in the root
			if (!splitResult.empty() && splitResult.back().first.empty()) kvs.root.setMetadata("endComment", splitResult.back().second);
			return kvs;
		}

		inline KVS fromEPOCHFile(const std::string &filename)
		{
			std::ifstream file(filename);
			if (!file.is_open())
			{
				throw std::runtime_error("Could not open file for reading: " + filename);
			}
			std::stringstream buffer;
			buffer << file.rdbuf();
			return fromEPOCHString(buffer.str());
		}
	}
}
#endif // KVSEPOCH_H

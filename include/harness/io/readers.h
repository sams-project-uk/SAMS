/*
 *    Copyright 2025 SAMS Team
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#ifndef IO_READERS_H
#define IO_READERS_H

#include "readerSimple.h"

#ifdef USE_HDF5
#include "readerHDF5.h"
#define HDF5_READER_DEF(param, T_reader) READER_DEF(param, T_reader)
#else
#define HDF5_READER_DEF(param, T_reader)
#endif

#ifdef USE_ADIOS2
#include "readerADIOS2.h"
#define ADIOS2_READER_DEF(param, T_reader) READER_DEF(param, T_reader)
#else
#define ADIOS2_READER_DEF(param, T_reader)
#endif

//The readers that are available. READER_DEF(param, T_reader) must be defined where this is used
#define READER_TYPES(param) \
	READER_DEF(param, SAMS::IO::simpleReader) \
	HDF5_READER_DEF(param, SAMS::IO::HDF5Reader) \
	ADIOS2_READER_DEF(param, SAMS::IO::ADIOS2Reader)

#endif

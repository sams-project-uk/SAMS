#ifndef IO_WRITERS_H
#define IO_WRITERS_H

#include "writerSimple.h"
#if defined(USE_HDF5)
#include "writerHDF5.h"
#endif
#if defined(USE_ADIOS2)
#include "writerADIOS2.h"
#endif

//The writers that are available. Each entry is WRITER_DEF(param, T_writer), where param is passed through from the caller
#if defined(USE_HDF5)
#define HDF5_WRITER_DEF(param, T_writer) WRITER_DEF(param, T_writer)
#else
#define HDF5_WRITER_DEF(param, T_writer)
#endif
#if defined(USE_ADIOS2)
#define ADIOS2_WRITER_DEF(param, T_writer) WRITER_DEF(param, T_writer)
#else
#define ADIOS2_WRITER_DEF(param, T_writer)
#endif
#define WRITER_TYPES(param) \
	WRITER_DEF(param, SAMS::IO::simpleFile) \
	HDF5_WRITER_DEF(param, SAMS::IO::HDF5File) \
	ADIOS2_WRITER_DEF(param, SAMS::IO::ADIOS2File)

#endif

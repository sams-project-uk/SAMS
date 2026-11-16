#define WRITER_DEF(eos, T_writer) \
	template void HLLE::HLLE3D<eos>::registerDumpMeshes<T_writer>(SAMS::IO::writer<T_writer> &, SAMS::IO::registryWriter<T_writer> &, const SAMS::outputLevel &); \
	template void HLLE::HLLE3D<eos>::registerDumpVariables<T_writer>(SAMS::IO::writer<T_writer> &, SAMS::IO::registryWriter<T_writer> &, const SAMS::outputLevel &, HLLE::HLLE3D<eos>::simulationData &);
#define EOS_DEF(eos) WRITER_TYPES(eos)
EOS_DENSITY_ENERGY
#undef EOS_DEF
#undef WRITER_DEF

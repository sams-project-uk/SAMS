#define WRITER_DEF(eos, T_writer) \
	template void KT::KT3D<eos>::registerDumpMeshes<T_writer>(SAMS::IO::writer<T_writer> &, SAMS::IO::registryWriter<T_writer> &, const SAMS::outputLevel &); \
	template void KT::KT3D<eos>::registerDumpVariables<T_writer>(SAMS::IO::writer<T_writer> &, SAMS::IO::registryWriter<T_writer> &, const SAMS::outputLevel &, KT::KT3D<eos>::simulationData &);
#define EOS_DEF(eos) WRITER_TYPES(eos)
EOS_DENSITY_ENERGY
#undef EOS_DEF
#undef WRITER_DEF

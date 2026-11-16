#define WRITER_DEF(eos, T_writer) \
	template void LARE::LARE3D<eos>::registerDumpMeshes<T_writer>(SAMS::IO::writer<T_writer> &, SAMS::IO::registryWriter<T_writer> &, const SAMS::outputLevel &); \
	template void LARE::LARE3D<eos>::registerDumpVariables<T_writer>(SAMS::IO::writer<T_writer> &, SAMS::IO::registryWriter<T_writer> &, const SAMS::outputLevel &, LARE::LARE3D<eos>::simulationData &); \
	template void LARE::LARE3D<eos>::writeDumpMeshes<T_writer>(SAMS::IO::writer<T_writer> &, SAMS::IO::registryWriter<T_writer> &, const SAMS::outputLevel &); \
	template void LARE::LARE3D<eos>::writeDumpVariables<T_writer>(SAMS::IO::writer<T_writer> &, SAMS::IO::registryWriter<T_writer> &, const SAMS::outputLevel &);
#define EOS_DEF(eos) WRITER_TYPES(eos)
EOS_DENSITY_ENERGY
#undef EOS_DEF
#undef WRITER_DEF

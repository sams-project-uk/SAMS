#define READER_DEF(eos, T_reader) \
	template void KT::KT3D<eos>::restart<T_reader>(SAMS::IO::reader<T_reader> &, SAMS::IO::registryReader<T_reader> &);
#define EOS_DEF(eos) READER_TYPES(eos)
EOS_DENSITY_ENERGY
#undef EOS_DEF
#undef READER_DEF

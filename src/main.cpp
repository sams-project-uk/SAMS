#define PRINT_PARALLELIZATION_INFO

#include "pp/range.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include "harness.h"
#include "shared_data.h"
#include "HLLE/hlle.h"
#include "KT/kt.h"
#ifdef USE_KOKKOS
//#include "LAREKokkos/shared_data.h"
#endif
#include "timer.h"
#include "axisRegistry.h"
#include "variableRegistry.h"
#include "mpiManager.h"
#include "welcome.h"
#include "commandLine.h"

#include "tags.h"
#include "braginskii.h"
#include "isotropic.h"
#include "mg.h"
#ifdef USE_PETSC
#include "petscMatrix.h"
#include "petscNoMatrix.h"
#endif
#include "TCTest.h"

//Example initial conditions
#include "SodShockTube.h"
#include "BrioAndWu.h"
#include "MHDRotor.h"
#include "OrszagTang.h"
#include "OrszagTang3D.h"
#include "EmeryWindTunnel.h"
#include "KarmanVortex.h"
#include "InitialConditions/HLLEOrszagTang/HLLEOrszagTang.h"
#include "InitialConditions/SoundWave/SoundWave.h"
#include "DeckInitialConditions.h"
#include "HLLEDeckInitialConditions.h"
#include "deckHelpers/deckServices.h"

#include "builtInBoundaryConditions.h"

#include "Operations/OutputControl/OutputControl.h"
#include "Operations/RollingRestart/RollingRestart.h"
#include "Operations/BookendRestart/BookendRestart.h"
#include "Operations/WalltimeTerminate/WalltimeTerminate.h"
#include "Operations/SimulationTimeTerminate/SimulationTimeTerminate.h"
#include "Operations/DerivedQuantities/DerivedQuantities.h"

#include "runner.h"
#include "lareic.h"

//The packages that the executable can run
using samsRunner = SAMS::runner<
    LARE::LARE3D<>, //The main LARE3D solver
    HLLE::HLLE3D<>, //HLLE MHD solver
    KT::KT3D<>, //Kurganov-Tadmor MHD solver
    LARE::LARE3DInitialConditions<>,// Classic LARE style initial conditions
    examples::SodShockTube<>, //Sod shock tube initial conditions
    examples::BrioAndWu<>, //Brio and Wu initial conditions
    examples::MHDRotor<>, //MHD rotor initial conditions
    examples::OrszagTang<>, //Orszag-Tang initial conditions
    examples::OrszagTang3D<>, //Orszag-Tang 3D initial conditions
    examples::EmeryWindTunnel<>, //Emery wind tunnel initial conditions
    examples::KarmanVortex<>, //Karman vortex initial conditions
    examples::HLLEOrszagTang<>, //Orszag-Tang initial conditions for the HLLE solver
    examples::SoundWave<>, //Sound wave pulse, as the SoundWave deck
    LARE::LAREDeckInitialConditions<>, //Initial conditions read from the deck for LARE3D
    conservedVariable::deckInitialConditions, //Initial conditions read from the deck for HLLE3D and KT3D
    SAMS::deckServices, //The host data of the deck, shared by the packages that use the deck
    SAMS::outputControl, //Schedules for writing collections of outputs
    SAMS::rollingRestart, //Restart dumps cycling through a fixed number of files
    SAMS::bookendRestart, //Restart dumps at the start and the end of the run
    SAMS::walltimeTerminate, //Ends the run once a wall time has passed
    SAMS::simulationTimeTerminate, //Ends the run once the simulation time reaches the end time
    SAMS::derivedQuantities<>, //Example of writing quantities derived from the variables of LARE3D
    SAMS::TCTest<SAMS::braginskii<>>, //Thermal conduction test initial conditions for electrons and ions
    SAMS::parabolic::mg<SAMS::braginskii<>>, //Multigrid solver for Braginskii electron and ion conduction
		SAMS::parabolic::mg<SAMS::isotropic<>> //Multigrid solver for isotropic conduction
#ifdef USE_PETSC
    ,SAMS::parabolic::petscMatrix<SAMS::braginskii<>>, //PETSc solver with a matrix for Braginskii electron and ion conduction
    SAMS::parabolic::petscNoMatrix<SAMS::braginskii<>>, //PETSc solver without a matrix for Braginskii electron and ion conduction
    SAMS::parabolic::petscMatrix<SAMS::isotropic<>>, //PETSc solver with a matrix for isotropic electron and ion conduction
    SAMS::parabolic::petscNoMatrix<SAMS::isotropic<>> //PETSc solver without a matrix for isotropic electron and ion conduction
#endif
#ifdef USE_KOKKOS
    //,LARE::LARE3DKokkos //LARE3D written directly in Kokkos, to compare with the portable wrapper
#endif
>;

/**
 * Flags that run a set of packages with no deck, as --packages=list does. A new set is a new line
 */
const std::vector<std::pair<std::string, std::vector<std::string_view>>> packageSets = {
    {"lare", {LARE::LARE3D<>::name, LARE::LARE3DInitialConditions<>::name}},
};

/**
 * What the command line asks for
 */
struct runOptions
{
    std::string runDir, deckFile, packageList, restartDump, error;
    //--help, --version and --packages do not run
    bool information = false;
    //Whether a deck is read with --restart, replacing the keys of the deck stored in the dump
    bool restartOverride = false;
};

/**
 * Register the command line options and parse the command line. The run directory is the single positional argument or --rundir
 * @return What the command line asks for. runDir is empty if nothing runs
 */
runOptions parseCommandLine(int argc, char **argv, SAMS::commandLine &cli)
{
    using argumentType = SAMS::commandLine::argumentType;
    cli.registerOption("rundir", "The directory that holds the input deck and receives the output", argumentType::required);
    cli.registerOption("deckfile", "The name of the input deck in the run directory, in place of input.deck", argumentType::required);
    cli.registerOption("packages", "Print the packages, core in green and default in yellow. --packages=list runs the comma separated packages with no deck", argumentType::optional);
    cli.registerOption("restart", "Restart from a restart dump in the run directory, with the deck stored in it", argumentType::required);
    cli.registerOption("allow-restart-override", "With --restart, the keys of a deck in the run directory replace those of the deck stored in the dump", argumentType::none);
    cli.registerOption("restart-deck-only", "With --restart, a deck in the run directory is ignored and only the deck stored in the dump is used", argumentType::none);
    for (const auto &[flag, names] : packageSets) {
        std::string description = "Run";
        for (std::string_view name : names) description += " " + std::string(name);
        cli.registerOption(flag, description + " with no deck, as --packages=list does", argumentType::none);
    }
    cli.registerOption("help", "Print the usage", argumentType::none);
    cli.registerOption("version", "Print the version", argumentType::none);
    runOptions options;
    const bool parsed = cli.parse(argc, argv);
    //The packages that --packages=list and the package set flags ask for
    std::string packageList = cli.has("packages") ? cli.get("packages") : "";
    for (const auto &[flag, names] : packageSets) {
        if (!cli.flagSet(flag)) continue;
        for (std::string_view name : names) packageList += (packageList.empty() ? "" : ",") + std::string(name);
    }
    if (!parsed) {
        options.error = cli.error();
    } else if (cli.flagSet("help") || cli.flagSet("version") || (cli.flagSet("packages") && cli.get("packages").empty())) {
        options.information = true;
    } else if (!packageList.empty() && cli.has("deckfile")) {
        options.error = "--packages=list cannot be used with --deckfile";
    } else if (!packageList.empty() && cli.has("restart")) {
        options.error = "--packages=list cannot be used with --restart";
    } else if (cli.flagSet("allow-restart-override") && cli.flagSet("restart-deck-only")) {
        options.error = "--allow-restart-override cannot be used with --restart-deck-only";
    } else if (cli.has("rundir") && !cli.positionalArguments().empty()) {
        options.error = "The run directory is given both as an argument and with --rundir";
    } else if (cli.positionalArguments().size() > 1) {
        options.error = "Too many arguments";
    } else if (cli.has("deckfile") && (cli.get("deckfile").find_first_of("/\\") != std::string::npos || cli.get("deckfile") == "." || cli.get("deckfile") == "..")) {
        options.error = "Invalid --deckfile: " + cli.get("deckfile") + " is not a file name";
    } else {
        if (cli.has("rundir")) options.runDir = cli.get("rundir");
        else if (cli.positionalArguments().size() == 1) options.runDir = cli.positionalArguments()[0];
        if (cli.has("deckfile")) options.deckFile = cli.get("deckfile");
        if (!packageList.empty()) {
            options.packageList = packageList;
            if (!options.runDir.empty() && (std::filesystem::exists(options.runDir + "/input.deck") || std::filesystem::exists(options.runDir + "/input.json"))) {
                options.error = "--packages=list cannot be used with a deck in the run directory";
                options.runDir.clear();
            }
        }
        if (cli.has("restart")) {
            options.restartDump = cli.get("restart");
            const bool deckPresent = cli.has("deckfile") || (!options.runDir.empty() && (std::filesystem::exists(options.runDir + "/input.deck") || std::filesystem::exists(options.runDir + "/input.json")));
            options.restartOverride = deckPresent && cli.flagSet("allow-restart-override");
            if (deckPresent && !cli.flagSet("allow-restart-override") && !cli.flagSet("restart-deck-only")) {
                options.error = "--restart cannot be used with a deck unless --allow-restart-override or --restart-deck-only is given";
                options.runDir.clear();
            }
        }
    }
    return options;
}

/**
 * Finish the welcome message and print what the command line asks for when nothing runs
 */
void printInformation(samsRunner &runner, const SAMS::commandLine &cli, const runOptions &options)
{
    if (!options.error.empty()) SAMS::cerr << options.error << "\n";
    //Otherwise the runner finishes the welcome message once the packages are set up
    SAMS::finishWelcomeMessage();
    if (cli.flagSet("packages") && cli.get("packages").empty()) {
        SAMS::cout << "Packages:\n";
        runner.printPackages();
    }
    if (cli.flagSet("help") || (!options.information && options.error.empty())) {
        SAMS::cout << "Usage : sams {output_directory} [options]\n";
        SAMS::cout << "        sams --rundir {output_directory} [options]\n";
        SAMS::cout << "Options:\n";
        for (const auto &[name, description] : cli.getDescriptions()) SAMS::cout << "  --" << name << " : " << description << "\n";
    }
}

int main(int argc, char *argv[]){

    //Initialize MPI
    SAMS::MPI::initialize(argc, argv);
    //Initialize portable wrapper
    portableWrapper::initialize(argc, argv);

    //Print welcome message
    SAMS::printWelcomeMessage();

    //Create and initialize the runner
    samsRunner runner;
    runner.initialize(argc, argv);

    SAMS::commandLine cli;
    const runOptions options = parseCommandLine(argc, argv, cli);
    if (options.runDir.empty()) {
        printInformation(runner, cli, options);
    } else {
        //The directory holds the input deck and receives the output
        if (!options.restartDump.empty()) {
            //The deck is the one stored in the restart dump, with the keys of a deck in the run directory replacing its keys
            if (options.restartOverride) runner.readInputDeck(options.runDir, options.deckFile);
            else runner.setOutputDir(options.runDir);
            runner.setRestartDump(options.restartDump);
        }
        else if (options.packageList.empty()) runner.readInputDeck(options.runDir, options.deckFile);
        else runner.readPackageList(options.runDir, options.packageList);
        //Initialize the packages
        runner.initializePackages();
        //Run the packages until a package requests to stop
        runner.runPackages();
    }
    //Finish the packages
    runner.finalizePackages();
    //Finalize the runner
    runner.finalize();
    portableWrapper::finalize();
    SAMS::MPI::finalize();
}

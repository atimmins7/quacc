"""Custom Espresso calculator and template."""

from __future__ import annotations

import os
import re
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase.atoms import Atoms
from ase.calculators.espresso import EspressoProfile as EspressoProfile_
from ase.calculators.espresso import EspressoTemplate as EspressoTemplate_
from ase.calculators.genericfileio import GenericFileIOCalculator
from ase.units import Bohr
from ase.io import read, write
from ase.io.espresso import (
    Namelist,
    read_espresso_ph,
    write_espresso_ph,
    write_fortran_namelist,
)
from ase.io.espresso_namelist.keys import ALL_KEYS

from quacc import get_settings
from quacc.calculators.espresso.utils import (
    espresso_prepare_dir,
    get_pseudopotential_info,
    remove_conflicting_kpts_kspacing,
)
from quacc.utils.dicts import Remove, recursive_dict_merge, remove_dict_entries
from quacc.utils.files import load_yaml_calc, safe_decompress_dir


import qeschema
import copy

if TYPE_CHECKING:
    from typing import Any,Optional

LOGGER = getLogger(__name__)


class EspressoTemplate(EspressoTemplate_):
    """
    A wrapper around the ASE Espresso template that allows for the use of
    other binaries such as pw.x, ph.x, cp.x, etc.
    """

    def __init__(
        self,
        binary: str = "pw",
        test_run: bool = False,
        autorestart: bool = False,
        outdir: str | Path | None = None,
        store_only_final: bool = True,
    ) -> None:
        """
        Initialize the Espresso template.

        Parameters
        ----------
        binary
            The name of the espresso binary to use. This is used to set the
            input/output file names. By default we fall back to "pw".
        test_run
            If True, a test run is performed to check that the calculation
            input_data is correct or to generate some files/info if needed.
        autorestart
            If True, the calculation will automatically switch to 'restart'
            if this calculator performs more than one run. (ASE-relax/MD/NEB)
        outdir
            The directory that will be used as `outdir` in the input_data. If
            None, the directory will be set to the current working directory.

        Returns
        -------
        None
        """
        super().__init__()

        self.inputname = f"{binary}.in"
        self.outputname = f"{binary}.out"
        self.errorname = f"{binary}.err"
        self.binary = binary
        self._ase_known_binary = self.binary in ALL_KEYS
        self.test_run = test_run
        self.nruns = 0
        self.autorestart = autorestart
        self.outdir = outdir
        self.store_only_final = store_only_final

    def write_input(
        self,
        profile: EspressoProfile,
        directory: Path | str,
        atoms: Atoms,
        parameters: dict[str, Any],
        properties: Any,
    ) -> None:
        """
        The function that should be used instead of the one in ASE EspressoTemplate to
        write the input file. It calls a customly defined write function.

        Parameters
        ----------
        profile
            The profile to use.
        directory
            The directory in which to write the input file.
        atoms
            The atoms object to use.
        parameters
            The parameters to use.
        properties
            Special ASE properties

        Returns
        -------
        None
        """
        directory = Path(directory)
        self._output_handler(parameters, directory)
        parameters = self._sanity_checks(parameters)

        if self.outdir:
            safe_decompress_dir(self.outdir)

        if self.test_run:
            self._test_run(parameters, directory)

        if self.binary == "pw":
            if self.autorestart and self.nruns > 0:
                parameters["input_data"]["electrons"]["startingpot"] = "file"
                parameters["input_data"]["electrons"]["startingwfc"] = "file"
            write(
                directory / self.inputname,
                atoms,
                format="espresso-in",
                pseudo_dir=str(profile.pseudo_dir),
                properties=properties,
                **parameters,
            )
        elif self.binary in ["ph", "phcg"]:
            with Path(directory, self.inputname).open(mode="w") as fd:
                write_espresso_ph(fd, properties=properties, **parameters)
        else:
            with Path(directory, self.inputname).open(mode="w") as fd:
                write_fortran_namelist(
                    fd,
                    binary=self.binary if self._ase_known_binary else None,
                    properties=properties,
                    **parameters,
                )

    def execute(self, *args: Any, **kwargs: Any) -> None:
        super().execute(*args, **kwargs)
        self.nruns += 1

    @staticmethod
    def _search_keyword(parameters: dict[str, Any], key_to_search: str) -> str | None:
        """
        Function that searches for a keyword in the input_data.

        Parameters
        ----------
        parameters
            input_data, to search for the keyword

        Returns
        -------
        str
            The value of the keyword
        """
        input_data = parameters.get("input_data", {})

        for section in input_data:
            for key in input_data[section]:
                if key == key_to_search:
                    return input_data[section][key]
        return None

    @staticmethod
    def _test_run(parameters: dict[str, Any], directory: Path) -> None:
        """
        Almost all QE binaries will do a test run if a file named <prefix>.EXIT is
        present in the working directory. This function will create this file.

        Parameters
        ----------
        parameters
            input_data, which are needed to know the prefix
        directory
            The directory in which to write the EXIT file.

        Returns
        -------
        None
        """
        prefix = EspressoTemplate._search_keyword(parameters, "prefix") or "pwscf"

        Path(directory, f"{prefix}.EXIT").touch()

    def read_results(self, directory: os.PathLike) -> dict[str, Any]:
        """
        The function that should be used instead of the one in ASE EspressoTemplate to
        read the output file. It calls a customly defined read function. It also adds
        the "energy" key to the results dictionnary if it is not present. This is needed
        if the calculation is not made with pw.x.

        Parameters
        ----------
        directory
            The directory in which to read the output file.

        Returns
        -------
        dict
            The results dictionnary
        """
        all_results = []
        if self.binary == "pw":

            atoms_traj = read(Path(directory) / self.outputname, format="espresso-out",index=":")

            if(type(atoms_traj) != list):
                atoms_traj = [atoms_traj]

            if(self.store_only_final):
                atoms_traj = [atoms_traj[-1]]

            for atoms in atoms_traj:
                new_results = dict(atoms.calc.properties())
                new_results['lengths_and_angles'] = atoms.get_cell_lengths_and_angles()
                new_results['positions'] = atoms.get_positions()
                all_results.append(new_results)

            all_results = self.fix_corrupted_results(all_results, directory)

        elif self.binary in ["ph", "phcg"]:
            with Path(directory, self.outputname).open() as fd:
                new_results = read_espresso_ph(fd)
                all_results.append(new_results)
        elif self.binary == "dos":
            with Path(directory, "pwscf.dos").open() as fd:
                lines = fd.readlines()
                match = re.search(r"-?\d+\.?\d*", lines[0])
                fermi = float(match.group(0)) if match else None
                dos = np.loadtxt(lines[1:])
            new_results = {"dos_results": {"dos": dos, "fermi": fermi}}
            all_results.append(new_results)
        elif self.binary == "projwfc":
            with Path(directory, "pwscf.pdos_tot").open() as fd:
                lines = np.loadtxt(fd.readlines())
                energy = lines[1:, 0]
                dos = lines[1:, 1]
                pdos = lines[1:, 2]
            new_results = {"projwfc_results": {"energy": energy, "dos": dos, "pdos": pdos}}
            all_results.append(new_results)
        elif self.binary == "matdyn":
            fldos = Path(directory, "matdyn.dos")
            if fldos.exists():
                phonon_dos = np.loadtxt(fldos)
                new_results = results = {"matdyn_results": {"phonon_dos": phonon_dos}}
                all_results.append(new_results)

        for idx,result in enumerate(all_results):
            if "energy" not in result:
                all_results[idx]["energy"] = None

        return all_results

    def fix_corrupted_results(self,initial_results,directory: os.PathLike, debug=False):
        """
        Detect and fix a known Quantum ESPRESSO output bug where the first
        ionic step's atomic positions/cell are printed incorrectly to the
        text output file (pw.out) after a restart.

        Some QE runs print the line "Atomic positions from file used, from
        input discarded" in the output, which indicates that the positions
        and cell reported for step #0 in the text output do not correspond
        to the input structure actually used for the calculation. When this
        is detected, this function attempts to recover the correct step #0
        positions and cell from the accompanying `data-file-schema.xml` file
        (which is written independently of the text output and is not
        affected by this bug).

        Recovery logic:
            1. If the corrupting line is not found in the output file,
               `initial_results` is returned unchanged.
            2. If the line is found but a unique `data-file-schema.xml`
               cannot be located in `directory`, the corrupted step #0 data
               cannot be verified or fixed; it is left as-is (with a debug
               message noting the failure).
            3. If a unique XML file is found, its step #0 positions are
               compared against the (possibly wrong) text-output positions.
               If they differ, the XML values are considered authoritative
               and used to overwrite step #0's 'positions' and
               'lengths_and_angles' in the returned results.
            4. If the corruption is detected but cannot be fixed (i.e.
               `step0_fixable` is False), step #0's 'positions',
               'lengths_and_angles', and 'forces' are cleared to empty
               lists so that corrupted data is not silently submitted.

        Parameters
        ----------
        initial_results : list[dict]
            List of per-ionic-step result dictionaries (as produced by
            `read_results`), each expected to contain at least 'positions'
            and 'lengths_and_angles' keys.
        directory : os.PathLike
            Directory containing the QE output file and, if present, the
            `data-file-schema.xml` file used for the fix.
        debug : bool, optional
            If True, print diagnostic messages describing what was detected
            and/or fixed. Default is False.

        Returns
        -------
        list[dict]
            A deep copy of `initial_results`, with step #0 either corrected
            using the XML data, left unchanged (if uncorrupted), or cleared
            (if corrupted and unfixable).
        """

        final_results = copy.deepcopy(initial_results)

        step0_corrupted = False
        step0_fixable = False

        pw_document = qeschema.PwDocument()

        input_atoms_discarded = self._check_if_input_atoms_discarded(Path(directory) / self.outputname)
        if input_atoms_discarded:
            step0_corrupted = True
            xml_paths = self._find_files(directory, filename="data-file-schema.xml")
            if len(xml_paths) == 1:
                step0_fixable = True
            elif debug:
                print("Input atoms were discarded, but data-file-schema.xml cannot be "
                      "uniquely identified.")

        if step0_corrupted:
            if step0_fixable:
                pw_positions = [result['positions'] for result in initial_results]

                pw_document.read(xml_paths[0])
                xml_data = pw_document.to_dict()

                xml_steps = xml_data['qes:espresso']['step']
                if isinstance(xml_steps, dict):
                    xml_steps = [xml_steps]

                xml_position_dicts = [xml_step['atomic_structure']['atomic_positions'] for xml_step in xml_steps]
                xml_positions = [self._atoms_dict_to_array(d, to_angstrom=True) for d in xml_position_dicts]

                xml_cell_dicts = [self._create_cell_array(xml_step['atomic_structure']['cell'], to_angstrom=True)
                                  for xml_step in xml_steps]
                xml_cellpars = [self._cell_to_cellpar(c) for c in xml_cell_dicts]

                step0_positions_match = np.allclose(pw_positions[0], xml_positions[0])
                if not step0_positions_match:
                    if debug:
                        print("Updating Info for Step #0")
                    final_results[0]['positions'] = xml_positions[0]
                    final_results[0]['lengths_and_angles'] = xml_cellpars[0]

            else:
                if debug:
                    print("Step #0 could not be fixed. Removing corrupted data from DB submission")
                final_results[0]['positions'] = []
                final_results[0]['lengths_and_angles'] = []
                final_results[0]['forces'] = []

        return final_results

    def _find_files(self,directory: str, filename: str) -> list[str]:
        """
        Recursively search `directory` for files named `filename`.

        Args:
            directory: Root directory to search.
            filename: Exact file name to match (e.g. "config.json").

        Returns:
            List of matching file paths as strings.
        """
        root = Path(directory)
        return [str(p) for p in root.rglob(filename) if p.is_file()]

    def _atoms_dict_to_array(self,data: dict, to_angstrom: bool = False) -> np.ndarray:
        """
        Convert a QE XML-style atom dictionary into an (N, 3) numpy array of positions.

        Args:
            data: Dict with key 'atom', a list of entries each containing
                  '$' (the [x, y, z] position in Bohr).
            to_angstrom: If True, convert positions from Bohr to Angstrom.

        Returns:
            (N, 3) numpy array of positions.
        """
        positions = np.array([atom['$'] for atom in data['atom']])

        if to_angstrom:
            positions *= Bohr

        return positions

    def _cell_to_cellpar(self,cell: np.ndarray, degrees: bool = True) -> np.ndarray:
        """
        Convert a 3x3 cell matrix (rows = lattice vectors a1, a2, a3) into
        a flat 6-element array [a, b, c, alpha, beta, gamma].

        Args:
            cell: (3, 3) numpy array. Rows are the lattice vectors.
            degrees: If True, return angles in degrees. If False, radians.

        Returns:
            (6,) numpy array: [a, b, c, alpha, beta, gamma], where
            alpha = angle between b and c,
            beta  = angle between a and c,
            gamma = angle between a and b.
        """
        cell = np.asarray(cell, dtype=float)
        a1, a2, a3 = cell

        a = np.linalg.norm(a1)
        b = np.linalg.norm(a2)
        c = np.linalg.norm(a3)

        def angle(u, v):
            cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # guard against rounding error
            return np.arccos(cos_theta)

        alpha = angle(a2, a3)
        beta = angle(a1, a3)
        gamma = angle(a1, a2)

        if degrees:
            alpha, beta, gamma = np.degrees([alpha, beta, gamma])

        return np.array([a, b, c, alpha, beta, gamma])

    def _create_cell_array(self,cell_dict, to_angstrom: bool = False):
        cell_array = []
        for k, v in cell_dict.items():
            new_row = np.array(v)
            if (to_angstrom):
                new_row *= Bohr
            cell_array.append(new_row)
        cell_array = np.array(cell_array)
        return cell_array

    def _check_if_input_atoms_discarded(self,filepath: str) -> bool:
        """
        Search a text file for the line containing:
        "Atomic positions from file used, from input discarded"

        Args:
            filepath: Path to the text file to search.

        Returns:
            True if the string is found, False otherwise.
        """
        target = "Atomic positions from file used, from input discarded"

        with open(filepath, "r") as f:
            for line in f:
                if target in line:
                    return True

        return False

    def _output_handler(
        self, parameters: dict[str, Any], directory: Path | str
    ) -> dict[str, Any]:
        """
        Function that handles the various output of espresso binaries. It will force the
        output directory and other output files to be set or deleted if needed.

        It will also prevent the user from setting environment variables that change the
        output directories.

        Parameters
        ----------
        parameters
            User-supplied kwargs
        directory
            The `directory` kwarg from the calculator.

        Returns
        -------
        dict[str, Any]
            The merged kwargs
        """
        os.environ.pop("ESPRESSO_TMPDIR", None)
        os.environ.pop("ESPRESSO_FILDVSCF_DIR", None)
        os.environ.pop("ESPRESSO_FILDRHO_DIR", None)

        espresso_outdir = Path(self.outdir or directory).expanduser().resolve()
        outkeys = espresso_prepare_dir(espresso_outdir, self.binary)

        input_data = parameters.get("input_data", {})
        input_data = recursive_dict_merge(input_data, outkeys, verbose=True)

        parameters["input_data"] = input_data

        return parameters

    def _sanity_checks(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Function that performs sanity checks on the input_data. It is meant
        to catch common mistakes that are not caught by the espresso binaries.

        Parameters
        ----------
        parameters
            The parameters dictionary which is assumed to already be in
            the nested format.

        Returns
        -------
        dict
            The modified dictionary parameters.
        """
        input_data = parameters.get("input_data", {})

        if self.binary == "pw":
            system = input_data.get("system", {})

            occupations = system.get("occupations", "fixed")
            smearing = system.get("smearing", None)
            degauss = system.get("degauss", None)

            if occupations == "fixed" and (smearing is not None or degauss is not None):
                LOGGER.warning(
                    "The occupations are set to 'fixed' but smearing or degauss is also set. This will be ignored."
                )
                system["smearing"] = Remove
                system["degauss"] = Remove

            parameters["input_data"]["system"] = system

        elif self.binary in ["ph", "phcg"]:
            input_ph = input_data.get("inputph", {})
            qpts = parameters.get("qpts", (0, 0, 0))

            qplot = input_ph.get("qplot", False)
            lqdir = input_ph.get("lqdir", False)
            recover = input_ph.get("recover", False)
            ldisp = input_ph.get("ldisp", False)

            is_grid = input_ph.get("start_q") or input_ph.get("start_irr")
            # Temporary patch for https://gitlab.com/QEF/q-e/-/issues/644
            if qplot and lqdir and recover and is_grid:
                prefix = input_ph.get("prefix", "pwscf")
                outdir = input_ph.get("outdir", ".")

                Path(outdir, "_ph0", f"{prefix}.q_1").mkdir(parents=True, exist_ok=True)
            if not (ldisp or qplot):
                if np.array(qpts).shape == (1, 4):
                    LOGGER.warning(
                        "qpts is a 2D array despite ldisp and qplot being set to False. Converting to 1D array"
                    )
                    qpts = tuple(qpts[0])
                if lqdir and is_grid and qpts != (0, 0, 0):
                    LOGGER.warning(
                        "lqdir is set to True but ldisp and qplot are set to False. The band structure will still be computed at each step. Setting lqdir to False"
                    )
                    input_ph["lqdir"] = False

            parameters["input_data"]["inputph"] = input_ph
            parameters["qpts"] = qpts

        return remove_dict_entries(parameters, remove_trigger=Remove)


class Espresso(GenericFileIOCalculator):
    """
    A wrapper around the ASE Espresso calculator that adjusts input_data
    parameters and allows for the use of presets.
    Templates are used to set the binary and input/output file names.
    """

    def __init__(
        self,
        input_atoms: Atoms | None = None,
        preset: str | Path | None = None,
        template: EspressoTemplate | None = None,
        allowed_return_codes: list | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the Espresso calculator.

        Parameters
        ----------
        input_atoms
            The input Atoms object to be used for the calculation.
        preset
            A YAML file containing a list of parameters to use as a "preset"
            for the calculator. If `preset` has a .yml or .yaml file extension, the
            path to this file will be used directly. If `preset` is a string without
            an extension, the corresponding YAML file will be assumed to be in the
            `ESPRESSO_PRESET_DIR`. Any user-supplied calculator **kwargs will
            override any corresponding preset values.
        template
            ASE calculator templace which can be used to specify which espresso
            binary will be used in the calculation. This is taken care of by recipe
            in most cases.
        **kwargs
            Additional arguments to be passed to the Espresso calculator. Takes all valid
            ASE calculator arguments, such as `input_data` and `kpts`. Refer to
            [ase.calculators.espresso.Espresso][] for details. Note that the full input
            must be described; use `{"system":{"ecutwfc": 60}}` and not the `{"ecutwfc": 60}`
            short-hand.

        Returns
        -------
        None
        """
        self.input_atoms = input_atoms or Atoms()
        self.preset = preset
        self.kwargs = kwargs
        self.user_calc_params = {}
        self._settings = get_settings()
        template = template or EspressoTemplate("pw")
        self._binary = template.binary
        full_path = Path(
            self._settings.ESPRESSO_BIN_DIR,
            self._settings.ESPRESSO_BINARIES[self._binary],
        )
        self._bin_path = str(full_path)

        if template._ase_known_binary:
            self._cleanup_params()
        else:
            LOGGER.warning(
                f"The binary you requested, `{self._binary}`, is not supported by ASE. This means that presets and usual checks will not be carried out, your `input_data` must be provided in nested format."
            )

            self.kwargs["input_data"] = Namelist(self.kwargs.get("input_data"))
            self.user_calc_params = self.kwargs

        self._pseudo_path = (
            self.user_calc_params.get("input_data", {})
            .get("control", {})
            .get("pseudo_dir", str(self._settings.ESPRESSO_PSEUDO))
        )

        cmd_prefix = os.environ.get(
            "PARSL_MPI_PREFIX", self._settings.ESPRESSO_PARALLEL_CMD[0]
        )
        cmd_suffix = self._settings.ESPRESSO_PARALLEL_CMD[1]

        profile = EspressoProfile(
            f"{cmd_prefix} {self._bin_path} {cmd_suffix}",
            self._pseudo_path,
            allowed_return_codes=allowed_return_codes
        )

        super().__init__(
            template=template,
            profile=profile,
            directory=".",
            parameters=self.user_calc_params,
        )

    def _cleanup_params(self) -> None:
        """
        Function that handles the kwargs. It will merge the user-supplied kwargs with
        the preset values, using the former as priority.

        Returns
        -------
        None
        """
        if self.kwargs.get("directory"):
            raise NotImplementedError("quacc does not support the directory argument.")

        self.kwargs["input_data"] = Namelist(self.kwargs.get("input_data"))
        self.kwargs["input_data"].to_nested(binary=self._binary, **self.kwargs)

        if self.preset:
            preset_path = (
                self.preset
                if Path(self.preset).suffix in (".yaml", ".yml")
                else self._settings.ESPRESSO_PRESET_DIR / f"{self.preset}.yaml"
            )
            calc_preset = load_yaml_calc(preset_path)
            calc_preset["input_data"] = Namelist(calc_preset.get("input_data"))
            calc_preset["input_data"].to_nested(binary=self._binary, **calc_preset)
            if "pseudopotentials" in calc_preset:
                ecutwfc, ecutrho, pseudopotentials = get_pseudopotential_info(
                    calc_preset["pseudopotentials"], self.input_atoms
                )
                calc_preset.pop("pseudopotentials", None)
                calc_preset = remove_conflicting_kpts_kspacing(calc_preset, self.kwargs)
                self.user_calc_params = recursive_dict_merge(
                    calc_preset,
                    {
                        "input_data": {
                            "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho}
                        },
                        "pseudopotentials": pseudopotentials,
                    },
                    self.kwargs,
                )
            else:
                self.user_calc_params = recursive_dict_merge(calc_preset, self.kwargs)
        else:
            self.user_calc_params = self.kwargs

        if self.user_calc_params.get("kpts") is not None and self.user_calc_params.get(
            "kspacing"
        ):
            raise ValueError("Cannot specify both kpts and kspacing.")

class EspressoProfile(EspressoProfile_):

    def __init__(self,command, pseudo_dir,
                 allowed_return_codes:list|None = None):

        super().__init__(command, pseudo_dir)
        # not Path object to avoid problems in remote calculations from Windows
        self.allowed_return_codes = allowed_return_codes
        self.run = self.new_run

        if(self.allowed_return_codes is None):
            self.allowed_return_codes = [0]
        else:
            self.allowed_return_codes.append(0)

    def new_run(self, directory: Path, inputfile: Optional[str],
            outputfile: str, errorfile: Optional[str] = None,
            append: bool = False
    ) -> None:
        """
        Run the command in the given directory. Redefined from
        function in ASE generifileio to replace subprocess.check_call()
        with subprocess.run(), because subprocess.check_call() raises an exception
        for non-zero exit-codes, which is overly-restrictive for relaxation
        jobs. The list of self.allowed_return_codes allows for easy definition of
        QE conditions that should not be considered as failures.

        Parameters
        ----------
        directory : pathlib.Path
            The directory to run the command in.
        inputfile : Optional[str]
            The name of the input file.
        outputfile : str
            The name of the output file.
        errorfile: Optional[str]
            the stderror file
        append: bool
            if True then use append mode
        """

        import os
        from subprocess import run as subprocess_run
        from contextlib import ExitStack

        argv_command = self.get_command(inputfile)
        mode = 'wb' if not append else 'ab'

        with ExitStack() as stack:
            output_path = directory / outputfile
            fd_out = stack.enter_context(open(output_path, mode))
            if errorfile is not None:
                error_path = directory / errorfile
                fd_err = stack.enter_context(open(error_path, mode))
            else:
                fd_err = None
            completed_result = subprocess_run(
                argv_command,
                cwd=directory,
                stdout=fd_out,
                stderr=fd_err,
                env=os.environ,
            )
            return_code = abs(completed_result.returncode)
            if(return_code not in self.allowed_return_codes):
                completed_result.check_returncode()
#%%

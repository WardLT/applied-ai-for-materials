"""Utilities for conformer optimization"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Set

from ase.calculators.psi4 import Psi4
from ase.constraints import FixInternals
from ase.optimize import BFGS
from ase.io.xyz import read_xyz
from ase import Atoms
from openbabel import OBMolBondIter
from io import StringIO
import networkx as nx
import torchani
import pybel
import os

calc = torchani.models.ANI2x().ase()


def get_initial_structure(smiles: str) -> Tuple[Atoms, Dict[int, Set[int]]]:
    """Generate an initial guess for a molecular structure
    
    Args:
        smiles: SMILES string
    Returns: 
        An ASE atoms object, bond graph
    """
    
    # Make the 3D structure
    mol = pybel.readstring("smi", smiles)
    mol.make3D()
    
    # Convert it to ASE
    atoms = next(read_xyz(StringIO(mol.write('xyz')), slice(None)))
    
    # Get the bonding graph
    g = nx.Graph()
    g.add_nodes_from(range(len(mol.atoms)))
    for bond in OBMolBondIter(mol.OBMol):
        g.add_edge(bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1, data={"rotor": bond.IsRotor()})
    return atoms, g

def relax_structure(atoms: Atoms) -> float:
    """Relax and return the energy of the ground state
    
    Args:
        atoms
    """
    
    atoms.set_calculator(calc)
    
    dyn = BFGS(atoms, logfile=os.devnull)
    dyn.run(fmax=1e-3)
    
    return atoms.get_potential_energy()


@dataclass()
class DihedralInfo:
    """Describes a dihedral angle within a molecule"""
    
    chain: Tuple[int, int, int, int] = None
    """Atoms that form the dihedral. ASE rotates the last atom when setting a dihedral angle"""
    group: Set[int] = None
    """List of atoms that should rotate along with this dihedral"""

    
def get_dihedral_info(graph: nx.Graph, bond: Tuple[int, int], backbone_atoms: Set[int]) -> DihedralInfo:
    """For a rotatable bond in a model, get the atoms which define the dihedral angle
    and the atoms that should rotate along with the right half of the molecule
    
    Args:
        graph: Bond graph of the molecule
        bond: Left and right indicies of the bond, respectively
        backbone_atoms: List of atoms defined as part of the backbone
    Returns:
        - Atom indices defining the dihedral. Last atom is the one that will be moved 
          by ase's "set_dihedral" function
        - List of atoms being rotated along with set_dihedral
    """
    
    # Pick the atoms to use in the dihedral, starting with the left
    points = list(bond)
    choices = set(graph[bond[0]]).difference(bond)
    bb_choices = choices.intersection(backbone_atoms)
    if len(bb_choices) > 0:  # Pick a backbone if available
        choices = bb_choices
    points.insert(0, min(choices))
    
    # Then the right
    choices = set(graph[bond[1]]).difference(bond)
    bb_choices = choices.intersection(backbone_atoms)
    if len(bb_choices) > 0:  # Pick a backbone if available
        choices = bb_choices
    points.append(min(choices))
    
    # Get the points that will rotate along with the bond
    h = graph.copy()
    h.remove_edge(*bond)
    a, b = nx.connected_components(h)
    if bond[1] in a:
        return DihedralInfo(chain=points, group=a)
    else:
        return DihedralInfo(chain=points, group=b)

    
def set_dihedrals_and_relax(atoms: Atoms, dihedrals: List[Tuple[float, DihedralInfo]]) -> float:
    """Set the dihedral angles and compute the energy of the system
    
    Args:
        atoms: Molecule to ajdust
        dihedrals: List of dihedrals to set to a certain angle
    Returns:
        Energy of the system
    """
    
    # Copy input so that we can loop over it twice (i.e., avoiding problems around zip being a generator)
    dihedrals = list(dihedrals)
    
    # Set the dihedral angles to desired settings
    for di in dihedrals:
        atoms.set_dihedral(*di[1].chain, di[0], indices=di[1].group)
        
    # Define the constraints
    dih_cnsts = [(di[0], di[1].chain) for di in dihedrals]
    atoms.set_constraint()
    atoms.set_constraint(FixInternals(dihedrals_deg=dih_cnsts))
    
    return relax_structure(atoms)

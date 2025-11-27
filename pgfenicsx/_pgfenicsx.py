# Moritz Feuerle, 2025

__all__ = ['DirichletBC', 'dirichletbc', 'collect_dirichletbcs', 'setup_boundary_meshtags', 'concatenate_meshtags', 'assemble_system']


import numpy as np
import scipy.sparse as sparse
from numbers import Number
from dolfinx import default_scalar_type
from dolfinx.fem.function import FunctionSpace
from dolfinx import mesh, fem
import ufl

from typing import TypeAlias
from collections.abc import Iterable, Callable
BoundaryFunctionType: TypeAlias = fem.Function | fem.Constant | fem.Expression | Number | Callable


def as_numpy_vector(x, dtype=None):
    """Ensure that x is a numpy array of the given dtype and shape (n,)."""
    return np.asarray(x, dtype=dtype).reshape((-1,))

def setup_boundary_meshtags(msh: mesh.Mesh, boundary_parts: Iterable[tuple[int, Callable[[np.ndarray], np.ndarray[bool]] | np.ndarray[int] | None]]) -> mesh.MeshTags:
    
    tags = [part[0] for part in boundary_parts]
    tags_unique = np.unique(tags)
    tags_occurrences = [np.where(tags==tag)[0] for tag in tags_unique]
    
    tags_leftovers = []
    mesh_tags = np.empty((2, 0), np.int32)
    
    for i in range(len(tags_unique)):
        tag = tags_unique[i]
        facets = np.empty((1, 0), np.int32)
        for j in tags_occurrences[i]:
            boundary = boundary_parts[j]
            if len(boundary) == 1 or boundary[1] is None:
                tags_leftovers.append(tag)
                continue
            if isinstance(boundary[1], np.ndarray):
                facets = np.append(facets, boundary[1])
            else:
                facets = np.append(facets, mesh.locate_entities_boundary(msh, dim=msh.topology.dim-1, marker=boundary[1]))
        facets = np.unique(facets)
        mesh_tags = np.append(mesh_tags, np.vstack((facets, np.full_like(facets, tag))),axis=1)
        
    if len(tags_leftovers) > 0:
        tags_leftovers = np.unique(tags_leftovers)
        all_bdry_facets = mesh.exterior_facet_indices(msh.topology)
        facets = np.setdiff1d(all_bdry_facets,mesh_tags[0])
        for tag in tags_leftovers:
            mesh_tags = np.append(mesh_tags,np.vstack((facets,  np.full_like(facets, tag))),axis=1)
            
    mesh_tags = mesh_tags[:, np.argsort(mesh_tags[0])]
    return mesh.meshtags(msh, msh.topology.dim-1, mesh_tags[0], mesh_tags[1])


def concatenate_meshtags(mesh: mesh.Mesh, meshtags: Iterable[mesh.MeshTags]) -> mesh.MeshTags:
    values = np.empty((0,), dtype=np.int32)
    indices = np.empty((0,), dtype=np.int32)
    for mt in meshtags:
        values = np.append(values, mt.values)
        indices = np.append(indices, mt.indices)
    idx = np.argsort(indices)
    return mesh.meshtags(mesh, mesh.topology.dim-1, indices[idx],  values[idx])

class DirichletBC:
    r"""Class representing a Dirichlet boundary condition."""
    
    def __init__(self, function_space: FunctionSpace, u: BoundaryFunctionType | np.ndarray[float], dofs: np.ndarray[int]):
        r"""Create a Dirichlet boundary condition restricting the ``dofs`` of ``function_space`` with the the values given by ``u``.
        
        Args:
            function_space:
                The function space the boundary condition is defined on.
            u:
                Object defining the dirichlet values at the given dofs. Can be given either the same way as in :func:`dirichletbc`, or as a discrete array directly specifying the values at each dof, i.e. an array of a) same length as ``dofs``, i.e. ``u[i]`` containes the dirichlet value of dof ``dofs[i]``, or b) the length of all dofs of the function space, i.e. ``u[dofs[i]]`` containes the dirichlet value of dof ``dofs[i]``.
            dofs:
                The dofs of the function space on which the Dirichlet condition is imposed.
        """
        
        if isinstance(u, fem.Function):
            if not u.function_space == function_space:
                raise ValueError('Function u must be defined on the same function space as the DirichletBC')
            values = u.x.array[dofs]
        elif isinstance(u, fem.Constant):
            values = np.full(dofs.shape, u.value, dtype=default_scalar_type)
        elif isinstance(u, Number):
            values = np.full(dofs.shape, default_scalar_type(u))
        else:
            try:
                values = as_numpy_vector(u, default_scalar_type)
            except:
                try:
                    u_func = fem.Function(function_space)
                    u_func.interpolate(u)
                    values = u_func.x.array[dofs]
                except:
                    raise TypeError('dont know what to do')
        
        dofs   = as_numpy_vector(dofs, np.int32)
        values = as_numpy_vector(values, default_scalar_type)
        
        def unique(x): seen = set(); return not any(i in seen or seen.add(i) for i in x)
        if not unique(dofs):
            raise ValueError('duplicate dofs in DirichletBC')
        
        if len(values) == function_space.dofmap.index_map.size_global:
            values = values[dofs]
        elif len(values) != len(dofs):
            raise ValueError('Number of values does not match the number of dirichlet dofs or function space dofs')
        
        idx = np.argsort(dofs)
        
        self.fixed_dofs: np.ndarray[int] = dofs[idx]
        """Fixed dofs corresponding to the dirichlet values (unique, i.e. no duplicates, and sorted in ascending order)."""
        self.values: np.ndarray[float] = values[idx]
        """Discrete dirichlet values, i.e. ``values[i]`` is the dirichlet value at dof ``fixed_dofs[i]``."""
        self.function_space: FunctionSpace = function_space
        """Function space the boundary condition is defined on."""
        self._free_dofs = None
        self.ndofs: int = self.function_space.dofmap.index_map.size_global
        """Total number of dofs of the function space (free+fixed)."""
        
    @property
    def free_dofs(self) -> np.ndarray[int]:
        """Free dofs where no dirichlet condition is applied (unique, i.e. no duplicates, and sorted in ascending order)."""
        if self._free_dofs is None:
            free_dofs = np.ones(self.ndofs, dtype=bool)
            free_dofs[self.fixed_dofs] = False
            self._free_dofs = np.sort(np.where(free_dofs)[0])
        return self._free_dofs

      
def dirichletbc(function_space: FunctionSpace, u: BoundaryFunctionType, facets: np.ndarray[int] | None = None) -> DirichletBC:
    r"""Create a Dirichlet boundary condition for ``function_space`` on the boundary part ``facets`` with the boundary values given by ``u``.
    
    This function operates on **facets not dofs**, unlike :func:`dolfinx.fem.dirichletbc`. This decision was made conserning Petrov-Galerkin formulations: the dofs of the test and trial space may differ, while the facets belong to the mesh and are thus independend of the space (thus avoiding potential hard to debug errors by defining dirichlet condition for one space while using the facet dofs generated based on another space). 
    If one wants to create a dirichlet boundary condition based on dofs, one can always use :class:`DirichletBC` directly. 
    
    Args:
        function_space:
            The function space the boundary condition is defined on.
        u:
            The boundary value. Can be a number, :class:`dolfinx.fem.Constant`, :class:`dolfinx.fem.Function` or any object that can be interpolated into a :class:`dolfinx.fem.Function`.
        facets:
            The boundary facets where the Dirichlet condition is applied. If ``None``, all exterior facets are used.
    """
   
    if facets is None:
        facets = mesh.exterior_facet_indices(function_space.mesh.topology)
    
    facets = as_numpy_vector(facets, np.int32)
    bdry_dofs = fem.locate_dofs_topological(function_space, function_space.mesh.topology.dim-1, facets)
    
    return DirichletBC(function_space, u, bdry_dofs)




def collect_dirichletbcs(bcs: Iterable[DirichletBC], function_space: FunctionSpace | Iterable[FunctionSpace] | None = None, check_tol: float = 1e-14) -> DirichletBC | list[DirichletBC]:
    r"""Collect multiple Dirichlet boundary conditions given in ``bcs`` into a single Dirichlet boundary condition for each space given in ``function_space``.
    
    Args:
        bcs:
            Iterable of Dirichlet boundary conditions (possibly defined on different function spaces) to be collected into one Dirichlet boundary condition per function space.
            
        function_space:
            The function space(s) for which the Dirichlet boundary conditions are to be collected. If ``None``, all function spaces occuring in ``bcs`` are used, in the same order as they appear in ``bcs``.
            
        check_tol:  
            Tolerance for checking conflicting Dirichlet values at the same dof. If two Dirichlet boundary conditions specify different values at the same dof, a warning is issued and the value from the first Dirichlet boundary condition in the ``bcs`` is used. This option as no impact on the output and is just to inform about conflicting conditions. To turn off this check, set ``check_tol`` to ``np.inf``.
        
    Returns:
        One Dirichlet boundary condition per function space containing all Dirichlet boundary conditions for that space. If ``function_space`` is a single function space, a single Dirichlet boundary condition is returned; if it is an iterable of function spaces or ``None``, a list of Dirichlet boundary conditions is returned in the same order as the function spaces in ``function_space``.
    """
    
    if function_space is None:
        function_space = list()
        for bc in bcs:
            space = bc.function_space
            if space not in function_space:
                function_space.append(space)
    
    # if more than one function space, return one BC for each space  
    if isinstance(function_space, Iterable):
        return [collect_dirichletbcs(bcs, fs, check_tol) for fs in function_space]
    
    hits = np.where([bc.function_space == function_space for bc in bcs])[0]
    if len(hits) == 0:      # empty bc
        return DirichletBC(function_space, np.array([],dtype=default_scalar_type), np.array([],dtype=np.int32))
    elif len(hits) == 1:    # only one bc, nothing to collect
        return bcs[hits[0]]
    
    all_dofs = np.empty((1, 0), np.int32)
    all_values = np.empty((1, 0), default_scalar_type)
    
    for i in hits:
        dofs = bcs[i].fixed_dofs
        values = bcs[i].values
        
        idx_duplicates = np.isin(dofs, all_dofs, assume_unique=True)
        idx_new = np.logical_not(idx_duplicates)
        all_dofs = np.append(all_dofs, dofs[idx_new])
        all_values = np.append(all_values, values[idx_new])
        
        for dupl_dof, dupl_value in zip(dofs[idx_duplicates], values[idx_duplicates]):
            existing_value = all_values[np.where(all_dofs == dupl_dof)][0]
            if not np.isclose(existing_value, dupl_value, atol=check_tol):
                from warnings import warn
                warn(f'\nConflicting Dirichlet BC value at dof#{dupl_dof}: {dupl_value} vs {existing_value}; using {existing_value}')
                
    return DirichletBC(function_space, all_values, all_dofs)


def assemble_system(F: ufl.form.Form | tuple[ufl.form.Form, ufl.form.Form], bcs: Iterable[DirichletBC], include_bc: bool = True) -> tuple[sparse.csr_array, np.ndarray] | tuple[sparse.csr_array, np.ndarray, DirichletBC]:
    r"""
    Assemble the system matrix and right-hand side of a Petrov-Galerkin variational problem, applying the Dirichlet boundary conditions given in ``bcs``.
    
    Args:
        F:
            The variational form, either one form that can be split using ``A,l = ufl.system(F)`` or the tuple ``(A, l)`` of a bilinear form ``A`` and a linear form ``l``.
        bcs:
            Iterable of Dirichlet boundary conditions to be applied, can contain different Dirichlet boundary conditions for the trial and test space.
        include_bc:
            If ``True``, the returned system includes the Dirichlet boundary conditions, if ``False``, the system is reduced to the free dofs only and the Dirichlet boundary conditions have to be applied manually.
    Returns:
        If ``include_bc`` is ``True``, the assembled system matrix ``A`` and right-hand side ``l``, if ``include_bc`` is ``False`` a single :class:`DirichletBC` object containing all Dirichlet boundary conditions for the trial space is returned additionally.
        
    Example:
        >>> from dolfinx import fem, mesh
        >>> from scipy.sparse.linalg import spsolve
        >>> # ... mesh creation ...
        >>> U = fem.functionspace(...)
        >>> V = fem.functionspace(...)
        >>> # ... boundary facets location ...
        >>> bcs = [dirichletbc(U, uD, facets_U), 
                   dirichletbc(V, vD, facets_V)]
        >>> # ... define variational forms A and l ...
        
        >>> A,l = assemble_system((A, l), bcs, include_bc=True)
        >>> u1 = spsolve(A, l)
        
        >>> A,l,trial_bc = assemble_system((A, l), bcs, include_bc=False)
        >>> u2 = np.empty((trial_bc.ndofs,))
        >>> u2[trial_bc.free_dofs]  = spsolve(A, l)
        >>> u2[trial_bc.fixed_dofs] = trial_bc.values
    """
    if isinstance(F, tuple):
        A,l = F
    else:
        A,l = ufl.system(F)
    trial_space = A.arguments()[1].ufl_function_space()
    test_space  = A.arguments()[0].ufl_function_space()
    
    if not test_space == l.arguments()[0].ufl_function_space():
        raise ValueError("Test space of A and l do not match.")
    
    [trial_bc,test_bc] = collect_dirichletbcs(bcs, [trial_space, test_space])
    
    A = fem.assemble_matrix(fem.form(A)).to_scipy()
    l = fem.assemble_vector(fem.form(l)).array
    
    if include_bc:
        A_dirichlet = sparse.coo_array((np.ones(len(trial_bc.fixed_dofs)), (np.arange(len(trial_bc.fixed_dofs)), trial_bc.fixed_dofs)), shape=(len(trial_bc.fixed_dofs), A.shape[1])).tocsr()
        A = sparse.vstack([A_dirichlet, A[test_bc.free_dofs,:]])
        l = np.concatenate((trial_bc.values, l[test_bc.free_dofs]))
        return A,l
    else:
        A = A[test_bc.free_dofs,:].tocsc()
        l = l[test_bc.free_dofs] - A[:, trial_bc.fixed_dofs] @ trial_bc.values
        A = A[:, trial_bc.free_dofs]
        return A,l,trial_bc
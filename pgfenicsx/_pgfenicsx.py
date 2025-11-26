# Moritz Feuerle, 2025

__all__ = ['DirichletBC', 'dirichletbc']


import numpy as np
from numbers import Number
from dolfinx import default_scalar_type
from dolfinx.fem import locate_dofs_topological, Function, Constant, Expression
from dolfinx.fem.function import FunctionSpace
from dolfinx.mesh import exterior_facet_indices

from typing import Callable, TypeAlias
BoundaryFunctionType: TypeAlias = Function | Constant | Expression | Number | Callable


def as_numpy_vector(x, dtype=None):
    """Ensure that x is a numpy array of the given dtype and shape (n,)."""
    return np.asarray(x, dtype=dtype).reshape((-1,))



class DirichletBC:
    r"""Class representing a Dirichlet boundary condition."""
    
    def __init__(self, function_space: FunctionSpace, u: BoundaryFunctionType | np.ndarray[float], dofs: np.ndarray[int]):
        r"""Create a Dirichlet boundary condition restricting the ``dofs`` of ``function_space`` with the the values given by ``u``.
        
        Parameters
        ----------
        function_space
            The function space the boundary condition is defined on.
        u
            Object defining the dirichlet values at the given dofs. Can be given either the same way as in :func:`dirichletbc`, or as a discrete array directly specifying the the values at each dof, i.e. an array of a) same length as ``dofs``, i.e. ``u[i]`` containes the dirichlet value of dof ``dofs[i]``, or b) the length of all dofs of the function space, i.e. ``u[dofs[i]]`` containes the dirichlet value of dof ``dofs[i]``.
        dofs
            The dofs of the function space on which the Dirichlet condition is imposed.
        """
        
        if isinstance(u, Function):
            if not u.function_space == function_space:
                raise ValueError('Function u must be defined on the same function space as the DirichletBC')
            values = u.x.array[dofs]
        elif isinstance(u, Constant):
            values = np.full(dofs.shape, u.value, dtype=default_scalar_type)
        elif isinstance(u, Number):
            values = np.full(dofs.shape, default_scalar_type(u))
        else:
            try:
                values = as_numpy_vector(u, default_scalar_type)
            except:
                try:
                    u_func = Function(function_space)
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
        """Fixed dofs corresponding to the dirichlet values"""
        self.values: np.ndarray[float] = values[idx]
        """Discrete dirichlet values, i.e. ``values[i]`` is the dirichlet value at dof ``fixed_dofs[i]``."""
        self.function_space: FunctionSpace = function_space
        """Function space the boundary condition is defined on."""
        self._free_dofs = None
        self.ndofs: int = self.function_space.dofmap.index_map.size_global
        """Total number of dofs of the function space (free+fixed)."""
        
    @property
    def free_dofs(self) -> np.ndarray[int]:
        """Free dofs where no dirichlet condition is applied."""
        if self._free_dofs is None:
            free_dofs = np.ones(self.ndofs, dtype=bool)
            free_dofs[self.fixed_dofs] = False
            self._free_dofs = np.sort(np.where(free_dofs)[0])
        return self._free_dofs

      
def dirichletbc(function_space: FunctionSpace, u: BoundaryFunctionType, facets: np.ndarray[int] | None = None) -> DirichletBC:
    r"""Create a Dirichlet boundary condition for ``function_space`` on the boundary part ``facets`` with the boundary values given by ``u``.
    
    This function operates on **facets not dofs**, unlike :func:`dolfinx.fem.dirichletbc`. This decision was made conserning Petrov-Galerkin formulations: the dofs of the test and trial space may differ, while the facets belong to the mesh and are thus independend of the space (thus avoiding potential hard to debug errors by defining dirichlet condition for one space while using the facet dofs generated based on another space). 
    If one wants to create a dirichlet boundary condition based on dofs, one can always use :class:`DirichletBC` directly. 
    
    Parameters
    ----------
    function_space
        The function space the boundary condition is defined on.
    u
        The boundary value. Can be a number, :class:`dolfinx.fem.Constant`, :class:`dolfinx.fem.Function` or any object that can be interpolated into a :class:`dolfinx.fem.Function`.
    facets
        The boundary facets where the Dirichlet condition is applied. If ``None``, all exterior facets are used.
    """
   
    if facets is None:
        facets = exterior_facet_indices(function_space.mesh.topology)
    
    facets = as_numpy_vector(facets, np.int32)
    bdry_dofs = locate_dofs_topological(function_space, function_space.mesh.topology.dim-1, facets)
    
    return DirichletBC(function_space, u, bdry_dofs)
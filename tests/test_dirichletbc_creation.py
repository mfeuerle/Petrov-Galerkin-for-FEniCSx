import pathlib
import sys

from ufl import functionspace

sys.path.append(str(pathlib.Path(__file__).parents[1]))

import numpy as np
import pytest

from mpi4py import MPI
from dolfinx.mesh import create_unit_square, locate_entities_boundary, exterior_facet_indices
from dolfinx.fem import functionspace, Constant, Function, locate_dofs_topological
from dolfinx import default_scalar_type
from numbers import Number

from pgfenicsx import DirichletBC, dirichletbc


mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)
mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
U = functionspace(mesh, ("Lagrange", 1))
        
sub_marker  = lambda x: np.isclose(x[0], 0)
sub_facets = locate_entities_boundary(mesh, dim=mesh.topology.dim-1, marker=sub_marker)
sub_dofs_fixed = np.sort(locate_dofs_topological(U, mesh.topology.dim-1, sub_facets))
sub_dofs_free = np.ones(U.dofmap.index_map.size_global, dtype=bool)
sub_dofs_free[sub_dofs_fixed] = False
sub_dofs_free = np.sort(np.where(sub_dofs_free)[0])

full_facets = exterior_facet_indices(mesh.topology)
full_dofs_fixed = np.sort(locate_dofs_topological(U, mesh.topology.dim-1, full_facets))
full_dofs_free = np.ones(U.dofmap.index_map.size_global, dtype=bool)
full_dofs_free[full_dofs_fixed] = False
full_dofs_free = np.sort(np.where(full_dofs_free)[0])


full_facets = exterior_facet_indices(mesh.topology)


def _test(bc, values, fixed_dofs, free_dofs, space):
    assert np.allclose(bc.values, values[fixed_dofs])
    assert np.array_equal(bc.fixed_dofs, fixed_dofs)
    assert np.array_equal(bc.free_dofs, free_dofs)
    assert bc.function_space == space
    


class _Test_dirichletbc_interface:
    
    u = None  # to be defined in subclasses
    expected_error = None
    
    sub_facets = sub_facets
    sub_dofs_fixed   = sub_dofs_fixed
    sub_dofs_free = sub_dofs_free
    full_facets = full_facets
    full_dofs_fixed   = full_dofs_fixed
    full_dofs_free = full_dofs_free
    
    _values = None
    @property
    def values(self):
        if self._values is None:
            if isinstance(self.u, Function):
                values = self.u.x.array
            elif isinstance(self.u, Constant):
                values = np.full(U.dofmap.index_map.size_global, self.u.value, dtype= default_scalar_type)
            elif isinstance(self.u, Number):
                values = np.full(U.dofmap.index_map.size_global, default_scalar_type(self.u))
            else:
                _u = Function(U)
                _u.interpolate(self.u)
                values = _u.x.array
            self._values = values
        return self._values
    
    def _test(self, facets, fixed_dofs, free_dofs):
        try:
            if facets is None:
                bc = dirichletbc(U, self.u)
            else:
                bc = dirichletbc(U, self.u, facets)
        except self.expected_error:
            return
        _test(bc, self.values, fixed_dofs, free_dofs, U)
        
    def test_fullboundary(self):
        self._test(self.full_facets, self.full_dofs_fixed, self.full_dofs_free)
        
    def test_subboundary(self):
        self._test(self.sub_facets, self.sub_dofs_fixed, self.sub_dofs_free)
        
    def test_noboundary(self):
        self._test(None, self.full_dofs_fixed, self.full_dofs_free)

    
class Test_dirichletbc_Number(_Test_dirichletbc_interface):
    u = 2.0
        
class Test_dirichletbc_Constant(_Test_dirichletbc_interface):
    u = Constant(mesh, 3.0)
    
class Test_dirichletbc_Callable(_Test_dirichletbc_interface):
    u = lambda self,x: x[0] + x[1]
    
class Test_dirichletbc_Function(_Test_dirichletbc_interface):
    u = Function(U)
    u.interpolate(lambda x: x[0] * x[1])
    
class Test_dirichletbc_wrong_space(_Test_dirichletbc_interface):
    u = Function(functionspace(mesh, ("Lagrange", 1)))
    u.interpolate(lambda x: x[0] + x[1])    
    expected_error = ValueError
    




values = np.linspace(0, 1, U.dofmap.index_map.size_global)
dofes_fixed = sub_dofs_fixed
dofs_free   = sub_dofs_free

def test_DirichletBC_dof_length():
    bc = DirichletBC(U, values[dofes_fixed], dofes_fixed)
    _test(bc, values, dofes_fixed, dofs_free, U)
    
def test_DirichletBC_space_length():
    bc = DirichletBC(U, values, dofes_fixed)
    _test(bc, values, dofes_fixed, dofs_free, U)
    
def test_DirichletBC_wrong_length():
    with pytest.raises(ValueError):
        DirichletBC(U, values[:len(dofs_free)+1], dofs_free)
        
def test_DirichletBC_duplicate_dofs():
    with pytest.raises(ValueError):
        dofs = np.concatenate([dofes_fixed, dofes_fixed[:1]])
        DirichletBC(U, values[dofs], dofs)
 

# if __name__ == "__main__":
#     T = Test_dirichletbc_Function()
#     T.test_fullboundary()
#     T.test_subboundary()
#     T.test_noboundary()
    
#     T = Test_dirichletbc_Callable()
#     T.test_fullboundary()
#     T.test_subboundary()
#     T.test_noboundary()
    
#     T = Test_dirichletbc_Constant()
#     T.test_fullboundary()
#     T.test_subboundary()
#     T.test_noboundary() 
    
#     T = Test_dirichletbc_Number()
#     T.test_fullboundary()
#     T.test_subboundary()
#     T.test_noboundary()
    
#     test_DirichletBC_dof_length()
#     test_DirichletBC_space_length()
#     test_DirichletBC_wrong_length()
#     test_DirichletBC_duplicate_dofs()

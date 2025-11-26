import numpy as np
import pytest

from mpi4py import MPI
from dolfinx.mesh import create_unit_square, locate_entities_boundary, exterior_facet_indices
from dolfinx.fem import functionspace, Constant, Function, locate_dofs_topological
from dolfinx import default_scalar_type

import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).parents[1]))
from pgfenicsx import DirichletBC, dirichletbc


mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)
mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
U = functionspace(mesh, ("Lagrange", 1))
        
marker  = lambda x: np.isclose(x[0], 0)
facets = locate_entities_boundary(mesh, dim=mesh.topology.dim-1, marker=marker)
fixed_dofs = np.sort(locate_dofs_topological(U, mesh.topology.dim-1, facets))


def _test_bc(bc, values, fixed_dofs, free_dofs, space):
    assert np.allclose(bc.values, values[fixed_dofs])
    assert np.array_equal(bc.fixed_dofs, fixed_dofs)
    assert np.array_equal(bc.free_dofs, free_dofs)
    assert bc.function_space == space
    

class _Test_DirichletBC_interface:
    
    u = None  # to be defined in subclasses
    values = None
    fixed_dofs = None 
    expected_error = None
    
    _free_dofs = None
    @property
    def free_dofs(self):
        if self._free_dofs is None:
            free_dofs = np.ones(U.dofmap.index_map.size_global, dtype=bool)
            free_dofs[self.fixed_dofs] = False
            self._free_dofs = np.sort(np.where(free_dofs)[0])
        return self._free_dofs
    
    def test(self):
        if self.expected_error is not None:
            with pytest.raises(self.expected_error):
                bc = DirichletBC(U, self.u, self.fixed_dofs)
        else:
            bc = DirichletBC(U, self.u, self.fixed_dofs)
            _test_bc(bc, self.values, self.fixed_dofs, self.free_dofs, U)

    
class Test_DirichletBC_Number(_Test_DirichletBC_interface):
    u = 2
    values = np.full(U.dofmap.index_map.size_global, 2.0, dtype=default_scalar_type)
    fixed_dofs = fixed_dofs
        
class Test_DirichletBC_Constant(_Test_DirichletBC_interface):
    u = Constant(mesh, 3.4)
    values = np.full(U.dofmap.index_map.size_global, 3.4, dtype=default_scalar_type)
    fixed_dofs = fixed_dofs
    
class Test_DirichletBC_Callable(_Test_DirichletBC_interface):
    u = lambda self,x: x[0] + x[1]
    values = Function(U)
    values.interpolate(lambda x: x[0] + x[1])
    values = values.x.array    
    fixed_dofs = fixed_dofs
class Test_DirichletBC_Function(_Test_DirichletBC_interface):
    u = Function(U)
    u.interpolate(lambda x: x[0] * x[1])
    values = u.x.array
    fixed_dofs = fixed_dofs
    
class Test_DirichletBC_Array_short(_Test_DirichletBC_interface):
    values = np.linspace(0, 1, U.dofmap.index_map.size_global)
    u = values[fixed_dofs]
    fixed_dofs = fixed_dofs
    
class Test_DirichletBC_Array_full(_Test_DirichletBC_interface):
    values = np.linspace(0, 1, U.dofmap.index_map.size_global)
    u = values
    fixed_dofs = fixed_dofs
    
class Test_DirichletBC_Array_wrong_length(_Test_DirichletBC_interface):
    values = np.linspace(0, 1, U.dofmap.index_map.size_global)
    u = np.concatenate([values, values[:1]])
    fixed_dofs = fixed_dofs
    expected_error = ValueError
    
class Test_DirichletBC_duplicate_dofs(_Test_DirichletBC_interface):
    values = np.linspace(0, 1, U.dofmap.index_map.size_global)
    u = np.concatenate([values, values[fixed_dofs[:1]]])
    fixed_dofs = np.concatenate([fixed_dofs, fixed_dofs[:1]])
    expected_error = ValueError
    
class Test_DirichletBC_wrong_space(_Test_DirichletBC_interface):
    u = Function(functionspace(mesh, ("Lagrange", 1)))
    u.interpolate(lambda x: x[0] + x[1])    
    values = u.x.array
    fixed_dofs = fixed_dofs
    expected_error = ValueError
    


free_dofs = np.ones(U.dofmap.index_map.size_global, dtype=bool)
free_dofs[fixed_dofs] = False
free_dofs = np.sort(np.where(free_dofs)[0])
            
facets_all = exterior_facet_indices(mesh.topology)
fixed_dofs_all = np.sort(locate_dofs_topological(U, mesh.topology.dim-1, facets_all))
free_dofs_all = np.ones(U.dofmap.index_map.size_global, dtype=bool)
free_dofs_all[fixed_dofs_all] = False
free_dofs_all = np.sort(np.where(free_dofs_all)[0])

def test_dirichletbc_facets_given():
    values = np.linspace(0, 1, U.dofmap.index_map.size_global)
    bc = dirichletbc(U, values, facets=facets)
    _test_bc(bc, values, fixed_dofs, free_dofs, U)
    
def test_dirichletbc_no_facets():
    values = np.linspace(0, 1, U.dofmap.index_map.size_global)
    bc = dirichletbc(U, values)
    _test_bc(bc, values, fixed_dofs_all, free_dofs_all, U)
 

# if __name__ == "__main__":
#     Test_DirichletBC_Function().test()
#     Test_DirichletBC_Constant().test()
#     Test_DirichletBC_Number().test()
#     Test_DirichletBC_Callable().test()
#     Test_DirichletBC_Array_short().test()
#     Test_DirichletBC_Array_full().test()
#     Test_DirichletBC_Array_wrong_length().test()
#     Test_DirichletBC_wrong_space().test()
#     Test_DirichletBC_duplicate_dofs().test()
    
#     test_dirichletbc_facets_given()
#     test_dirichletbc_no_facets()

import numpy as np
from scipy.sparse.linalg import lsqr
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from petsc4py import PETSc
from enum import IntEnum
import pgfenicsx

np.set_printoptions(edgeitems=30, linewidth=100000, precision=2)


############################################
# Create mesh and define problem parameters
############################################
I     = [0.0, 3.0]
Omega = [0.0, 2.0]

nx = 15
nt = np.ceil((I[1]-I[0])/((Omega[1]-Omega[0])/(nx+1))).astype('int')    # ensure CFL condition

msh = mesh.create_rectangle(MPI.COMM_WORLD, [[I[0], Omega[0]], [I[1], Omega[1]]], [nt, nx])
msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)

tx = ufl.SpatialCoordinate(msh)
n  = ufl.FacetNormal(msh)

u_exact = lambda tx: np.sin(np.pi*tx[0])*tx[1] + 1  # +1 to avoid zero boundary values
f  = -ufl.pi**2 * ufl.sin(ufl.pi*tx[0])*tx[1]       # right-hand side
u0 = u_exact                                        # dirichlet BC at initial time    
u1 = ufl.pi * ufl.cos(ufl.pi*tx[0])*tx[1]           # neuman BC at initial time
g0 = u_exact                                        # dirichlet BC at spatial boundary

############################################
# Define function spaces and boundary parts
############################################
U = fem.functionspace(msh, ("Lagrange", 1))
V = fem.functionspace(msh, ("Lagrange", 1))

class bdry_prt(IntEnum): t0,T,gamma = range(3)
bdry  = [(bdry_prt.t0, lambda tx: np.isclose(tx[0], I[0])),  # initial time
         (bdry_prt.T,  lambda tx: np.isclose(tx[0], I[1])),  # terminal time
         (bdry_prt.gamma,)]                                  # boundary in space (= the remaining bounary parts)

############################################
# Setup Dirichlet BCs usning pgfenicsx
############################################
bdry_tagged = pgfenicsx.setup_boundary_meshtags(msh, bdry)  # just a convenience function to create meshtags for boundaries
ds = ufl.Measure("ds", domain=msh, subdomain_data=bdry_tagged)

# bcs = [ pgfenicsx.dirichletbc(U, u0, bdry_tagged.find(bdry_prt.t0)),
#         pgfenicsx.dirichletbc(U, g0, bdry_tagged.find(bdry_prt.gamma)),
#         pgfenicsx.dirichletbc(V, 0,  bdry_tagged.find(bdry_prt.T)),
#         pgfenicsx.dirichletbc(V, 0,  bdry_tagged.find(bdry_prt.gamma))]

############################################
# Setup Dirichlet BCs usning fenicsx
############################################
u0_ = fem.Function(U)
u0_.interpolate(u0)
g0_ = fem.Function(U)
g0_.interpolate(g0)
tdim = msh.topology.dim-1
bcs = [ fem.dirichletbc(u0_, fem.locate_dofs_topological(U,tdim,bdry_tagged.find(bdry_prt.t0))),
        fem.dirichletbc(g0_, fem.locate_dofs_topological(U,tdim,bdry_tagged.find(bdry_prt.gamma))),
        fem.dirichletbc(0.0, fem.locate_dofs_topological(V,tdim,bdry_tagged.find(bdry_prt.T)), V),
        fem.dirichletbc(0.0, fem.locate_dofs_topological(V,tdim,bdry_tagged.find(bdry_prt.gamma)), V)]
u = ufl.TrialFunction(U)
v = ufl.TestFunction(V)

############################################
# Setup variational problem
############################################
A =  - ufl.inner(ufl.grad(u)[0],ufl.grad(v)[0]) * ufl.dx
for i in range(1,len(tx)):
    A += ufl.inner(ufl.grad(u)[i],ufl.grad(v)[i]) * ufl.dx

l = f*v*ufl.dx + u1*v*ds(bdry_prt.t0)

u_exact_ = fem.Function(U)
u_exact_.interpolate(u_exact)

############################################
# Solve the variational problem using SciPy
#############################################
A_scipy,l_scipy = pgfenicsx.assemble_system((A,l), bcs, petsc=False)
u_scipy = fem.Function(U)
u_scipy.x.array[:] = lsqr(A_scipy,l_scipy)[0]

print(np.linalg.norm(u_scipy.x.array - u_exact_.x.array, ord=np.inf))

############################################
# Solve the variational problem using PETSc
#############################################
A_petsc,l_petsc = pgfenicsx.assemble_system((A,l), bcs, petsc=True) 
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(A_petsc)
solver.setType("preonly")
solver.getPC().setType("qr")
solver.setFromOptions()
u_petsc = fem.Function(U)
u_petsc.x.petsc_vec.ghostUpdate(
    addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
)
solver.solve(l_petsc, u_petsc.x.petsc_vec)

print(np.linalg.norm(u_petsc.x.array - u_exact_.x.array, ord=np.inf))

############################################
# Visualise the solution using pyvista
#############################################   
try:
    import pyvista
    from dolfinx import plot
    from pathlib import Path
    
    results_folder = Path(__file__).parent / f'plots_{Path(__file__).stem}'
    
    def plot_pyvista(u,name, plotter):
        cells, types, x = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(cells, types, x)
        grid.point_data["u"] = u
        grid.set_active_scalars("u")
        grid.rotate_z(180, inplace=True)
        plotter.add_mesh(grid.warp_by_scalar(), show_edges=True)
        plotter.show_grid(xtitle='t', ytitle='x', ztitle='u(t,x)')
        plotter.add_text(name)
        
    
    plotter = pyvista.Plotter(shape=(1,3))
    plotter.subplot(0, 0)
    plot_pyvista(u_exact_.x.array, "u_exact", plotter)
    plotter.subplot(0, 1)
    plot_pyvista(u_scipy.x.array, "u1", plotter)
    plotter.subplot(0, 2)
    plot_pyvista(u_petsc.x.array, "u2", plotter)
    # plotter.subplot(1, 1)
    # plot_pyvista(u1 - u_exact_discrete.x.array, "error u1", plotter)
    # plotter.subplot(1, 2)
    # plot_pyvista(u2 - u_exact_discrete.x.array, "error u2", plotter)
    
    if pyvista.OFF_SCREEN:
        plotter.screenshot(results_folder / f".png")
    else:
        plotter.show()

except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution.")
    print("To install pyvista with pip: 'python3 -m pip install pyvista' or conda: 'conda install -c conda-forge pyvista'.")
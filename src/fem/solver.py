import sys
import numpy as np
from configs.hyperparameters import (
    GAMMA_V, RHO_L_V, U_L_V, P_L_V, RHO_R_V, U_R_V, P_R_V,
    X_DIAPH, T_FINAL, DT, NX, BETA_V, K_S,
    U1_REF_V, U2_REF_V, U3_REF_V, SNAPSHOT_FILE
)
from src.utils.hashing import fem_fingerprint

def run_fem(snapshot_path: str = SNAPSHOT_FILE) -> None:
    """
    Run the SUPG-YZβ Sod-shock-tube solve in FEniCS and write the snapshot bundle.
    """
    import numpy as _np
    if not hasattr(_np, "product"):       _np.product       = _np.prod
    if not hasattr(_np, "cumproduct"):    _np.cumproduct    = _np.cumprod
    if not hasattr(_np, "alltrue"):       _np.alltrue       = _np.all
    if not hasattr(_np, "sometrue"):      _np.sometrue      = _np.any

    from fenics import (
        Constant, IntervalMesh, FiniteElement, MixedElement, FunctionSpace,
        Function, TestFunctions, split, as_vector, as_matrix, dot, sqrt, dx,
        CellDiameter, UserExpression, interpolate, derivative,
        NonlinearVariationalProblem, NonlinearVariationalSolver,
        parameters, set_log_level, LogLevel
    )

    sys.setrecursionlimit(1000000000)
    parameters["allow_extrapolation"] = True
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["representation"] = "uflacs"
    parameters["form_compiler"]["no-evaluate_basis_derivatives"] = False
    parameters['form_compiler']['quadrature_degree'] = 9
    set_log_level(LogLevel.INFO)

    gamma   = Constant(GAMMA_V)
    gamma_v = GAMMA_V
    rho_L_v, u_L_v, p_L_v = RHO_L_V, U_L_V, P_L_V
    rho_R_v, u_R_v, p_R_v = RHO_R_V, U_R_V, P_R_V
    x_diaph = X_DIAPH
    U1_ref  = Constant(U1_REF_V)
    U2_ref  = Constant(U2_REF_V)
    U3_ref  = Constant(U3_REF_V)
    beta_v  = BETA_V

    mesh = IntervalMesh(NX, 0.0, 1.0)
    print("Number of Cells:", mesh.num_cells())
    print("Number of Nodes:", mesh.num_vertices())
    
    from fenics import interval as _interval
    P1 = FiniteElement('P', _interval, 1)
    element = MixedElement([P1, P1, P1])
    V = FunctionSpace(mesh, element)

    U   = Function(V)
    rho, q, E = split(U)
    U_n = Function(V)
    rho_n, q_n, E_n = split(U_n)
    phi1, phi2, phi3 = TestFunctions(V)
    
    U_vec   = as_vector([rho,   q,   E])
    U_n_vec = as_vector([rho_n, q_n, E_n])
    W_vec   = as_vector([phi1,  phi2, phi3])

    class InitialConditions(UserExpression):
        def eval(self, values, x):
            if x[0] < x_diaph:
                r, u, p = rho_L_v, u_L_v, p_L_v
            else:
                r, u, p = rho_R_v, u_R_v, p_R_v
            values[0] = r
            values[1] = r * u
            values[2] = p / (gamma_v - 1.0) + 0.5 * r * u * u
        def value_shape(self):
            return (3,)
            
    init = InitialConditions(degree=2)
    U_n  = interpolate(init, V)
    U.assign(U_n)
    rho_n, q_n, E_n = split(U_n)
    U_n_vec = as_vector([rho_n, q_n, E_n])

    u_expr = q / rho
    p_expr = (gamma - 1.0) * (E - 0.5 * q * q / rho)
    H_expr = (E + p_expr) / rho
    a_expr = sqrt(gamma * p_expr / rho)
    
    u_n_expr = q_n / rho_n
    p_n_expr = (gamma - 1.0) * (E_n - 0.5 * q_n * q_n / rho_n)
    H_n_expr = (E_n + p_n_expr) / rho_n
    a_n_expr = sqrt(gamma * p_n_expr / rho_n)

    def jacobian_A(rho_, q_, E_):
        u_  = q_ / rho_
        p_  = (gamma - 1.0) * (E_ - 0.5 * q_ * q_ / rho_)
        H_  = (E_ + p_) / rho_
        A11 = Constant(0.0); A12 = Constant(1.0); A13 = Constant(0.0)
        A21 = 0.5 * (gamma - 3.0) * u_ * u_
        A22 = (3.0 - gamma) * u_
        A23 = gamma - 1.0
        A31 = u_ * (0.5 * (gamma - 1.0) * u_ * u_ - H_)
        A32 = H_ - (gamma - 1.0) * u_ * u_
        A33 = gamma * u_
        return as_matrix([[A11, A12, A13],
                          [A21, A22, A23],
                          [A31, A32, A33]])
                          
    A_cur = jacobian_A(rho,   q,   E)
    A_lin = jacobian_A(rho_n, q_n, E_n)

    dU_dx   = as_vector([rho.dx(0),   q.dx(0),   E.dx(0)])
    dU_n_dx = as_vector([rho_n.dx(0), q_n.dx(0), E_n.dx(0)])
    dW_dx   = as_vector([phi1.dx(0),  phi2.dx(0), phi3.dx(0)])

    R_strong = (U_vec - U_n_vec) / DT + A_cur * dU_dx
    h        = CellDiameter(mesh)
    smax     = abs(u_n_expr) + a_n_expr
    tau_SUPG = h / (2.0 * smax)

    Yinv_vec = as_vector([1.0/U1_ref, 1.0/U2_ref, 1.0/U3_ref])
    Z_vec    = A_cur * dU_dx
    YinvZ    = as_vector([Yinv_vec[0]*Z_vec[0], Yinv_vec[1]*Z_vec[1], Yinv_vec[2]*Z_vec[2]])
    YinvdU   = as_vector([Yinv_vec[0]*dU_dx[0], Yinv_vec[1]*dU_dx[1], Yinv_vec[2]*dU_dx[2]])
    eps_floor    = Constant(1.0e-12)
    normYinvZ_sq = YinvZ[0]**2  + YinvZ[1]**2  + YinvZ[2]**2  + eps_floor
    normYinvdU_2 = YinvdU[0]**2 + YinvdU[1]**2 + YinvdU[2]**2 + eps_floor
    h_SHOC  = h
    nu_SHOC = (sqrt(normYinvZ_sq) * normYinvdU_2**(beta_v/2.0 - 1.0) * (h_SHOC/2.0)**beta_v)

    GAL  = dot(W_vec, (U_vec - U_n_vec)/DT) * dx + dot(W_vec, A_cur * dU_dx) * dx
    A_lin_T = as_matrix([[A_lin[0,0], A_lin[1,0], A_lin[2,0]],
                         [A_lin[0,1], A_lin[1,1], A_lin[2,1]],
                         [A_lin[0,2], A_lin[1,2], A_lin[2,2]]])
    AT_dW = A_lin_T * dW_dx
    SUPG  = tau_SUPG * dot(AT_dW, R_strong) * dx
    SHOC  = nu_SHOC * dot(dW_dx, dU_dx) * dx
    F     = GAL + SUPG + SHOC

    problem = NonlinearVariationalProblem(F, U, J=derivative(F, U))
    solver  = NonlinearVariationalSolver(problem)
    prm = solver.parameters["newton_solver"]
    prm["absolute_tolerance"] = 1E-10
    prm['relative_tolerance'] = 1E-10
    prm['maximum_iterations'] = 50
    prm['convergence_criterion'] = 'incremental'
    prm['krylov_solver']['absolute_tolerance']      = 1E-10
    prm['krylov_solver']['relative_tolerance']      = 1E-10
    prm['krylov_solver']['maximum_iterations']      = 1000
    prm['krylov_solver']['monitor_convergence']     = True
    prm['krylov_solver']['nonzero_initial_guess']   = True
    prm['krylov_solver']['error_on_nonconvergence'] = True
    prm['krylov_solver']['report']                  = True

    x_nodes        = mesh.coordinates().flatten().astype(np.float64)
    x_nodes_sorted = x_nodes[np.argsort(x_nodes)]
    snap_times, snap_rho, snap_q, snap_E = [], [], [], []
    
    def _nodal(field):
        out = np.empty_like(x_nodes_sorted)
        for i, xp in enumerate(x_nodes_sorted):
            out[i] = field(xp)
        return out

    t = 0.0
    while t < T_FINAL - 1e-14:
        dt_local = (T_FINAL - t) if (t + DT > T_FINAL) else DT
        t += dt_local
        solver.solve()
        rho_sol, q_sol, E_sol = U.split()
        
        snap_times.append(float(t))
        snap_rho.append(_nodal(rho_sol))
        snap_q  .append(_nodal(q_sol))
        snap_E  .append(_nodal(E_sol))
        if len(snap_times) > K_S:
            snap_times.pop(0); snap_rho.pop(0); snap_q.pop(0); snap_E.pop(0)
            
        U_n.assign(U)
        rho_n, q_n, E_n = split(U_n)
        U_n_vec = as_vector([rho_n, q_n, E_n])
        print("t =", t)

    fp = fem_fingerprint()
    np.savez(snapshot_path,
             x        = x_nodes_sorted,
             t_snap   = np.asarray(snap_times,             dtype=np.float64),
             rho_snap = np.stack(snap_rho, axis=0).astype(np.float64),
             q_snap   = np.stack(snap_q,   axis=0).astype(np.float64),
             E_snap   = np.stack(snap_E,   axis=0).astype(np.float64),
             gamma    = np.float64(GAMMA_V),
             U_ref    = np.array([U1_REF_V, U2_REF_V, U3_REF_V], dtype=np.float64),
             t_final  = np.float64(snap_times[-1]),
             dt       = np.float64(DT),
             nx       = np.int64(NX),
             fem_fingerprint = np.array(fp))
             
    print(f"  -> {snapshot_path} saved ({K_S} snapshots, "
          f"{x_nodes_sorted.size} nodes each, fingerprint={fp})")
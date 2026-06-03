# ===========================================================================
#                       Exact Riemann solver (Toro, 2009)
# ===========================================================================

from configs.hyperparameters import (GAMMA_V, RHO_L_V, U_L_V, P_L_V,
                                            RHO_R_V, U_R_V, P_R_V, X_DIAPH)

def exact_riemann(x_arr, t_eval,
                  gamma_v=GAMMA_V,
                  rho_L=RHO_L_V, u_L=U_L_V, p_L=P_L_V,
                  rho_R=RHO_R_V, u_R=U_R_V, p_R=P_R_V,
                  x0=X_DIAPH):
    import numpy as np
    from scipy.optimize import brentq
    g  = gamma_v
    aL = np.sqrt(g * p_L / rho_L)
    aR = np.sqrt(g * p_R / rho_R)
    def fK(p, rhoK, pK, aK):
        if p > pK:
            AK = 2.0/((g+1.0)*rhoK); BK = (g-1.0)/(g+1.0)*pK
            return (p - pK) * np.sqrt(AK/(p + BK))
        return 2.0*aK/(g-1.0) * ((p/pK)**((g-1.0)/(2.0*g)) - 1.0)
    Ffun  = lambda p: fK(p,rho_L,p_L,aL) + fK(p,rho_R,p_R,aR) + (u_R - u_L)
    p_star = brentq(Ffun, 1e-8, 10.0*max(p_L,p_R))
    u_star = 0.5*(u_L+u_R) + 0.5*(fK(p_star,rho_R,p_R,aR)
                                  - fK(p_star,rho_L,p_L,aL))
    rho_=np.zeros_like(x_arr); u_=np.zeros_like(x_arr); p_=np.zeros_like(x_arr)
    for i, xi in enumerate(x_arr):
        s = (xi - x0) / t_eval
        if s <= u_star:
            if p_star <= p_L:
                rsL = rho_L*(p_star/p_L)**(1.0/g)
                asL = aL*(p_star/p_L)**((g-1.0)/(2.0*g))
                SHL = u_L - aL; STL = u_star - asL
                if s < SHL:    rho_[i],u_[i],p_[i] = rho_L, u_L, p_L
                elif s > STL:  rho_[i],u_[i],p_[i] = rsL, u_star, p_star
                else:
                    rho_[i] = rho_L*(2.0/(g+1.0)+(g-1.0)/((g+1.0)*aL)*(u_L-s))**(2.0/(g-1.0))
                    u_[i]   = 2.0/(g+1.0)*(aL+(g-1.0)*0.5*u_L+s)
                    p_[i]   = p_L*(2.0/(g+1.0)+(g-1.0)/((g+1.0)*aL)*(u_L-s))**(2.0*g/(g-1.0))
            else:
                pr = p_star/p_L
                rsL = rho_L*(pr+(g-1.0)/(g+1.0))/((g-1.0)/(g+1.0)*pr+1.0)
                SL  = u_L-aL*np.sqrt((g+1.0)/(2.0*g)*pr+(g-1.0)/(2.0*g))
                if s < SL:     rho_[i],u_[i],p_[i] = rho_L, u_L, p_L
                else:          rho_[i],u_[i],p_[i] = rsL, u_star, p_star
        else:
            if p_star <= p_R:
                rsR = rho_R*(p_star/p_R)**(1.0/g)
                asR = aR*(p_star/p_R)**((g-1.0)/(2.0*g))
                SHR = u_R + aR; STR = u_star + asR
                if s > SHR:    rho_[i],u_[i],p_[i] = rho_R, u_R, p_R
                elif s < STR:  rho_[i],u_[i],p_[i] = rsR, u_star, p_star
                else:
                    rho_[i] = rho_R*(2.0/(g+1.0)-(g-1.0)/((g+1.0)*aR)*(u_R-s))**(2.0/(g-1.0))
                    u_[i]   = 2.0/(g+1.0)*(-aR+(g-1.0)*0.5*u_R+s)
                    p_[i]   = p_R*(2.0/(g+1.0)-(g-1.0)/((g+1.0)*aR)*(u_R-s))**(2.0*g/(g-1.0))
            else:
                pr = p_star/p_R
                rsR = rho_R*(pr+(g-1.0)/(g+1.0))/((g-1.0)/(g+1.0)*pr+1.0)
                SR  = u_R+aR*np.sqrt((g+1.0)/(2.0*g)*pr+(g-1.0)/(2.0*g))
                if s > SR:     rho_[i],u_[i],p_[i] = rho_R, u_R, p_R
                else:          rho_[i],u_[i],p_[i] = rsR, u_star, p_star
    return rho_, u_, p_
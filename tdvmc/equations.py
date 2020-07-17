from sympy import *
import tdvmc.lark_test.mathematica_to_sympy as m2s
from functools import reduce
from operator import mul
from tdvmc.code_man import from_expr
from numba import njit
from pdb import set_trace
import numpy as np
from sympy.utilities.lambdify import lambdastr
from tdvmc.variables import *




def compute_funcs(equations, variables, does_debug=True, inline="always"):


    variables_wave_function = ((variables["f_0"]), (variables["f_i"]), (variables["f_ij"])) 
    variables_potential = ((variables["v_i"]), (variables["v_ij"]))

    cfg = (x, Dh, variables_potential, variables_wave_function)
    
    
    fi_0 =   Lambda((*cfg,),      equations["f_0"])
    fi_i =   Lambda((*cfg,i),     equations["f_i"])
    fi_ij =  Lambda((*cfg,i,j,),  equations["f_ij"])
    
    V_i =   Lambda((*cfg,i),     equations["V_i"])
    V_ij =  Lambda((*cfg,i,j,),  equations["V_ij"])
    
    Op_0 = Lambda((*cfg,),       equations["Op_0"])
    Op_1 = Lambda((*cfg,i),      equations["Op_1"])
    Op_2 = Lambda((*cfg,),       equations["Op_2"])


    r_i = Lambda((*cfg,i), Sum(x[i,a]**2, (a,0,2)).doit())
    r_ij = Lambda((*cfg,i,j), Sum((x[i,a] - x[j,a])**2, (a,0,2)).doit())

    
    
    ddf_f = [[simplify(fi_i(*cfg,i).diff(x[i,a],2) / fi_i(*cfg,i)) for a in range(3)] , 
             [powsimp(refine((fi_ij(*cfg,i,j).diff(x[i,a],2) / fi_ij(*cfg,i,j)).rewrite(Piecewise), Q.is_true(Ne(i,j)))) for a in range(3)],]
    df_f = [[simplify(fi_i(*cfg,i).diff(x[i,a]) / fi_i(*cfg,i)) for a in range(3)] , 
            [powsimp(refine((fi_ij(*cfg,i,j).diff(x[i,a]) / fi_ij(*cfg,i,j)).rewrite(Piecewise), Q.is_true(Ne(i,j)))) for a in range(3)],]
   
    ln_fi_0 =  Lambda((*cfg,), expand_log(ln(fi_0(*cfg,)),force=True))
    ln_fi_i =  Lambda((*cfg,i), expand_log(ln(fi_i(*cfg,i)), force=True))
    ln_fi_ij = Lambda((*cfg,i,j), expand_log(ln(fi_ij(*cfg,i,j)), force=True))

    dfi_0_dc =  [Lambda((*cfg,) , simplify(fi_0(*cfg).diff(c0) / fi_0(*cfg))) for c0 in config["Variables"]["f_0"]]
    dfi_i_dc =  [Lambda((*cfg,i,) , simplify(fi_i(*cfg,i).diff(ci) / fi_i(*cfg,i))) for ci in config["Variables"]["f_1"]]
    dfi_ij_dc = [Lambda((*cfg,i,j) , simplify(fi_ij(*cfg,i,j).diff(cij) / fi_ij(*cfg,i,j))) for cij in config["Variables"]["f_2"]]
   
    
    
    ddfi_i_dx = Lambda((*cfg,i,a), Piecewise(*((simplify(fi_i(*cfg,i).diff(x[i,_a],2) / fi_i(*cfg,i)), Eq(a,_a)) for _a in range(3))))
    ddfi_ij_dx = Lambda((*cfg,i,j,a), Piecewise(*((powsimp(refine((fi_ij(*cfg,i,j).diff(x[i,_a],2) / fi_ij(*cfg,i,j)).rewrite(Piecewise), Q.is_true(Ne(i,j)))), Eq(a,_a)) for _a in range(3))))
  
    dfi_i_dx = Lambda((*cfg,i,a), Piecewise(*((simplify(fi_i(*cfg,i).diff(x[i,_a]) / fi_i(*cfg,i)), Eq(a,_a)) for _a in range(3))))
    dfi_ij_dx = Lambda((*cfg,i,j,a), Piecewise(*((powsimp(refine((fi_ij(*cfg,i,j).diff(x[i,_a]) / fi_ij(*cfg,i,j)).rewrite(Piecewise), Q.is_true(Ne(i,j)))), Eq(a,_a)) for _a in range(3))))

    
    
    def dd_f_f(i,j,a):
        if i == j: return ddfi_i_dx(*cfg,i,a)
        else: return ddfi_ij_dx(*cfg,i,j,a)
        
    def d_f_f(i,j,a):
        if i == j: return dfi_i_dx(*cfg,i,a)
        else: return dfi_ij_dx(*cfg,i,j,a)

    sqrt = np.sqrt
    
    def sym_to_num(func): return njit()(eval(lambdastr(*func.args),{**globals(),**{"sqrt":np.sqrt}}))
#     def sym_to_num(func): return eval(lambdastr(*func.args))

        
    r_i, r_ij = map(sym_to_num, (r_i, r_ij))
    fi_0, fi_i, fi_ij= map(sym_to_num, (fi_0, fi_i, fi_ij))
    ln_fi_0, ln_fi_i, ln_fi_ij = map(sym_to_num, (ln_fi_0, ln_fi_i, ln_fi_ij))
    
    dfi_0_dc, dfi_i_dc, dfi_ij_dc =  map(lambda fs: tuple(map(sym_to_num, fs)), (dfi_0_dc, dfi_i_dc, dfi_ij_dc))
    ddfi_i_dx, ddfi_ij_dx = map(sym_to_num, (ddfi_i_dx, ddfi_ij_dx))
    dfi_i_dx, dfi_ij_dx= map(sym_to_num, (dfi_i_dx, dfi_ij_dx))
    V_i, V_ij = map(sym_to_num, (V_i, V_ij))
    
    
    @njit
    def diff_ln_Psi2(x, y, Dh, p, c):
        n = len(x)
        cfg1 = (x, Dh, p, c)
        cfg2 = (y, Dh, p, c)
        diff_ln = 0.0
        diff_ln += ln_fi_0(*cfg1) - ln_fi_0(*cfg2)
        for i in range(n): 
            for j in range(i,n):
                if i == j: diff_ln += ln_fi_i(*cfg1,i) - ln_fi_i(*cfg2,i)
                else: diff_ln += ln_fi_ij(*cfg1,i,j) - ln_fi_ij(*cfg2,i,j)
        return 2*diff_ln

    @jit
    def d_param(*cfg,i,k):
        (x, Dh, p, c) = cfg
        n = len(x)
        res = 0.0
        if i == 0: res = dfi_0_dc[k](*cfg)
        elif i == 1:
            for i in range(n):
                res += dfi_i_dc[k](*cfg,i)
            return res
        elif i == 2: 
            for i in range(n):
                for j in range(i+1,n):
                    res += dfi_ij_dc[k](*cfg,i,j)

        return res

    
    
    
    @njit
    def Op(*cfg, i):
        n = len(x)
        res = 0.0
        if i == 0: return Op0(*cfg)
        elif i == 1:
            for i in range(n): res +=  Op1(*cfg, i)
        elif i == 2:
            for i in range(n):
                for j in range(i+1,n): res += Op2(*cfg, i, j)
        return res
    

    @njit(fastmath=True)
    def El(x : float, Dh: float, p: Tuple(), c: Tuple()) -> float:
    #     cfg = (x, Dh, (w, k), (c0, gamma, alpha, c3)) = x, Dh, p, c
        c0s, cis, cijs = c
        cfg = (x, Dh, p,  c)
        res = 0.0
        v = 0.0
        n = len(x)
        for a in range(3):
            for i in range(n):
                cum_sum = 0.0
                for j in range(n):
                    if j == i:
                        if a == 0: v += V_i(*cfg, i)
                        res += ddfi_i_dx(*cfg, i, a)
                        res -= dfi_i_dx(*cfg, i, a)**2
                        cum_sum += dfi_i_dx(*cfg, i, a)
                    else:
                        if a == 0 and j >= i: v += V_ij(*cfg, i,j)
                        res += ddfi_ij_dx(*cfg, i,j, a)
                        res -= dfi_ij_dx(*cfg, i,j, a)**2
                        cum_sum += dfi_ij_dx(*cfg, i,j, a)

                res += cum_sum**2

        return -Dh*res + v

    @njit()
    def El_Op_dc_iter(Dh, w, k, c0, alpha, gamma, n_bodies, n_iters, sigma, L, seed=0, x=None, beta=None,):

        cfg = (Dh, (w, k), (c0, gamma, alpha, 0))

        def compute(x):
            "Computes El, Op_1 ... Op_3, D[Psi,Param[1]]/Psi ... D[Psi,Param[3]]/Psi"
            el = El(x, *cfg)
            op = np.array([Op(x,*cfg, 0),Op(x,*cfg, 1), Op(x,*cfg, 2)])
            diff_param = np.array([
                d_param(x,*cfg, 0),
                d_param(x,*cfg, 1),
                d_param(x,*cfg, 2)])
            return el , op, diff_param

        np.random.seed(seed)
        x = np.random.uniform(-L, L,(n_bodies, 3)) if x is None else x
        y = np.copy(x)
        el, op, diff_param = compute(x)

        # Where we save the values
        Op_list = np.empty((n_iters, n_bodies, 3))
        El_list = np.empty((n_iters, n_bodies))
        diff_param_list = np.empty((n_iters, n_bodies,3))

        for i in range(n_iters):
            for body in range(n_bodies):
                delta = np.random.uniform(-sigma, sigma,3)
                y[:,:] = x[:,:] 
                y[body,:] += delta
                if  diff_ln_Psi2(y,x,*cfg) >= np.log(np.random.rand()):
                    x[:,:] = y[:,:] # update bodies sampling position
                    el, op, diff_param = compute(x)

                diff_param_list[i,body,:] = diff_param
                Op_list[i,body,:] = op 
                El_list[i,body] = el

        return El_list, Op_list, diff_param_list

    @jit(fastmath=True)
    def Ev_Matesq_Vdret(Dh, p, c, n_bodies, n_iters, sigma, L, seed=0, x=None, beta=None):

        c0s, cis, cijs = c
        cfg = (Dh, p, c)

        def compute(x):
            "Computes El, Op_1 ... Op_3, D[Psi,Param[1]]/Psi ... D[Psi,Param[3]]/Psi"
            el = El(x,*cfg)
            op = np.array([
                Op(x,*cfg, 0),
                Op(x,*cfg, 1),
                Op(x,*cfg, 2)])
            diff_param = numpy.zeros((len(c0s) + len(cis) + len(cijs),))
            lens_params = np.array([len(c0s), len(cis), len(cijs)])
            for end, size, values_param in zip(np.cumsum(lens_params),lens_params, d_param(x,*cfg)):
                diff_param[end-size:end] = values_param
            diff_param = np.concatenate(d_param(x,*cfg))
            return el , op.reshape((-1,1)) * diff_param, el * op

        np.random.seed(seed)
        x = np.random.uniform(-L, L,(n_bodies, 3)) if x is None else x
        y = np.copy(x)

        # Where we save the values
        el_sum = 0
        matesq_sum = np.zeros((3,len(c0s) + len(cis) + len(cijs)))
        vdret_sum = np.zeros((3,))
        el, matesq, vdret = compute(x)
        for i in range(n_iters):
            sum_El = sum_El2 = 0.0
            for body in range(n_bodies):
                delta = np.random.uniform(-sigma, sigma,3)
                y[:,:] = x[:,:] 
                y[body,:] += delta
                if  diff_ln_Psi2(y,x, Dh, p, c) >= np.log(np.random.rand()):
                    x[:,:] = y[:,:] # update bodies sampling position
                    el, matesq, vdret = compute(x)

                vdret_sum += vdret
                matesq_sum += matesq
                el_sum += el

        times = n_iters * n_bodies

        return el_sum/times, matesq_sum/times, vdret_sum/times

    return El, El_Op_dc_iter, Ev_Matesq_Vdret
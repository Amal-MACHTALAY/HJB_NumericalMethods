#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import numpy as np
import math 
import matplotlib.pyplot as plt


# ## Numerical Hamiltonian

# In[2]:


def max_H(p1,p2,H):
    a, b = min(p1,p2), max(p1,p2)
    max_value = b
    for x in np.linspace(a,b,101):
        if H(max_value) < H(x): max_value = x
    return H(max_value)
    
def min_H(p1,p2,H):
    a, b = min(p1,p2), max(p1,p2)
    min_value = b
    for x in np.linspace(a,b,101):
        if H(min_value) > H(x): min_value = x
    return H(min_value)

# min-max of H'(p) in [a,b]
def max_abs_dpH(p1,p2,dpH):
    a, b = min(p1,p2), max(p1,p2)
    max_value = b
    for x in np.linspace(a,b,101):
        if abs(dpH(max_value)) < abs(dpH(x)): max_value = x
    return abs(dpH(max_value))
    
def min_abs_dpH(p1,p2,dpH):
    a, b = min(p1,p2), max(p1,p2)
    min_value = b
    for x in np.linspace(a,b,101):
        if abs(dpH(min_value)) > abs(dpH(x)): min_value = x
    return abs(dpH(min_value))

def max_dpH(p1,p2,dpH):
    a, b = min(p1,p2), max(p1,p2)
    max_value = b
    for x in np.linspace(a,b,101):
        if dpH(max_value) < dpH(x): max_value = x
    return dpH(max_value)
    
def min_dpH(p1,p2,dpH):
    a, b = min(p1,p2), max(p1,p2)
    min_value = b
    for x in np.linspace(a,b,101):
        if dpH(min_value) > dpH(x): min_value = x
    return dpH(min_value)



#************************** First-order monotone schemes *************

# Upwind 
def H_upwind(p1,p2,H,dpH):  #p1=p-,p2=p+
    if dpH(p1)<=0:
        return H(p2)
    else: return H(p1)

# Local-Lax-Friedrichs
def H_lax(p1,p2,alpha_x,H):   #p1=p-,p2=p+
    return H(0.5*(p1+p2))-0.5*alpha_x*(p2-p1)

# Godunov    
def H_godunov(p1,p2,H):    #p1=p-,p2=p+
    if p1<=p2: 
        return min_H(p1,p2,H)
    else: 
        return max_H(p1,p2,H)

# Roe with LLF entropy correction
def H_Roe(p1,p2,alpha_x,H,dpH):   #p1=p-,p2=p+
    if min_dpH(p1,p2,dpH)*max_dpH(p1,p2,dpH)>=0.0:   
        return H_upwind(p1,p2,H,dpH) 
    else: return H_lax(p1,p2,alpha_x,H)


# ## slope reconstruction

# In[3]:


def P_deg1(u1,u2,dx):
       # For p- : u1=u[i-1] , u2=u[i]
       # For p+ : u1=u[i] , u2=u[i+1]
    return (u2-u1)/dx

#************************* TVD Method ********************
def P_TVD2(u1,u2,dx):
       # For p- and p+ : u1=u[i-1] , u2=u[i+1]
    return (u2-u1)/(2*dx)

def P_TVD3(u1,u2,u3,u4,dx):
       # For p- : u1=u[i-2] , u2=u[i-1] , u3=u[i] , u4=u[i+1]
       # For p+ : u1=-u[i+2] , u2=-u[i+1] , u3=-u[i] , u4=-u[i-1]
    return ((u1/6)-u2+(u3/2)+(u4/3))/dx

#************************ ENO Schemes ********************************
def Pm_ENO2(u1,u2,u3,u4,dx):
# For p- : u1=u[i-2] , u2=u[i-1] , u3=u[i] , u4=u[i+1]
    CL=(u3-2*u2+u1)/(2*(dx**2))
    CR=(u4-2*u3+u2)/(2*(dx**2))
    if abs(CL)<abs(CR):
        return P_deg1(u2,u3,dx)+dx*CL
    else:
        return P_deg1(u2,u3,dx)+dx*CR

def Pp_ENO2(u1,u2,u3,u4,dx):
# For p+ : u1=u[i-1] , u2=u[i] , u3=u[i+1] , u4=u[i+2]
    CL=(u3-2*u2+u1)/(2*(dx**2))
    CR=(u4-2*u3+u2)/(2*(dx**2))
    if abs(CL)<abs(CR):
        return P_deg1(u2,u3,dx)-dx*CL
    else:
        return P_deg1(u2,u3,dx)-dx*CR
    
def Pm_ENO3(u1,u2,u3,u4,u5,u6,dx):
# For p- : u1=u[i-3] , u2=u[i-2] , u3=u[i-1] , u4=u[i] ,u5=u[i+1] , u6=u[i+2]
    Ca=(u4-3*u3+3*u2-u1)/(6*(dx**3))
    Cb=(u6-3*u5+3*u4-u3)/(6*(dx**3))
    Cc=(u5-3*u4+3*u3-u2)/(6*(dx**3))
    CL=(u4-2*u3+u2)/(2*(dx**2))
    CR=(u5-2*u4+u3)/(2*(dx**2))
    if abs(CL)<abs(CR) and abs(Ca)<abs(Cc):
        return Pm_ENO2(u2,u3,u4,u5,dx)+2*(dx**2)*Ca
    if abs(CL)>abs(CR) and abs(Cc)>abs(Cb):
        return Pm_ENO2(u2,u3,u4,u5,dx)-(dx**2)*Cb
    else:
        return Pm_ENO2(u2,u3,u4,u5,dx)-(dx**2)*Cc
        
def Pp_ENO3(u1,u2,u3,u4,u5,u6,dx):
# For p+ : u1=u[i-2] , u2=u[i-1] , u3=u[i] , u4=u[i+1] ,u5=u[i+2] , u6=u[i+3]
    Ca=(u4-3*u3+3*u2-u1)/(6*(dx**3))
    Cb=(u6-3*u5+3*u4-u3)/(6*(dx**3))
    Cc=(u5-3*u4+3*u3-u2)/(6*(dx**3))
    CL=(u4-2*u3+u2)/(2*(dx**2))
    CR=(u5-2*u4+u3)/(2*(dx**2))
    if abs(CL)<abs(CR) and abs(Ca)<abs(Cc):
        return Pp_ENO2(u2,u3,u4,u5,dx)-(dx**2)*Ca
    if abs(CL)>abs(CR) and abs(Cc)>abs(Cb):
        return Pp_ENO2(u2,u3,u4,u5,dx)+2*(dx**2)*Cb
    else:
            return Pp_ENO2(u2,u3,u4,u5,dx)+2*(dx**2)*Cc
        

#************************ WENO Schemes ********************************
def phi_WENO(a,b,c,d,epsilon):
    ISa=13*(a-b)**2+3*(a-3*b)**2
    ISb=13*(c-d)**2+3*(3*c-d)**2
    ISc=13*(b-c)**2+3*(b+c)**2
    alpha_a=1/((epsilon+ISa)**2)
    alpha_b=3/((epsilon+ISb)**2)
    alpha_c=6/((epsilon+ISc)**2)
    omega_a=alpha_a/(alpha_a+alpha_b+alpha_c)
    omega_b=alpha_b/(alpha_a+alpha_b+alpha_c)
    
    return omega_a*(a-2*b+c)/3+(omega_b-0.5)*(b-2*c+d)/6

def Pm_WENO5(u1,u2,u3,u4,u5,u6,dx):
# For p- : u1=u[i-3] , u2=u[i-2] , u3=u[i-1] , u4=u[i] ,u5=u[i+1] , u6=u[i+2]
    p1=P_deg1(u1,u2,dx)
    p2=P_deg1(u2,u3,dx)
    p3=P_deg1(u3,u4,dx)
    p4=P_deg1(u4,u5,dx)
    p5=P_deg1(u5,u6,dx)
    a=p2-p1
    b=p3-p2
    c=p4-p3
    d=p5-p4
    return (-p2+7*p3+7*p4-p5)/12-phi_WENO(a,b,c,d,1.e-6)

def Pp_WENO5(u1,u2,u3,u4,u5,u6,dx):
# For p+ : u1=u[i-2] , u2=u[i-1] , u3=u[i] , u4=u[i+1] ,u5=u[i+2] , u6=u[i+3]
    p1=P_deg1(u1,u2,dx)
    p2=P_deg1(u2,u3,dx)
    p3=P_deg1(u3,u4,dx)
    p4=P_deg1(u4,u5,dx)
    p5=P_deg1(u5,u6,dx)
    a=p5-p4
    b=p4-p3
    c=p3-p2
    d=p2-p1
    return (-p1+7*p2+7*p3-p4)/12+phi_WENO(a,b,c,d,1.e-6)


# ## Numerical Schemes with errors

# In[4]:


## arg=[a,b,tf,Initial_condition,exact_solution,H,dpH]
def frst_schemes_error(u1, u2, u3, nx, dx, dt, grid, error1, error2, error3, arg, order):
    a=arg[0]; b=arg[1]; tf=arg[2]
    Initial_condition=arg[3];exact_solution=arg[4]
    H=arg[5]; dpH=arg[6]
    t=0.0
    un1 = np.zeros(nx)
    un2 = np.zeros(nx)
    un3 = np.zeros(nx)
    uex = np.zeros(nx)
    for i in range(len(grid)):
        uex[i]=Initial_condition(grid[i])
    
    while True:
        if t!=0.0:
            uex = np.zeros(nx)
            for i in range(len(grid)):
                uex[i] = exact_solution(grid[i],t)
        
        for i in range(nx):
            un1[i] = u1[i]
            un2[i] = u2[i]
            un3[i] = u3[i]
            
        for i in range(1,nx-1):
            # p1:p- , p2:p+
            p1_lax= P_deg1(un1[i-1],un1[i],dx)
            p2_lax= P_deg1(un1[i],un1[i+1],dx)
            p1_go= P_deg1(un2[i-1],un2[i],dx)
            p2_go= P_deg1(un2[i],un2[i+1],dx)
            p1_roe= P_deg1(un3[i-1],un3[i],dx)
            p2_roe= P_deg1(un3[i],un3[i+1],dx)

            alpha_x1=max_abs_dpH(p1_lax,p2_lax,dpH)
            alpha_x3=max_abs_dpH(p1_roe,p2_roe,dpH)
            
            u1[i]=un1[i]-dt*H_lax(p1_lax,p2_lax,alpha_x1,H)
            u2[i]=un2[i]-dt*H_godunov(p1_go,p2_go,H)
            u3[i]=un3[i]-dt*H_Roe(p1_roe,p2_roe,alpha_x3,H,dpH)
        
#         # Neumann conditions 
#         u1[0]=u1[1]
#         u1[nx-1]=u1[nx-2]
#         u2[0]=u2[1]
#         u2[nx-1]=u2[nx-2]
#         u3[0]=u3[1]
#         u3[nx-1]=u3[nx-2]
        
        err1= round(np.linalg.norm(uex-u1,ord=order)/np.linalg.norm(uex,ord=order) ,4)
        err2= round(np.linalg.norm(uex-u2,ord=order)/np.linalg.norm(uex,ord=order) ,4)
        err3= round(np.linalg.norm(uex-u3,ord=order)/np.linalg.norm(uex,ord=order) ,4)
        
        error1.append(err1)
        error2.append(err2)
        error3.append(err3)
        
        t += dt
        t = min(t,tf)
        if t >= tf:
            break  
    return 0

def TVD_deg2_error(v1, v2, v3, nx, dx, dt, grid, error1_TVD2, error2_TVD2, error3_TVD2, arg, order):
    a=arg[0]; b=arg[1]; tf=arg[2]
    Initial_condition=arg[3];exact_solution=arg[4]
    H=arg[5]; dpH=arg[6]
    t=0.0
    vn1 = np.zeros(nx)
    vn2 = np.zeros(nx)
    vn3 = np.zeros(nx)
    vex = np.zeros(nx)
    for i in range(len(grid)):
        vex[i]=Initial_condition(grid[i])
    
    while True:
        if t!=0.0:
            vex = np.zeros(nx)
            for i in range(len(grid)):
                vex[i] = exact_solution(grid[i],t)
                
        for i in range(nx):
            vn1[i] = v1[i]
            vn2[i] = v2[i]
            vn3[i] = v3[i]
            
        for i in range(1,nx-1):
            p_lax= P_TVD2(vn1[i-1],vn1[i+1],dx)
            p_go= P_TVD2(vn2[i-1],vn2[i+1],dx)
            p_roe= P_TVD2(vn3[i-1],vn3[i+1],dx)

            alpha_x1=abs(dpH(p_lax))
            alpha_x3=abs(dpH(p_roe))
            
            v1[i]=vn1[i]-dt*H_lax(p_lax,p_lax,alpha_x1,H)
            v2[i]=vn2[i]-dt*H_godunov(p_go,p_go,H)
            v3[i]=vn3[i]-dt*H_Roe(p_roe,p_roe,alpha_x3,H,dpH)
        
        err1= round(np.linalg.norm(vex-v1,ord=order)/np.linalg.norm(vex,ord=order) ,4)  
        err2= round(np.linalg.norm(vex-v2,ord=order)/np.linalg.norm(vex,ord=order) ,4)
        err3= round(np.linalg.norm(vex-v3,ord=order)/np.linalg.norm(vex,ord=order) ,4)
        
        error1_TVD2.append(err1)
        error2_TVD2.append(err2)
        error3_TVD2.append(err3)
        
        t += dt
        t = min(t,tf)
        if t >= tf:
            break  
    return 0

def TVD_deg3_error(w1, w2, w3, nx, dx, dt, grid, error1_TVD3, error2_TVD3, error3_TVD3, arg, order):
    a=arg[0]; b=arg[1]; tf=arg[2]
    Initial_condition=arg[3];exact_solution=arg[4]
    H=arg[5]; dpH=arg[6]
    t=0.0
    wn1 = np.zeros(nx)
    wn2 = np.zeros(nx)
    wn3 = np.zeros(nx)
    wex = np.zeros(nx)
    for i in range(len(grid)):
        wex[i]=Initial_condition(grid[i])
    
    while True:
        if t!=0.0:
            wex = np.zeros(nx)
            for i in range(len(grid)):
                wex[i] = exact_solution(grid[i],t)
                
        for i in range(nx):
            wn1[i] = w1[i]
            wn2[i] = w2[i]
            wn3[i] = w3[i]

        for i in range(2,nx-2):
            p1_lax= P_TVD3(wn1[i-2],wn1[i-1],wn1[i],wn1[i+1],dx)
            p2_lax= P_TVD3(-wn1[i+2],-wn1[i+1],-wn1[i],-wn1[i-1],dx)
            p1_go= P_TVD3(wn2[i-2],wn2[i-1],wn2[i],wn2[i+1],dx)
            p2_go= P_TVD3(-wn2[i+2],-wn2[i+1],-wn2[i],-wn2[i-1],dx)
            p1_roe= P_TVD3(wn3[i-2],wn3[i-1],wn3[i],wn3[i+1],dx)
            p2_roe= P_TVD3(-wn3[i+2],-wn3[i+1],-wn3[i],-wn3[i-1],dx)

            alpha_x1=max_abs_dpH(p1_lax,p2_lax,dpH)
            alpha_x3=max_abs_dpH(p1_roe,p2_roe,dpH)
            
            w1[i]=wn1[i]-dt*H_lax(p1_lax,p2_lax,alpha_x1,H)
            w2[i]=wn2[i]-dt*H_godunov(p1_go,p2_go,H)
            w3[i]=wn3[i]-dt*H_Roe(p1_roe,p2_roe,alpha_x3,H,dpH)
        
        err1= round(np.linalg.norm(wex-w1,ord=order)/np.linalg.norm(wex,ord=order) ,4) 
        err2= round(np.linalg.norm(wex-w2,ord=order)/np.linalg.norm(wex,ord=order) ,4)
        err3= round(np.linalg.norm(wex-w3,ord=order)/np.linalg.norm(wex,ord=order) ,4)
        
        error1_TVD3.append(err1)
        error2_TVD3.append(err2)
        error3_TVD3.append(err3)
        
        t += dt
        t = min(t,tf)
        if t >= tf:
            break  
    return 0

def ENO_deg2_error(y1, y2, y3, nx, dx, dt, grid, error1_ENO2, error2_ENO2, error3_ENO2, arg, order):
    a=arg[0]; b=arg[1]; tf=arg[2]
    Initial_condition=arg[3];exact_solution=arg[4]
    H=arg[5]; dpH=arg[6]
    t=0.0
    yn1 = np.zeros(nx)
    yn2 = np.zeros(nx)
    yn3 = np.zeros(nx)
    yex = np.zeros(nx)
    for i in range(len(grid)):
        yex[i]=Initial_condition(grid[i])
    
    while True:
        if t!=0.0:
            yex = np.zeros(nx)
            for i in range(len(grid)):
                yex[i] = exact_solution(grid[i],t)
        
        for i in range(nx):
            yn1[i] = y1[i]
            yn2[i] = y2[i]
            yn3[i] = y3[i]

        for i in range(2,nx-2):
            p1_lax= Pm_ENO2(yn1[i-2],yn1[i-1],yn1[i],yn1[i+1],dx)
            p2_lax= Pp_ENO2(yn1[i-1],yn1[i],yn1[i+1],yn1[i+2],dx)
            p1_go= Pm_ENO2(yn2[i-2],yn2[i-1],yn2[i],yn2[i+1],dx)
            p2_go= Pp_ENO2(yn2[i-1],yn2[i],yn2[i+1],yn2[i+2],dx)
            p1_roe= Pm_ENO2(yn3[i-2],yn3[i-1],yn3[i],yn3[i+1],dx)
            p2_roe= Pp_ENO2(yn3[i-1],yn3[i],yn3[i+1],yn3[i+2],dx)

            alpha_x1=max_abs_dpH(p1_lax,p2_lax,dpH)
            alpha_x3=max_abs_dpH(p1_roe,p2_roe,dpH)
            
            y1[i]=yn1[i]-dt*H_lax(p1_lax,p2_lax,alpha_x1,H)
            y2[i]=yn2[i]-dt*H_godunov(p1_go,p2_go,H)
            y3[i]=yn3[i]-dt*H_Roe(p1_roe,p2_roe,alpha_x3,H,dpH)
            
        
        err1= round(np.linalg.norm(yex-y1,ord=order)/np.linalg.norm(yex,ord=order) ,4)  
        err2= round(np.linalg.norm(yex-y2,ord=order)/np.linalg.norm(yex,ord=order) ,4)
        err3= round(np.linalg.norm(yex-y3,ord=order)/np.linalg.norm(yex,ord=order) ,4)
        
        error1_ENO2.append(err1)
        error2_ENO2.append(err2)
        error3_ENO2.append(err3)
        
        t += dt
        t = min(t,tf)
        if t >= tf:
            break  
    return 0

def ENO_deg3_error(z1, z2, z3, nx, dx, dt, grid, error1_ENO3, error2_ENO3, error3_ENO3, arg, order):
    a=arg[0]; b=arg[1]; tf=arg[2]
    Initial_condition=arg[3];exact_solution=arg[4]
    H=arg[5]; dpH=arg[6]
    t=0.0
    zn1 = np.zeros(nx)
    zn2 = np.zeros(nx)
    zn3 = np.zeros(nx)
    zex = np.zeros(nx)
    for i in range(len(grid)):
        zex[i]=Initial_condition(grid[i])
    
    while True:
        if t!=0.0:
            zex = np.zeros(nx)
            for i in range(len(grid)):
                zex[i] = exact_solution(grid[i],t)
                
        for i in range(nx):
            zn1[i] = z1[i]
            zn2[i] = z2[i]
            zn3[i] = z3[i]

        for i in range(3,nx-3):
            p1_lax= Pm_ENO3(zn1[i-3],zn1[i-2],zn1[i-1],zn1[i],zn1[i+1],zn1[i+2],dx)
            p2_lax= Pp_ENO3(zn1[i-2],zn1[i-1],zn1[i],zn1[i+1],zn1[i+2],zn1[i+3],dx)
            p1_go= Pm_ENO3(zn2[i-3],zn2[i-2],zn2[i-1],zn2[i],zn2[i+1],zn2[i+2],dx)
            p2_go= Pp_ENO3(zn2[i-2],zn2[i-1],zn2[i],zn2[i+1],zn2[i+2],zn2[i+3],dx)
            p1_roe= Pm_ENO3(zn3[i-3],zn3[i-2],zn3[i-1],zn3[i],zn3[i+1],zn3[i+2],dx)
            p2_roe= Pp_ENO3(zn3[i-2],zn3[i-1],zn3[i],zn3[i+1],zn3[i+2],zn3[i+3],dx)
            
            alpha_x1=max_abs_dpH(p1_lax,p2_lax,dpH)
            alpha_x3=max_abs_dpH(p1_roe,p2_roe,dpH)
            
            z1[i]=zn1[i]-dt*H_lax(p1_lax,p2_lax,alpha_x1,H)
            z2[i]=zn2[i]-dt*H_godunov(p1_go,p2_go,H)
            z3[i]=zn3[i]-dt*H_Roe(p1_roe,p2_roe,alpha_x3,H,dpH)
        
        
        err1= round(np.linalg.norm(zex-z1,ord=order)/np.linalg.norm(zex,ord=order) ,4) 
        err2= round(np.linalg.norm(zex-z2,ord=order)/np.linalg.norm(zex,ord=order) ,4)
        err3= round(np.linalg.norm(zex-z3,ord=order)/np.linalg.norm(zex,ord=order) ,4)
        
        error1_ENO3.append(err1)
        error2_ENO3.append(err2)
        error3_ENO3.append(err3)
        
        t += dt
        t = min(t,tf)
        if t >= tf:
            break  
    return 0

def WENO_deg5_error(s1, s2, s3, nx, dx, dt, grid, error1_WENO5, error2_WENO5, error3_WENO5, arg, order):
    a=arg[0]; b=arg[1]; tf=arg[2]
    Initial_condition=arg[3];exact_solution=arg[4]
    H=arg[5]; dpH=arg[6]
    t=0.0
    sn1 = np.zeros(nx)
    sn2 = np.zeros(nx)
    sn3 = np.zeros(nx)
    sex = np.zeros(nx)
    for i in range(len(grid)):
        sex[i]=Initial_condition(grid[i])
    
    while True:
        if t!=0.0:
            sex = np.zeros(nx)
            for i in range(len(grid)):
                sex[i] = exact_solution(grid[i],t)
                
        for i in range(nx):
            sn1[i] = s1[i]
            sn2[i] = s2[i]
            sn3[i] = s3[i]
            
        for i in range(3,nx-3):
            p1_lax= Pm_WENO5(sn1[i-3],sn1[i-2],sn1[i-1],sn1[i],sn1[i+1],sn1[i+2],dx)
            p2_lax= Pp_WENO5(sn1[i-2],sn1[i-1],sn1[i],sn1[i+1],sn1[i+2],sn1[i+3],dx)
            p1_go= Pm_WENO5(sn2[i-3],sn2[i-2],sn2[i-1],sn2[i],sn2[i+1],sn2[i+2],dx)
            p2_go= Pp_WENO5(sn2[i-2],sn2[i-1],sn2[i],sn2[i+1],sn2[i+2],sn2[i+3],dx)
            p1_roe= Pm_WENO5(sn3[i-3],sn3[i-2],sn3[i-1],sn3[i],sn3[i+1],sn3[i+2],dx)
            p2_roe= Pp_WENO5(sn3[i-2],sn3[i-1],sn3[i],sn3[i+1],sn3[i+2],sn3[i+3],dx)
            
            alpha_x1=max_abs_dpH(p1_lax,p2_lax,dpH)
            alpha_x3=max_abs_dpH(p1_roe,p2_roe,dpH)
            
            s1[i]=sn1[i]-dt*H_lax(p1_lax,p2_lax,alpha_x1,H)
            s2[i]=sn2[i]-dt*H_godunov(p1_go,p2_go,H)
            s3[i]=sn3[i]-dt*H_Roe(p1_roe,p2_roe,alpha_x3,H,dpH)
        
        err1= round(np.linalg.norm(sex-s1,ord=order)/np.linalg.norm(sex,ord=order) ,4)  
        err2= round(np.linalg.norm(sex-s2,ord=order)/np.linalg.norm(sex,ord=order) ,4)
        err3= round(np.linalg.norm(sex-s3,ord=order)/np.linalg.norm(sex,ord=order) ,4)
        
        error1_WENO5.append(err1)
        error2_WENO5.append(err2)
        error3_WENO5.append(err3)
        
        t += dt
        t = min(t,tf)
        if t >= tf:
            break  
    return 0


# ## Plot functions with errors

# In[5]:


## Errors based on Number of points

def Error_nbpoints(Nx,dt,fct,arg,text,order):
    error_nx1=[]
    error_nx2=[]
    error_nx3=[]
    a=arg[0]; b=arg[1]; tf=arg[2]
    Initial_condition=arg[3];exact_solution=arg[4]
    h=[]
    for nx in Nx:
        dx = (b-a) / (nx-1)
        h.append(round(dx,3))
        grid = np.linspace(a,b,nx) # creating the space grid         
        u0 = np.zeros(nx)
        uex = np.zeros(nx)
        for i in range(len(grid)):
            u0[i] = Initial_condition(grid[i]) 
            uex[i] = exact_solution(grid[i],tf)

        u1 = u0.copy()
        u2 = u0.copy()
        u3 = u0.copy()

        error1=[]
        error2=[]
        error3=[]

        fct(u1, u2, u3, nx, dx, dt, grid, error1, error2, error3, arg, order)

        error_nx1.append(error1[len(error1)-1])
        error_nx2.append(error2[len(error2)-1])
        error_nx3.append(error3[len(error3)-1])
    plt.figure()
    plt.plot(Nx,error_nx1,':b',label='LLF')
    plt.plot(Nx,error_nx2,'-.r',label='Godunov')
    plt.plot(Nx,error_nx3,'--g',label='Roe')
    plt.grid()
    plt.legend()
    plt.title("{Text}, L_{order}=f(nx) at {t}".format(Text=text,order=order,t=round(tf,2)))
    plt.xlabel('nx')
    plt.ylabel('Error')
    
    n=len(Nx)
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    data =np.zeros((n,8))
    data[:,0]=Nx
    data[:,1]=h
    data[:,2]=error_nx1
    for j in range(1,n):
        data[j,3]= round(math.log(data[j,2]/data[j-1,2])/math.log(data[j,1]/data[j-1,1]),4)
    data[:,4]=error_nx2
    for j in range(1,n):
        data[j,5]= round(math.log(data[j,4]/data[j-1,4])/math.log(data[j,1]/data[j-1,1]),4)
    data[:,6]=error_nx3
    for j in range(1,n):
        data[j,7]= round(math.log(data[j,6]/data[j-1,6])/math.log(data[j,1]/data[j-1,1]),4)
    collabel=("Number-points","dx", "Error-LLF", "Order-LLF", "Error-Godunov", "Order-Godunov", "Error-Roe", "Order-Roe")
    plt.table(cellText=data,colWidths=[0.4] * 8,colLabels=collabel,loc='center')
    plt.tight_layout()

    plt.show()
    
    return 0


def solutionAll_error(nx,dt,fct,arg,text,order):
    a=arg[0]; b=arg[1]; tf=arg[2]
    Initial_condition=arg[3];exact_solution=arg[4]
    
    dx = (b-a) / (nx-1)
    t = np.arange(0,tf,dt) 
    grid = np.linspace(a,b,nx) # creating the space grid  
    u0 = np.zeros(nx)
    uex = np.zeros(nx)
    for i in range(len(grid)):
        u0[i] = Initial_condition(grid[i]) 
        uex[i] = exact_solution(grid[i],tf)

    u1 = u0.copy()
    u2 = u0.copy()
    u3 = u0.copy()

    error1=[]
    error2=[]
    error3=[]

    e1 = np.zeros(nx)
    e2 = np.zeros(nx)
    e3 = np.zeros(nx)

    ## Outputs
    fct(u1, u2, u3, nx, dx, dt, grid, error1, error2, error3, arg, order)

    for i in range(len(grid)):
        e1[i]=abs(uex[i]-u1[i])
        e2[i]=abs(uex[i]-u2[i])
        e3[i]=abs(uex[i]-u3[i])


    #Plotting data
    plt.figure(figsize=(20, 20))
    plt.subplot(3,2,1)
    plt.plot(grid,uex,':b',label='Viscosity solution')
    plt.plot(grid,u1,'--r',label='LLF')
    plt.legend()
    #print(u1)
    plt.grid()
    plt.title("{Text}, LLF, {N} points".format(Text=text,N=nx))
    plt.xlabel('X')
    plt.ylabel('U')
    plt.subplot(3,2,2)
    plt.plot(grid,uex,':b',label='Viscosity solution')
    plt.plot(grid,u2,'--r',label='Godunov')
    plt.legend()
    #print(u2)
    plt.grid()
    plt.title("{Text}, Godunov, {N} points".format(Text=text,N=nx))
    plt.xlabel('X')
    plt.ylabel('U')
    plt.subplot(3,2,3)
    plt.plot(grid,uex,':b',label='Viscosity solution')
    plt.plot(grid,u3,'--r',label='Roe')
    plt.legend()
    #print(u2)
    plt.grid()
    plt.title("{Text}, Roe, {N} points".format(Text=text,N=nx))
    plt.xlabel('X')
    plt.ylabel('U')
    plt.subplot(3,2,4)
    plt.plot(grid,uex,':',label='Viscosity solution')
    plt.plot(grid,u1,'-.b',label='LLF')
    plt.plot(grid,u2,'+r',label='Godunov')
    plt.plot(grid,u3,'--g',label='Roe')
    plt.legend()
    #print(u2)
    plt.grid()
    plt.title("{Text}, {N} points".format(Text=text,N=nx))
    plt.xlabel('X')
    plt.ylabel('U')
    plt.subplot(3,2,5)
    plt.plot(t[1:],error1[1:],':',label='Error-LLF')
    plt.plot(t[1:],error2[1:],'-.',label='Error_Godunov')
    plt.plot(t[1:],error3[1:],'--',label='Error_Roe')
    plt.legend()
    #print(u2)
    plt.grid()
    plt.title("{Text}, L_{order} Errors in time with {N} points".format(Text=text,order=order,N=nx)) 
    plt.xlabel('t')
    plt.ylabel('Error') 
    plt.subplot(3,2,6)
    plt.plot(grid,e1,':',label='Error-LLF')
    plt.plot(grid,e2,'-.',label='Error_Godunov')
    plt.plot(grid,e3,'--',label='Error_Roe')
    plt.legend()
    #print(u2)
    plt.grid()
    plt.title("{Text}, L_{order} Errors in space at {t} sec".format(Text=text,order=order,t=round(tf,2)))
    #plt.ylim([-1.5,1.5])
    plt.xlabel('X')
    plt.ylabel('Error') 
    return 0


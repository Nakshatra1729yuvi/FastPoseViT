import numpy as np 
from scipy.spatial.transform import Rotation as R

def get_target(q,T,x_min,y_min,w,h,K):
    bx=x_min+w/2
    by=y_min+h/2

    q_scipy=[q[1],q[2],q[3],q[0]]
    R_mat=R.from_quat(q_scipy).as_matrix()


    fx,fy=K[0,0],K[1,1]
    cx,cy=K[0,2],K[1,2]

    X,Y,Z=T

    x=fx*X/Z+cx
    y=fy*Y/Z+cy

    Ux=(x-bx)/w
    Uy=(y-by)/h

    W,H=1920,1200

    alpha=1.6

    sx=W/(alpha*w)
    sy=H/(h)

    Uz=0.5*(1/sx+1/sy)*Z


    T_vec=np.array([X,Y,Z])
    T_hat=T_vec/(np.linalg.norm(T_vec)+1e-8)

    ez=np.array([0,0,1])

    u=np.cross(T_hat,ez)

    norm_u=np.linalg.norm(u)

    if norm_u<1e-6:
        delta_R=np.eye(3)

    else:
        u=u/norm_u
        theta=np.arccos(np.clip(np.dot(T_hat,ez),-1,1))

        skew_symm_m=np.array([
            [0, -u[2], u[1]],
            [u[2], 0, -u[0]],
            [-u[1], u[0], 0]
        ])

        delta_R=np.eye(3)+np.sin(theta)*skew_symm_m+(1-np.cos(theta))*(skew_symm_m@skew_symm_m)
    
    R_prime=delta_R@R_mat

    r1_hat=R_prime[:,0]
    r2_hat=R_prime[:,1]

    target=np.concatenate([[Ux,Uy,Uz],r1_hat,r2_hat])

    return target


def get_inference(pred,x_min,y_min,w,h,K):

    bx=x_min+w/2
    by=y_min+h/2

    fx,fy=K[0,0],K[1,1]
    cx,cy=K[0,2],K[1,2]

    W,H=1920,1200

    alpha=1.6

    Ux,Uy,Uz=pred[:3]
    r1_hat=pred[3:6]
    r2_hat=pred[6:9] 

    sx=W/(alpha*w)
    sy=H/h

    Z=Uz/(0.5*(1/sx+1/sy))

    X=(bx+Ux*w-cx)*Z/fx
    Y=(by+Uy*h-cy)*Z/fy

    r1=r1_hat/(np.linalg.norm(r1_hat)+1e-8)
    r2=r2_hat-np.dot(r1,r2_hat)*r1
    r2=r2/(np.linalg.norm(r2)+1e-8)
    r3=np.cross(r1,r2)
    R_prime=np.stack([r1,r2,r3],axis=1)


    T_vec=np.array([X,Y,Z])
    T_hat=T_vec/(np.linalg.norm(T_vec)+1e-8)

    ez=np.array([0,0,1])

    u=np.cross(T_hat,ez)

    norm_u=np.linalg.norm(u)

    if norm_u<1e-6:
        delta_R=np.eye(3)

    else:
        u=u/norm_u
        theta=np.arccos(np.clip(np.dot(T_hat,ez),-1,1))

        skew_symm_m=np.array([
            [0, -u[2], u[1]],
            [u[2], 0, -u[0]],
            [-u[1], u[0], 0]
        ])

        delta_R=np.eye(3)+np.sin(theta)*skew_symm_m+(1-np.cos(theta))*(skew_symm_m@skew_symm_m)

    R_pred=delta_R.T@R_prime

    return T_vec,R_pred





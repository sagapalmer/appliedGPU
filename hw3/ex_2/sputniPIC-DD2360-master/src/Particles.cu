#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover


__global__ void mover_PC_gpu_kernel(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param,
                                    FPpart *d_x, FPpart* d_y, FPpart* d_z, FPpart* d_u, FPpart* d_v, FPpart* d_w,
                                    FPfield* d_Ex_flat, FPfield* d_Ey_flat, FPfield* d_Ez_flat, FPfield* d_Bxn_flat, FPfield* d_Byn_flat, 
                                    FPfield* d_Bzn_flat, FPfield* d_XN_flat, FPfield* d_YN_flat, FPfield* d_ZN_flat) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nxn = grd->nxn;
    int nyn = grd->nyn;

    if (i >= part->nop) {
        return;
    }
        
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    //FPfield weight[8];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    for (int i_sub = 0; i_sub < part->n_sub_cycles; i_sub++) {
        xptilde = d_x[i];
        yptilde = d_y[i];
        zptilde = d_z[i];
        // calculate the average velocity iteratively
        for(int innter=0; innter < part->NiterMover; innter++){
            // interpolation G-->P
            ix = 2 +  int((d_x[i] - grd->xStart)*grd->invdx);
            iy = 2 +  int((d_y[i] - grd->yStart)*grd->invdy);
            iz = 2 +  int((d_z[i] - grd->zStart)*grd->invdz);
            
            // calculate weights
            // xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
            // eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
            // zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
            // xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
            // eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
            // zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
            xi[0]   = d_x[i] - d_XN_flat[(ix-1) + nxn * (iy + nyn * iz)];
            eta[0]  = d_y[i] - d_YN_flat[ix + nxn * ((iy-1) + nyn * iz)];
            zeta[0] = d_z[i] - d_ZN_flat[ix + nxn * (iy + nyn * (iz-1))];
            xi[1]   = d_XN_flat[ix + nxn * (iy + nyn * iz)] - d_x[i];
            eta[1]  = d_YN_flat[ix + nxn * (iy + nyn * iz)] - d_y[i];
            zeta[1] = d_ZN_flat[ix + nxn * (iy + nyn * iz)] - d_z[i];
            for (int ii = 0; ii < 2; ii++) {
                for (int jj = 0; jj < 2; jj++) {
                    for (int kk = 0; kk < 2; kk++) {
                        weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                    }
                }
            }
            
            // set to zero local electric and magnetic field
            Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
            
            for (int ii=0; ii < 2; ii++)
                for (int jj=0; jj < 2; jj++)
                    for(int kk=0; kk < 2; kk++){
                        int index = (ix- ii) + nxn * ((iy -jj) + nyn * (iz- kk));
                        // Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                        // Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                        // Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                        // Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                        // Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                        // Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        Exl += weight[ii][jj][kk]*d_Ex_flat[index];
                        Eyl += weight[ii][jj][kk]*d_Ey_flat[index];
                        Ezl += weight[ii][jj][kk]*d_Ez_flat[index];
                        Bxl += weight[ii][jj][kk]*d_Bxn_flat[index];
                        Byl += weight[ii][jj][kk]*d_Byn_flat[index];
                        Bzl += weight[ii][jj][kk]*d_Bzn_flat[index];
                    }
            
            // end interpolation
            omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
            denom = 1.0/(1.0 + omdtsq);
            // solve the position equation
            ut= d_u[i] + qomdt2*Exl;
            vt= d_v[i] + qomdt2*Eyl;
            wt= d_w[i] + qomdt2*Ezl;
            udotb = ut*Bxl + vt*Byl + wt*Bzl;
            // solve the velocity equation
            uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
            vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
            wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
            // update position
            d_x[i] = xptilde + uptilde*dto2;
            d_y[i] = yptilde + vptilde*dto2;
            d_z[i] = zptilde + wptilde*dto2;
            
            
        } // end of iteration
        // update the final position and velocity
        d_u[i]= 2.0*uptilde - d_u[i];
        d_v[i]= 2.0*vptilde - d_v[i];
        d_w[i]= 2.0*wptilde - d_w[i];
        d_x[i] = xptilde + uptilde*dt_sub_cycling;
        d_y[i] = yptilde + vptilde*dt_sub_cycling;
        d_z[i] = zptilde + wptilde*dt_sub_cycling;
        
        
        //////////
        //////////
        ////////// BC
                                    
        // X-DIRECTION: BC particles
        if (d_x[i] > grd->Lx){
            if (param->PERIODICX==true){ // PERIODIC
                d_x[i] = d_x[i] - grd->Lx;
            } else { // REFLECTING BC
                d_u[i] = -d_u[i];
                d_x[i] = 2*grd->Lx - d_x[i];
            }
        }
                                                                    
        if (d_x[i] < 0){
            if (param->PERIODICX==true){ // PERIODIC
                d_x[i] = d_x[i] + grd->Lx;
            } else { // REFLECTING BC
                d_u[i] = -d_u[i];
                d_x[i] = -d_x[i];
            }
        }
            
        
        // Y-DIRECTION: BC particles
        if (d_y[i] > grd->Ly){
            if (param->PERIODICY==true){ // PERIODIC
                d_y[i] = d_y[i] - grd->Ly;
            } else { // REFLECTING BC
                d_v[i] = -d_v[i];
                d_y[i] = 2*grd->Ly - d_y[i];
            }
        }
                                                                    
        if (d_y[i] < 0){
            if (param->PERIODICY==true){ // PERIODIC
                d_y[i] = d_y[i] + grd->Ly;
            } else { // REFLECTING BC
                d_v[i] = -d_v[i];
                d_y[i] = -d_y[i];
            }
        }
                                                                    
        // Z-DIRECTION: BC particles
        if (d_z[i] > grd->Lz){
            if (param->PERIODICZ==true){ // PERIODIC
                d_z[i] = d_z[i] - grd->Lz;
            } else { // REFLECTING BC
                d_w[i] = -d_w[i];
                d_z[i] = 2*grd->Lz - d_z[i];
            }
        }
                                                                    
        if (d_z[i] < 0){
            if (param->PERIODICZ==true){ // PERIODIC
                d_z[i] = d_z[i] + grd->Lz;
            } else { // REFLECTING BC
                d_w[i] = -d_w[i];
                d_z[i] = -d_z[i];
            }
        }
        
    }

}



int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param) {

    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    //@@ declaring variables

    // PART
    struct particles* d_part;
    cudaMalloc(&d_part, sizeof(struct particles));
    cudaMemcpy(d_part, part, sizeof(struct particles), cudaMemcpyHostToDevice);

    struct EMfield* d_field;
    cudaMalloc(&d_field, sizeof(struct EMfield));
    cudaMemcpy(d_field, field, sizeof(struct EMfield), cudaMemcpyHostToDevice);

    struct grid* d_grd;
    cudaMalloc(&d_grd, sizeof(struct grid));
    cudaMemcpy(d_grd, grd, sizeof(struct grid), cudaMemcpyHostToDevice);

    struct parameters* d_param;
    cudaMalloc(&d_param, sizeof(struct parameters));
    cudaMemcpy(d_param, param, sizeof(struct parameters), cudaMemcpyHostToDevice);
    
    
    //Allocating d_x, d_y, d_z, d_u, d_v, d_w
    FPpart *d_x, *d_y, *d_z, *d_u, *d_v, *d_w;
    cudaMalloc(&d_x, part->npmax * sizeof(FPpart));
    cudaMalloc(&d_y, part->npmax * sizeof(FPpart));
    cudaMalloc(&d_z, part->npmax * sizeof(FPpart));
    cudaMalloc(&d_u, part->npmax * sizeof(FPpart));
    cudaMalloc(&d_v, part->npmax * sizeof(FPpart));
    cudaMalloc(&d_w, part->npmax * sizeof(FPpart));

    //copying d_x, d_y, d_z, d_u, d_v, d_w
    cudaMemcpy(d_x, part->x, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, part->y, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, part->z, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, part->u, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, part->v, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, part->w, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    printf("d_part allocation done!\n");
    
    FPfield *d_Ex_flat, *d_Ey_flat, *d_Ez_flat;
    FPfield *d_Bxn_flat, *d_Byn_flat, *d_Bzn_flat;
    cudaMalloc(&d_Ex_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMalloc(&d_Ey_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMalloc(&d_Ez_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMalloc(&d_Bxn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMalloc(&d_Byn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMalloc(&d_Bzn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);

    FPfield *h_Ex_flat, *h_Ey_flat, *h_Ez_flat;
    FPfield *h_Bxn_flat, *h_Byn_flat, *h_Bzn_flat;
    h_Ex_flat = newArr1<FPfield>(grd->nxn * grd->nyn * grd->nzn); //another argument?
    h_Ey_flat = newArr1<FPfield>(grd->nxn * grd->nyn * grd->nzn);
    h_Ez_flat = newArr1<FPfield>(grd->nxn * grd->nyn * grd->nzn);
    h_Bxn_flat = newArr1<FPfield>(grd->nxn * grd->nyn * grd->nzn);
    h_Byn_flat = newArr1<FPfield>(grd->nxn * grd->nyn * grd->nzn);
    h_Bzn_flat = newArr1<FPfield>(grd->nxn * grd->nyn * grd->nzn);
    
    FPfield *d_XN_flat, *d_YN_flat, *d_ZN_flat;
    cudaMalloc(&d_XN_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMalloc(&d_YN_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMalloc(&d_ZN_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);

    FPfield *h_XN_flat, *h_YN_flat, *h_ZN_flat;
    h_XN_flat = newArr1<FPfield>(grd->nxn * grd->nyn * grd->nzn);
    h_YN_flat = newArr1<FPfield>(grd->nxn * grd->nyn * grd->nzn);
    h_ZN_flat = newArr1<FPfield>(grd->nxn * grd->nyn * grd->nzn);

    for (int i = 0; i < grd->nxn; ++i) {
        for (int j = 0; j < grd->nyn; ++j) {
            for (int k = 0; k < grd->nzn; ++k) {
                h_XN_flat[i * grd->nyn * grd->nzn + j * grd->nzn + k] = grd->XN[i][j][k];
                h_YN_flat[i * grd->nyn * grd->nzn + j * grd->nzn + k] = grd->YN[i][j][k];
                h_ZN_flat[i * grd->nyn * grd->nzn + j * grd->nzn + k] = grd->ZN[i][j][k];
                h_Ex_flat[i * grd->nyn * grd->nzn + j * grd->nzn + k] = field->Ex[i][j][k];
                h_Ey_flat[i * grd->nyn * grd->nzn + j * grd->nzn + k] = field->Ey[i][j][k];
                h_Ez_flat[i * grd->nyn * grd->nzn + j * grd->nzn + k] = field->Ez[i][j][k];
                h_Bxn_flat[i * grd->nyn * grd->nzn + j * grd->nzn + k] = field->Bxn[i][j][k];
                h_Byn_flat[i * grd->nyn * grd->nzn + j * grd->nzn + k] = field->Byn[i][j][k];
                h_Bzn_flat[i * grd->nyn * grd->nzn + j * grd->nzn + k] = field->Bzn[i][j][k];
            }
        }
    }

    cudaMemcpy(d_XN_flat, h_XN_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_YN_flat, h_YN_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ZN_flat, h_ZN_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);

    
    //@@ Initialize the grid and block dimensions
    int TPB = 256;
    int BLOCKS = (part->nop  + TPB - 1) / TPB;

    dim3 blockDim(TPB, 1, 1);
    dim3 gridDim(BLOCKS, 1, 1);

    
    //@@ Launch the GPU Kernel
    
    // start subcycling
    //for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
    printf("Launchin the kernel...\n");
        // move each particle with new fields
    mover_PC_gpu_kernel<<<gridDim, blockDim>>>(d_part, d_field, d_grd, d_param,
        d_x, d_y, d_z, d_u, d_v, d_w, d_Ex_flat, d_Ey_flat, d_Ez_flat, d_Bxn_flat, d_Byn_flat, 
        d_Bzn_flat, d_XN_flat, d_YN_flat, d_ZN_flat);
    cudaDeviceSynchronize();
    //}
    //printf("Done launching the kernel...\n");

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaStatus));
    }

    printf("Start copying back to CPU...\n");
    //@@ Copy the GPU memory back to the CPU
    cudaMemcpy(part->x, d_x, sizeof(FPpart) * part->npmax, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->y, d_y, sizeof(FPpart) * part->npmax, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->z, d_z, sizeof(FPpart) * part->npmax, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->u, d_u, sizeof(FPpart) * part->npmax, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->v, d_v, sizeof(FPpart) * part->npmax, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->w, d_w, sizeof(FPpart) * part->npmax, cudaMemcpyDeviceToHost);

    printf("Copying back to CPU finished!\n");
    //cudaMemcpy(part, d_part, sizeof(particles), cudaMemcpyDeviceToHost);

    //printf("Finished copying to CPU!\n");
    //@@ Free the GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_Ex_flat);
    cudaFree(d_Ey_flat); 
    cudaFree(d_Ez_flat);
    cudaFree(d_Bxn_flat); 
    cudaFree(d_Byn_flat); 
    cudaFree(d_Bzn_flat); 
    cudaFree(d_XN_flat); 
    cudaFree(d_YN_flat); 
    cudaFree(d_ZN_flat);
    cudaFree(d_part);
    cudaFree(d_field);
    cudaFree(d_grd);
    cudaFree(d_param);

    delArray1(h_Ex_flat); 
    delArray1(h_Ey_flat); 
    delArray1(h_Ez_flat); 
    delArray1(h_Bxn_flat); 
    delArray1(h_Byn_flat); 
    delArray1(h_Bzn_flat); 
    delArray1(h_XN_flat); 
    delArray1(h_YN_flat); 
    delArray1(h_ZN_flat);   

    //cudaFree(d_part);

    //printf("Freed memory\n");

    return(0);
}



/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}

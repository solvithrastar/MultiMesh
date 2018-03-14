//  Created by Dirk-Philip van Herwaarden on 4/21/17.
//  Copyright © 2017 Dirk-Philip van Herwaarden. All rights reserved.

#include <stdio.h>
#include <math.h>

// Global variables
const double mNodesR[] = {-1, -1, +1, +1, -1, +1, +1, -1};
const double mNodesS[] = {-1, +1, +1, -1, -1, -1, +1, +1};
const double mNodesT[] = {-1, -1, -1, -1, +1, +1, +1, +1};
const int nDim = 3;
const int nNodes = 8;

// Function definitions
void interpolateAtPoint(double pnt[], double *interpolator);
double referenceToElementMapping(double v0, double v1,
                                 double v2, double v3,
                                 double v4, double v5,
                                 double v6, double v7,
                                 double r, double s,
                                 double t);

void coordinateTransform(double pnt[], double vtx[8][3], double solution[]);
double dNdR(int N, double S, double T);
double dNdS(int N, double R, double T);
double dNdT(int N, double R, double S);
int inverseCoordinateTransform(double pnt[3], double vtx[8][3], double solution[3]);
void inverseJacobianAtPoint(double pnt[3], double vtx[8][3], double invJac[3][3]);

// Linear algebra helper functions
void dot_product_matrix_matrix(double Dn[3][8], double vtx[8][3], double jac[3][3]);
void dot_product_matrix_vector(double Jac_transpose[3][3],
                               double objective_function[3], double solution[3]);
double determinant(double m[3][3]);
void mInverse(double m[3][3], double minv[3][3], double det);
void transpose(double m[3][3], double mTranspose[3][3]);
int checkHull(double pnt[3], double vtx[8][3], double solution[3]);

// trilinear interpolation routine for hexahedral elements
long long int triLinearInterpolator(
    long long int nelem,                        // number of elements in mesh
        long long int nelem_to_search,          // number of elements to be tested for check hull for each point
        long long int npoints,                  // points to be interpolated
        long long int npoints_mesh,             // npoints_mesh
        long long int *nearest_element_indices, // shape [npoints, nelem_to_search] specifies row nr in connectivity
        long long int *connectivity,            // connectivity of hexahedral elements
        long long int *enclosing_elem_indices,  // element indices of the enclosing element [npoints, nNodes]
        double* nodes,                          // nodes with shape [npoints_mesh, 3]
        double* weights,                        // matrix [npoints, nNodes] containg interpolation weights
        double* points)                         // points that require interpolation
{
    long long int i, j, k, l, m, n, idx;
    double vtx[8][3];
    double pnt[3];
    double solution[3];
    double interpolator[8];
    long long int npoints_failed = 0;

    long long int elem_number;
    for (i=0; i<npoints; i = i + 1){
        for (m = 0; m < nDim; m = m + 1)
            pnt[m] = points[i * nDim + m];       // fill pnt

        for (j = 0; j < nelem_to_search; j = j + 1){
            //Get element number from list of nearest elements
            elem_number = nearest_element_indices[i * nelem_to_search + j];


            for (k = 0; k < nNodes; k = k + 1){
                idx = connectivity[elem_number * nNodes + k];       // get node number for each node
                for (l = 0; l < nDim; l = l + 1)
                    vtx[k][l] = nodes[idx * nDim + l];              // fill vtx with element nodes
            }

            //check if inside element, if trye get interpolation weights and node indices
            if (checkHull(pnt, vtx, solution)){
                interpolateAtPoint(solution, interpolator);         // fill interpolator with weights at ref. coords
                for (n = 0; n < nNodes; n = n + 1){
                    weights[i * nNodes + n] = interpolator[n];
                    enclosing_elem_indices[i * nNodes + n] = connectivity[elem_number * nNodes + n];
                }
                break; //interpolation weights found, go to next point
            }
            else if  (j == nelem_to_search - 1)
                npoints_failed = npoints_failed + 1; // count number of points that failed
        }
    }
    return npoints_failed;
}

void coordinateTransform(double pnt[], double vtx[8][3], double solution[]){
    int i;
    for (i = 0; i < nDim; i = i + 1)
        solution[i] = referenceToElementMapping(vtx[0][i], vtx[1][i], vtx[2][i], vtx[3][i], vtx[4][i],
                                                vtx[5][i], vtx[6][i], vtx[7][i], pnt[0], pnt[1], pnt[2]);
}

int checkHull(double pnt[3], double vtx[8][3], double solution[3])
{
    int i;
    if (inverseCoordinateTransform(pnt, vtx, solution))
    {
        // if converged, check if inside element
        for (i = 0; i < 3; i = i + 1)
        {
            if (fabs(solution[i]) > (1 + 0.025))
                return 0; // if not in element return False
        }
        return 1; // if converged and in element return true
    }
    return 0;  //if not converged return false
}

void interpolateAtPoint(double pnt[], double *interpolator)
    {
        double r, s, t;
        r = pnt[0];
        s = pnt[1];
        t = pnt[2];

        interpolator[0] = -0.125 * r * s * t + 0.125 * r * s + 0.125 * r * t - 0.125 * r + \
                          0.125 * s * t - 0.125 * s - 0.125 * t + 0.125;
        interpolator[1] = +0.125 * r * s * t - 0.125 * r * s + 0.125 * r * t - 0.125 * r - \
                          0.125 * s * t + 0.125 * s - 0.125 * t + 0.125;
        interpolator[2] = -0.125 * r * s * t + 0.125 * r * s - 0.125 * r * t + 0.125 * r - \
                          0.125 * s * t + 0.125 * s - 0.125 * t + 0.125;
        interpolator[3] = +0.125 * r * s * t - 0.125 * r * s - 0.125 * r * t + 0.125 * r + \
                          0.125 * s * t - 0.125 * s - 0.125 * t + 0.125;
        interpolator[4] = +0.125 * r * s * t + 0.125 * r * s - 0.125 * r * t - 0.125 * r - \
                          0.125 * s * t - 0.125 * s + 0.125 * t + 0.125;
        interpolator[5] = -0.125 * r * s * t - 0.125 * r * s + 0.125 * r * t + 0.125 * r - \
                          0.125 * s * t - 0.125 * s + 0.125 * t + 0.125;
        interpolator[6] = +0.125 * r * s * t + 0.125 * r * s + 0.125 * r * t + 0.125 * r + \
                          0.125 * s * t + 0.125 * s + 0.125 * t + 0.125;
        interpolator[7] = -0.125 * r * s * t - 0.125 * r * s - 0.125 * r * t - 0.125 * r + \
                          0.125 * s * t + 0.125 * s + 0.125 * t + 0.125;
    }

double referenceToElementMapping(double v0, double v1, double v2, double v3,
                                 double v4, double v5, double v6, double v7,
                                 double r, double s, double t)
{
    return v0 + 0.5 * (r + 1.0) * (-v0 + v3) +
                0.5 * (s + 1.0) * (-v0 + v1 - 0.5 * (r + 1.0) * (-v0 + v3) +
                0.5 * (r + 1.0) * (-v1 + v2)) +
                0.5 * (t + 1.0) * (-v0 + v4 - 0.5 * (r + 1.0) * (-v0 + v3) +
                0.5 * (r + 1.0) * (-v4 + v5) -
                0.5 * (s + 1.0) * (-v0 + v1 - 0.5 * (r + 1.0) * (-v0 + v3) +
                0.5 * (r + 1.0) * (-v1 + v2)) +
                0.5 * (s + 1.0) * (-v4 + v7 - 0.5 * (r + 1.0) * (-v4 + v5) +
                0.5 * (r + 1.0) * (v6 - v7)));
}

double dNdR(int N, double S, double T)
{
    return 0.125 * mNodesR[N] * (S * mNodesS[N] + 1) * (T * mNodesT[N] + 1);
}

double dNdS(int N, double R, double T)
{
    return 0.125 * mNodesS[N] * (R * mNodesR[N] + 1) * (T * mNodesT[N] + 1);
}

double dNdT(int N, double R, double S)
{
    return 0.125 * mNodesT[N] * (R * mNodesR[N] + 1) * (S * mNodesS[N] + 1);
}

// Computes inverse jacobian
void inverseJacobianAtPoint(double pnt[3], double vtx[8][3], double invJac[3][3]){

    // Initializing variables
    double R, S, T;
    R = pnt[0];
    S = pnt[1];
    T = pnt[2];
    double det = 0;
    double Dn[nDim][nNodes];
    double jac[3][3];

    int J, I;
    for (J = 0; J < nNodes; J = J + 1)
    {
        for (I = 0; I < nDim; I = I + 1)
        {
            if (I == 0)
                Dn[I][J] = dNdR(J, S, T);
            else if (I == 1)
                Dn[I][J] = dNdS(J, R, T);
            else if (I == 2)
                Dn[I][J] = dNdT(J, R, S);
        }
    }
    dot_product_matrix_matrix(Dn, vtx, jac); //places product into jac
    det = determinant(jac);                  //computes determinant of jac
    mInverse(jac, invJac, det);              //computes inverse of jac and places into invJac
}

// Gets reference coordinates for pnt in vtx and stores them in solution
int inverseCoordinateTransform(double pnt[3], double vtx[8][3], double solution[3])
{
    double scalexy;
    double scale;
    int max_iter = 15;
    double tol;
    int num_iter = 0;
    double update[3];
    double jacobian_inverse[3][3];
    double jacobian_inverse_t[3][3];
    double T[3];
    double objective_function[3];
    int i;

    // Initialize solution with zeros
    for (i = 0; i < nDim; i = i + 1)
        solution[i] = 0;

    scalexy = fabs((vtx[1][0] - vtx[0][0])) > fabs((vtx[1][1] - vtx[0][1])) ?
              fabs(vtx[1][0] - vtx[0][0]) : fabs(vtx[1][1] - vtx[0][1]);
    scale = fabs((vtx[1][2] - vtx[0][2])) > scalexy ? fabs(vtx[1][2] - vtx[0][2]) : scalexy;

    tol = 1e-10 * scale;
    while (num_iter < max_iter)
    {
        coordinateTransform(solution, vtx, T);

        for (i = 0; i < nDim; i = i + 1)
            objective_function[i] = pnt[i] - T[i];

        if ((fabs(objective_function[0]) < tol) && (fabs(objective_function[1]) < tol)
                && (fabs(objective_function[0]) < tol))
            return 1;
        else
        {
            inverseJacobianAtPoint(solution, vtx, jacobian_inverse);  //compute inverse of jacobian
            transpose(jacobian_inverse, jacobian_inverse_t);
            dot_product_matrix_vector(jacobian_inverse_t, objective_function, update);

            for (i = 0; i < nDim; i = i + 1)
                solution[i] = solution[i] + update[i];
        }
        num_iter = num_iter + 1;
    }
    return 0;
}

////////////////////// Linear Algebra Helper Functions ///////////////////////////
// Transpose of 3 by 3 matrix
void transpose(double m[3][3], double mTranspose[3][3])
{
    int i, j;
    for(i = 0; i <= 2; i = i + 1)
    {
        for(j = 0; j <= 2; j = j + 1)
            mTranspose[j][i] = m[i][j];
    }
}

//determinant of a 3 by 3 matrix
double determinant(double m[3][3])
{
    double det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
                 m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                 m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    return det;
}

// Inverse of a 3 by 3 matrix
void mInverse(double m[3][3], double minv[3][3], double det)
{
    double invdet = 1 / det;
    minv[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * invdet;
    minv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invdet;
    minv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invdet;
    minv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invdet;
    minv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invdet;
    minv[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * invdet;
    minv[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * invdet;
    minv[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * invdet;
    minv[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * invdet;
}

// Dot product of Dn and vtx to compute jacobian
void dot_product_matrix_matrix(double Dn[3][8], double vtx[8][3], double jac[3][3])
{
    int q, i, j;
    double sum;
    for (q = 0; q < 3; q = q + 1)
    {
        for (j = 0; j < 3; j = j + 1)
        {
            sum = 0;
            for (i = 0; i < 8; i = i + 1)
                sum = sum + Dn[q][i] * vtx[i][j];

            jac[q][j] = sum;
        }
    }
}

// dot product of jac_t and objective function to get update to solution
void dot_product_matrix_vector(double Jac_transpose[3][3],
                               double objective_function[3], double solution[3])
{
    int i, j;
    double sum;
    for (i = 0; i <= 2; i = i + 1)
    {
        sum = 0;
        for (j = 0; j <= 2; j = j + 1)
            sum = sum + Jac_transpose[i][j] * objective_function[j];

        solution[i] = sum;
    }
}

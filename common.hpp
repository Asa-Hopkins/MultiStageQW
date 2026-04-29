#pragma once
#include <eigen3/Eigen/Core>
#include <math.h>

#define PI 3.1415926535897932384626

//Get least significant bit
unsigned int LSB(int n){
  return n & (-n);
}

//Convert index to grey code
unsigned int grey(unsigned int n){
  return n ^ (n >> 1);
}

//Get position of the only set bit
//Taken from http://graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn
unsigned int log2(unsigned int v){
  static const int MultiplyDeBruijnBitPosition2[32] = 
  {
    0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 
    31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
  };
  return MultiplyDeBruijnBitPosition2[(uint32_t)(v * 0x077CB531U) >> 27];
}

//Taken from https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
//Credit goes to the author, njuffa, for this and the next function.
float my_erfcinvf (float a)
{
    float r;

    if ((a >= 2.1875e-3f) && (a <= 1.998125f)) { // max. ulp err. = 2.77667
        float p, t;
        t = fmaf (-a, a, a + a);
        t = logf (t);
        p =              5.43877832e-9f;  //  0x1.75c000p-28 
        p = fmaf (p, t,  1.43286059e-7f); //  0x1.33b458p-23 
        p = fmaf (p, t,  1.22775396e-6f); //  0x1.49929cp-20 
        p = fmaf (p, t,  1.12962631e-7f); //  0x1.e52bbap-24 
        p = fmaf (p, t, -5.61531961e-5f); // -0x1.d70c12p-15 
        p = fmaf (p, t, -1.47697705e-4f); // -0x1.35be9ap-13 
        p = fmaf (p, t,  2.31468701e-3f); //  0x1.2f6402p-9 
        p = fmaf (p, t,  1.15392562e-2f); //  0x1.7a1e4cp-7 
        p = fmaf (p, t, -2.32015476e-1f); // -0x1.db2aeep-3 
        t = fmaf (p, t,  8.86226892e-1f); //  0x1.c5bf88p-1 
        r = fmaf (t, -a, t);
    } else {
        float p, q, s, t;
        t = (a >= 1.0f) ? (2.0f - a) : a;
        t = 0.0f - logf (t);

        s = sqrtf (1.0f / t);
        p =              2.23100796e+1f;  //  0x1.64f616p+4
        p = fmaf (p, s, -5.23008537e+1f); // -0x1.a26826p+5
        p = fmaf (p, s,  5.44409714e+1f); //  0x1.b3871cp+5
        p = fmaf (p, s, -3.35030403e+1f); // -0x1.0c063ap+5
        p = fmaf (p, s,  1.38580027e+1f); //  0x1.bb74c2p+3
        p = fmaf (p, s, -4.37277269e+0f); // -0x1.17db82p+2
        p = fmaf (p, s,  1.53075826e+0f); //  0x1.87dfc6p+0
        p = fmaf (p, s,  2.97993328e-2f); //  0x1.e83b76p-6
        p = fmaf (p, s, -3.71997419e-4f); // -0x1.86114cp-12
        p = fmaf (p, s, s);
        r = 1.0f / p;
        if (a >= 1.0f) r = 0.0f - r;
    }
    return -r;
}

/* Compute inverse of the CDF of the standard normal distribution.
   max ulp err = 4.08385
*/
float my_normcdfinvf (float a)
{
    return fmaf (-1.41421356f, my_erfcinvf (a + a), 0.0f);
}

//For comparison to Callison et al.
float heur[20] = {1.2082979794574937, 1.3131560483482256, 1.4100589067449547, 1.5006739957742985, 1.5861415036770354, 1.6672845086916428, 1.7447219000349252, 1.81893401086401, 1.8903031784517639, 1.9591400720339947, 2.025701482876633, 2.090202753583327, 2.1528267101788385, 2.2137302372981815, 2.2730492199794945, 2.330902325676296, 2.3873939451016604, 2.4426165114475493, 2.4966523525078355, 2.5495751865531977};

// Load H_P for a given problem from file, returning E_loc and E_abs
// J and h must be pre-allocated to (n,n) and (n,) respectively
void load_problem(std::ifstream& file, unsigned int n,
                  Eigen::ArrayXf& H_P, Eigen::ArrayXXf& J, Eigen::ArrayXf& h, unsigned int& E_loc, float& E_abs){
  using namespace Eigen;

  double temp[n*(n+1)/2];

  unsigned int N = 1 << n;
  ArrayXXf state(n,n);

  //Read next set of parameters
  file.read(reinterpret_cast<char*>(temp), 4*n*(n+1));

  state.setConstant(1);
  J.setConstant(0);
  h.setConstant(0);

  //Load J matrix
  int k = 0;
  for (int i = 1; i < n; i++){
    for (int j = 0; j < i; j++){
      J(i,j) = temp[k];
      k++;
    }
  }

  //Absorb 1/2 factor into J to follow paper
  J += J.transpose().eval();
  J /= 2;

  for (int i = 0; i < n; i++){
    h(i) = temp[n*(n-1)/2 + i];
  }

  //The way we calculate E is prone to error so use a double
  //Start with the energy of the all -1s state
  double E = J.sum() - h.sum();
  H_P[0] = E;
  
  double E_0 = E, E_max = E;
  E_loc = 0;

  //Use a grey code to efficiently evaluate all energies
  for (unsigned int i = 1; i < N; i++){
    unsigned int flip = log2(LSB(i));
    state.row(flip) *= -1;
    state.col(flip) *= -1;
    state(flip,flip) *= -1;
    E += 4*(J.row(flip)*state.row(flip)).sum() - 2*h(flip)*state(flip,flip);
    H_P[grey(i)] = E;

    //keep track of ground state
    if (E < E_0){
      E_0 = E;
      E_loc = grey(i);
    }

    //keep track of highest state too
    if (E > E_max){
      E_max = E;
    }
  }

  //We want to shift H_P to reduce the spectral radius
  //This doesn't change the result but shortens the calculation
  H_P -= (E_max + E_0)/2;
  E_abs = (E_max - E_0)/2;
  return;
}

// Compute per-problem gammas
Eigen::ArrayXf compute_gammas(unsigned int n, unsigned int m, float HP2, float& E_est){
  // This formula is from https://math.stackexchange.com/questions/89030/expectation-of-the-maximum-of-gaussian-random-variables/89147#89147
  //Calculates estimated maximum energy level using the known variance and assuming a normal distribution    
  unsigned int N = 1 << n;
  float b = my_normcdfinvf(1/(float)N);
  float e_m = 0.577215664901532860;
  float e = 2.718281828459045;
  E_est = (1 - e_m)*b + e_m*my_normcdfinvf(1/(e*N));
  E_est *= sqrt(HP2);
  Eigen::ArrayXf gammas(m);
  //old heuristic
  //gammas[i] = heur[n - 5]/tan(PI*(i+1)/(2*m + 2));
  for (int i = 0; i<m; i++){gammas[i] = E_est/tan(PI*(i+1)/(2*m + 2))/n;}

  return gammas;
}

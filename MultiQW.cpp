#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/FFT>
#include "vcl2/vectorclass.h"
#include <math.h>
#include <complex>
#include <iostream>
#include <random>
#include <fstream>
#include "ApproxTools/Chebyshev.hpp"

#define PI 3.1415926535897932384626

using namespace Eigen;

//Get least significant bit
unsigned int LSB(int n){
  return n & (-n);
}

//Convert value to grey code
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

void Clenshaw_step(float* b1, float* b2, float* hp, float* psi, const unsigned int n, float scale, float gamma, float coef){
  //This function is a bit of a mess, originally it calculated b2 += H_G @ b1 by essentially using a fast walsh-hadamard transform
  //But the first part of it is compute heavy enough that we can put bandwidth limited calculations next to it for free
  //So now it does a full step of the Clenshaw algorithm, setting b2 = coef*psi + 2*(H @ b1) - b2
  //Where H = (H_P - gamma*H_G) / scale

  //Multiplying H_G by something is slow here as it'd require a multiplication on every "b += a"
  //We rescale b2 and then scale the final result so that H_G has a coefficient of 1

  // We set new_scale = -2*gamma/scale, then the calculation is
  // b2 = new_scale * (H_G @ b1 - (H_P @ b1)/gamma - b2/new_scale)
  // In the final step we add psi*coeff as it saves a second pass over b2
  // At that point in the calculation, 

  const unsigned int N = (1 << n);

  Vec16f a;
  Vec16f b;
  Vec16f H;
  int h = 16;
  //(1 << max_cache) should line up with cache size in some sense
  //Needs to be tuned for each machine ideally

  constexpr int max_cache = 15;
  float new_scale = -2.0f*gamma/scale;
  float new_scale_inv = 1.0f/new_scale;

  //Use permutations for h<16 cases
  //We do enough computation here that we can load some out-of-cache data for free
  for (int i = 0; i<N; i+=16){
    a.load(b1+i);
    b.load(b2+i);

    b *= -new_scale_inv;
    H.load(hp+i);
    b -= a*H/gamma;

    b += permute16<1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14>(a);
    b += permute16<2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13>(a);
    b += permute16<4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11>(a);
    b += permute16<8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7>(a);

    //For n larger than this, we can't fit vectors in cache anymore so these are
    //guaranteed cache misses
    if (n > max_cache){
      for (h = 1 << max_cache; h < N; h*=2){
        int temp = h&i ? -h : h;
        a.load(b1 + i + temp);
        b += a;
      }
    }

    b.store(b2+i);
  }

  //The vector can be kept in cache for these sizes
  int end = N < (1 << max_cache) ? N : 1 << max_cache;
  //I want to keep this check outside the loop
  if (coef != 0){
    for (int i = 0; i<N; i+=16){
      b.load(b2+i);
      for (h = 16; h < end; h*=2){
        int temp = h&i ? -h : h;
        a.load(b1 + i + temp);
        b += a;
      }
      b *= new_scale;
      //add psi*coef
      H.load(psi+i);
      b += H*coef;
      b.store(b2+i);
    }
  } else {
    for (int i = 0; i<N; i+=16){
      b.load(b2+i);
      for (h = 16; h < end; h*=2){
        int temp = h&i ? -h : h;
        a.load(b1 + i + temp);
        b += a;
      }
      b *= new_scale;
      b.store(b2+i);
    }
  }
  return;
}

//Returns n'th order Chebyshev expansion of exp(i*x*scale) on [-1,1]

ArrayXf Clenshaw(Eigen::Ref<VectorXf> coeffs,
                  Eigen::Ref<ArrayXf> psi,
                  Eigen::Ref<ArrayXf> H_P,
                  float gamma,
                  float scale, bool psi_real = false) {
  // Calculate exp(-i*H*t)@psi using the Clenshaw algorithm
  // with polynomial coefficients stored in coeffs.

  int N = H_P.size();
  int n = log2(H_P.size());
  thread_local ArrayXf b1, b2;
  b2.setZero(2*N);
  b1.setZero(2*N);

  //Our arrays are N real values and then N imaginary values
  //Since H_P and coeffs are real valued, we can skip unnecessary calculations
  auto Re = [&](auto& x) -> decltype(auto) { return x.head(N); };
  auto Im = [&](auto& x) -> decltype(auto) { return x.tail(N); };

  bool first = true;

  //On second iteration, b1 == 0 so could optimise for that
  bool second = false;
  int im_coef = psi_real ? 0 : 1;
  
  for (int r = coeffs.size() - 1; r > 0; --r) {
    if (not first){
      // Apply H_G to (b1r,b1i) -> (b2r,b2i)
      //Odd terms are imaginary
      if (r&1){
        Clenshaw_step(b1.data(), b2.data(), H_P.data(), psi.data()+N, n, scale, gamma, -im_coef*coeffs[r]);
        Clenshaw_step(b1.data()+N, b2.data()+N, H_P.data(), psi.data(), n, scale, gamma, coeffs[r]);
      } else {
        Clenshaw_step(b1.data(), b2.data(), H_P.data(), psi.data(), n, scale, gamma, coeffs[r]);
        Clenshaw_step(b1.data()+N, b2.data()+N, H_P.data(), psi.data()+N, n, scale, gamma, im_coef*coeffs[r]);
      }

    } else {
      if (r&1){
        if (not psi_real){Re(b2) -= Im(psi)*coeffs[r];}
        Im(b2) += Re(psi)*coeffs[r];
      }
      else{
        if (not psi_real){Im(b2) += Im(psi)*coeffs[r];}
        Re(b2) += Re(psi)*coeffs[r];
      }
    }

    second = first;
    first = false;

    // Swap b1 and b2 without actually copying
    std::swap(b1, b2);
  }
  // Final iteration
  Clenshaw_step(b1.data(), b2.data(), H_P.data(), psi.data(), n, 2.0*scale, gamma, coeffs[0]);
  Clenshaw_step(b1.data()+N, b2.data()+N, H_P.data(), psi.data()+N, n, 2.0*scale, gamma, im_coef*coeffs[0]);

  return b2;
}


float heur[20] = {1.2082979794574937, 1.3131560483482256, 1.4100589067449547, 1.5006739957742985, 1.5861415036770354, 1.6672845086916428, 1.7447219000349252, 1.81893401086401, 1.8903031784517639, 1.9591400720339947, 2.025701482876633, 2.090202753583327, 2.1528267101788385, 2.2137302372981815, 2.2730492199794945, 2.330902325676296, 2.3873939451016604, 2.4426165114475493, 2.4966523525078355, 2.5495751865531977};

int main(int argc, char* argv[]){
  //Arguments are number of spins, number of walk stages, filename, start position and number of problems
  //Last two are to allow for easier multi-threading, just start the program multiple times with different starts
  if (argc < 4) return -1;
  unsigned int n = atoi(argv[1]);
  unsigned int N = 1 << n;
  unsigned int m = atoi(argv[2]);
  
  char* filename = argv[3];

  unsigned int start = 0;
  unsigned int problems = 2000;

  float* results = (float*)malloc(problems*4);

  //Number of samples for Monte-Carlo integral
  int samples = 100;

  if (argc == 6){
    start = atoi(argv[4]);
    problems = atoi(argv[5]);
  }

  char output[20];
  std::sprintf(output, "output_%d_%d", n, m);
  std::ofstream outFile(output, std::ios::binary | std::ios::app);

  std::ifstream file(filename, std::ios::binary);

  //Seek to beginning, each problem has (n-1)*n parameters, and a float has 4 bytes.
  file.seekg(4*(n+1)*n*start, file.beg);

  Eigen::ArrayXXf times(m,samples);
  Eigen::ArrayXf success_probabilities(samples);

  ArrayXf gammas(m);
  ArrayXf onenorms(m);

  //Problem energy levels
  ArrayXf H_P(N);
  //This is used for calculating H_P, state(i,j) = sigma_z_i * sigma_z_j
  ArrayXXf state(n,n);

  //Our quantum register, with real and imaginary parts stored contiguously
  ArrayXf psi(2*N);

  //Ising problem parameters
  ArrayXXf J(n,n);
  ArrayXf h(n);

  
  //Used for reading in parameters from files
  char buffer[4*n*(n+1)];
  double temp[n*(n+1)/2];

  for (int problem = 0; problem < problems; problem++){

    double E_0 = 0;
    double E_max = 0;
    unsigned int E_loc = 0;

    //Read next set of parameters
    file.read(buffer, 4*n*(n+1));
    std::memcpy(temp, buffer, 4*n*(n+1));

    J.setConstant(0);

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

    state.setConstant(1);
    psi.head(N).setConstant(1/sqrt(N));
    psi.tail(N).setZero();
    bool psi_real = true;

    //The way we calculate E is prone to error so use a double
    //Start with the energy of the all -1s state
    double E = J.sum() - h.sum();
    H_P[0] = E;

    //In case first state is ground
    E_0 = E;
    E_max = E;
  
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
    float E_abs = (E_max - E_0)/2;
    //Now H_P has been calculated

    // This formula is from https://math.stackexchange.com/questions/89030/expectation-of-the-maximum-of-gaussian-random-variables/89147#89147
    //Calculates estimated maximum energy level using the known variance and assuming a normal distribution    
    float b = my_normcdfinvf(1/(float)N);
    float e_m = 0.577215664901532860;
    float e = 2.718281828459045;
    float a = (1 - e_m)*b + e_m*my_normcdfinvf(1/(e*N));

    //If the h was 0, we'd just want HP2 = 2*(J*J).sum()
    //If we were to map to an n+1 qubit problem we'd get h/2 in both a row and a column
    //so we need to add 2*2*(h/2).dot(h/2) = h.dot(h) 
    float HP2 = 2*(J*J).sum() + (h*h).sum();

    //std::cout << HP2 << " " << E_0 << "\n";

    //Can in theory use higher moments for a better approximation, but this is difficult in practice
    //float kurt = (H_P*H_P*H_P*H_P).mean() / pow((H_P*H_P).mean(),2);
    //float skew = (H_P*H_P*H_P).mean() / pow((H_P*H_P).mean(),1.5);
    //std::cout << (E_abs - a*sqrt(HP2))/E_abs << " " << kurt << " " << sqrt(HP2)*(my_normcdfinvf(1/(e*N)) - b)*sqrt(PI*PI/6)/E_abs << "\n";
    //std::cout << (H_P * H_P).sum()/N << " " << HP2 << " " << heur[n - 5]*n << " " << E_0 << "\n\n";
  

    //Should really use higher quality RNG
    std::mt19937 gen(29552825458725);

    //Calculate gammas, upper bounds on spectral radius and generate evolution times.
    //Old heuristic
    //gammas[i] = heur[n - 5]/tan(PI*(i+1)/(2*m + 2));
    for (int i = 0; i<m; i++){gammas[i] = a*sqrt(HP2)/tan(PI*(i+1)/(2*m + 2))/n;}
    float total_t = 0;
    //Calculate first and last stage times
    float delta2 = 16*(J*J).sum() + 4*(h*h).sum();

    for (int i = 0; i<m; i++){

      //The denominator for the last stage should be gamma*(a - n*E_0) where a is the sum of energy levels one bit flip from the ground state
      //I estimate "a" as n times the mean of a half-normal distribution with variance <(delta_ij)^2 / n>
      //This seems to consistently under-estimate by a factor of 1.5 - 2, which is fine
      float last_denom = sqrt(2/PI) * gammas[i] * sqrt(n * delta2);

      //estimate expected change in <H_G> for this stage
      float gamma_last = (i == 0) ? 1 : gammas[i - 1] / sqrt(1 + gammas[i - 1]*gammas[i - 1]);
      float gamma_next = (i+1 == m) ? 0 : gammas[i + 1] / sqrt(1 + gammas[i + 1]*gammas[i + 1]);
      float dH = gamma_last - gamma_next;

      //It's possible to write the heuristics in terms of either <H_P> or <H_G>
      //The <H_P> one has an extra approximation though so I use <H_G> for now 
      float first_t = sqrt(4*n*dH/delta2);
      float last_t = sqrt(2*n*dH/last_denom);
      float short_t;
      //No harm in evolving too long so we pick the longest, except the first stage which we know exactly
      if (i == 0){short_t = first_t;}
      else{short_t = std::max(first_t, last_t);}
      std::uniform_real_distribution<float> rand_t(short_t, 2*short_t);
      total_t += short_t;
      for(int j = 0; j < samples; j++) {
        times(i,j) = rand_t(gen);
      }
    }
    std::cout << total_t << " ";

    //Optimisation for single-stage
    if (m == 1){
      std::sort(times.row(0).begin(), times.row(0).end());
      //Have to calculate backwards to avoid modifying data that's still needed
      times(0,seq(placeholders::last,1,-1)) -= times(0,seq(placeholders::last-1,0,-1));
    }
    
    for (int j = 0; j < samples; j++){
      //Loop through all the times and calculate the success probability
      for (int i = 0; i < m; i++){
        float gamma = gammas[i];
        float onenorm = (E_abs + gamma*n);

        double scale = onenorm * times(i,j);
        std::function<double(double)> f = [scale](double x) {return sin(scale*x) + cos(scale*x);};
        ArrayXf coeffs = Chebyshev<double>::RCF_odd_even(f, 1e-6).coeffs.cast<float>().array();
        psi = Clenshaw(coeffs, psi, H_P, gamma, onenorm, psi_real);
        //Approximation errors make this method non-unitary so we renormalise
        psi /= psi.matrix().norm();
        psi_real = false;
      }
      success_probabilities(j) = psi[E_loc]*psi[E_loc] + psi[E_loc+N]*psi[E_loc+N];
      if (m != 1){
        psi.head(N).setConstant(1/sqrt(N));
        psi.tail(N).setZero();
        psi_real = true;
      }

    }
  results[problem] = success_probabilities.sum()/samples;
  std::cout << results[problem] << "\n";
  }
  outFile.write(reinterpret_cast<const char*>(results), problems * sizeof(float));
}

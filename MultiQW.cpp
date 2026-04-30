#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/FFT>
#include "vcl2/vectorclass.h"
#include <math.h>
#include <complex>
#include <iostream>
#include <random>
#include <fstream>
#include "ApproxTools/Chebyshev.hpp"
#include "common.hpp"

#define PI 3.1415926535897932384626

using namespace Eigen;

//This is the SplitMix64 PRNG, which I use to generate seeds for C++'s Mersene Twister
uint64_t seed_hash(uint64_t x) {
    x += 0x9e3779b97f4a7c15;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
    return x ^ (x >> 31);
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

void Clenshaw(Eigen::Ref<VectorXf> coeffs,
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

  psi = b2;
}

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

  //Number of samples for Monte-Carlo integral
  int samples = 100;

  if (argc >= 6){
    start = atoi(argv[4]);
    problems = atoi(argv[5]);
  }

  std::string output_dir = (argc >= 7) ? argv[6] : "./results";
  std::string output = output_dir + "/output_" + std::to_string(n) + "_" + std::to_string(m);

  #ifdef VERBOSE
    std::cout << "Running program with n=" << n << ", m=" << m
              << ", filename=" << filename
              << ", start_point=" << start
              << ", problems=" << problems << "\n";
  #endif

  std::ofstream outFile(output, std::ios::binary | std::ios::in | std::ios::out);
  std::ifstream file(filename, std::ios::binary);

  //Seek to beginning, each problem has (n+1)*n/2 parameters, and a double has 8 bytes.
  file.seekg(4*(n+1)*n*start, file.beg);

  Eigen::ArrayXXf times(m,samples);
  Eigen::ArrayXf success_probabilities(samples);

  ArrayXf gammas(m);
  ArrayXf onenorms(m);

  //Problem energy levels
  ArrayXf H_P(N);

  //Our quantum register, with real and imaginary parts stored contiguously
  ArrayXf psi(2*N);

  //Ising problem parameters
  ArrayXXf J(n,n);
  ArrayXf h(n);

  long int base_seed = 29552825458725;

  for (int problem = 0; problem < problems; problem++){
    float E_abs = 0;
    unsigned int E_loc = 0;

    //Initialise PRNG in a reproducible way that avoids correlations.
    std::mt19937 gen(seed_hash(base_seed + start + problem));

    load_problem(file, n, H_P, J, h, E_loc, E_abs);

    psi.head(N).setConstant(1/sqrt(N));
    psi.tail(N).setZero();
    bool psi_real = true;

    //If h was 0, we'd just want HP2 = 2*(J*J).sum()
    //If we were to map to an n+1 qubit problem we'd get h/2 in both a row and a column
    //so we need to add 2*2*(h/2).dot(h/2) = h.dot(h) 
    float HP2 = 2*(J*J).sum() + (h*h).sum();
    float E_est = 0;
    //Can in theory use higher moments for a better approximation, but this is difficult in practice

    //Calculate gammas, estimate of energy spread and generate evolution times.
    gammas = compute_gammas(n,m,HP2,E_est);

    float total_t = 0;
    //Calculate first and last stage times
    //For the average squared energy difference, we use the J' formula from Appendix B
    float delta2 = 16*(J*J).sum() + 4*(h*h).sum();

    float short_t;
    for (int i = 0; i<m; i++){

      //The denominator for the last stage should be gamma*4*|E_0|, so we use E_est from our earlier heuristic
      //I use only 2*E_est as it's important to overestimate evolution time rather than underestimate. 
      float last_denom = gammas[i] * 2 * E_est;

      //estimate expected change in <H_G> for this stage
      float gamma_last = (i == 0) ? 1 : gammas[i - 1] / sqrt(1 + gammas[i - 1]*gammas[i - 1]);
      float gamma_next = (i+1 == m) ? 0 : gammas[i + 1] / sqrt(1 + gammas[i + 1]*gammas[i + 1]);
      float dE = 2*n*(gamma_last - gamma_next);

      //It's possible to write the heuristics in terms of either <H_P> or <H_G>
      //The <H_P> one has an extra approximation though so I use <H_G> for now 
      float first_t = sqrt(2*dE/delta2);
      float last_t = sqrt(dE/last_denom);

      //No harm in evolving too long so we pick the longest, except the first stage which we know exactly
      if (i == 0){short_t = first_t;}
      else{short_t = std::max(first_t, last_t);}

      #ifdef VERBOSE
        std::cout << short_t << " " << gammas[i] << "\n";
      #endif

      std::uniform_real_distribution<float> rand_t(short_t, 2*short_t);
      total_t += short_t;
      for(int j = 0; j < samples; j++) {
        times(i,j) = rand_t(gen);
      }
    }

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

        auto f = [scale](double x) { return sin(scale * x) + cos(scale * x); };
        
        ArrayXf coeffs = Chebyshev<double>::RCF_odd_even(f, 1e-6).coeffs.cast<float>().array();

        Clenshaw(coeffs, psi, H_P, gamma, onenorm, psi_real);
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

  //Write results one at a time so less data is lost in a crash
  float result = success_probabilities.sum() / samples;
  outFile.seekp((start + problem) * sizeof(float));
  outFile.write(reinterpret_cast<char*>(&result), sizeof(float));
  std::cout << result << "\n\n";
  }
}

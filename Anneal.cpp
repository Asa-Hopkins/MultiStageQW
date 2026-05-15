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

#include <functional>
#include <unordered_map>

#define PI 3.1415926535897932384626

//Positions and weights for 16-point Gauss-Legendre quadrature on [0,1]
static float gauss_x[16] = {0.00529953,0.02771249,0.0671844,0.1222978,0.19106188,0.27099161
  ,0.35919822,0.45249375,0.54750625,0.64080178,0.72900839,0.80893812
  ,0.8777022,0.9328156,0.97228751,0.99470047};
  
static float gauss_w[16] = {0.01357623,0.03112676,0.04757926,0.06231449,0.07479799,0.08457826
  ,0.09130171,0.09472531,0.09472531,0.09130171,0.08457826,0.07479799
  ,0.06231449,0.04757926,0.03112676,0.01357623};

//Degree 50 polynomials representing D-Wave's anneal schedules
//I accidentally swapped around A and B compared to D-Wave's documentation

/*
static Chebyshev A = Chebyshev(Eigen::VectorXd{{20.070031952707506, 24.220359108235975, 4.367808269765021, -2.589786403589441, -0.4192307104919708, 0.8965653959283482, -0.055968320341575506, -0.37156774031474094, 0.11099564962026509, 0.1580898581720695, -0.09049871068922966, -0.06460104061598992, 0.0616719632259658, 0.023427999640300543, -0.03870108801841404, -0.0060040053344346915, 0.023013003017160333, -0.0006619057868628604, -0.013094187680436409, 0.002637675182436254, 0.007137050966597919, -0.0027300569437756126, -0.003709270416316135, 0.0022048593075167312, 0.0018126038127567606, -0.0015839261420041455, -0.000809499568170436, 0.0010613670576735221, 0.00030469988658528616, -0.0006739787424978146, -7.209931205322088e-05, 0.0004125959252769214, -2.3115201185225365e-05, -0.00024136337167866416, 4.957313126325441e-05, 0.00013698322971268532, -5.00823595305927e-05, -7.13375893490882e-05, 4.000439320633875e-05, 3.5747386695064656e-05, -3.066608282570327e-05, -1.4632378584101503e-05, 2.139855368686605e-05, 7.038674244610662e-06, -1.4898321208933751e-05, -2.2113497534980074e-06, 8.730903999068507e-06, 1.9151781356852093e-06, -4.514549804948698e-06, 7.854240985381376e-07, 1.9127931090434647e-06}});

static Chebyshev B = Chebyshev(Eigen::VectorXd{{11.563630397586653, -18.109532145909338, 7.575505974573865, 0.18580757699361916, -1.8146856487639695, 0.2828941839379147, 0.6135973229979538, -0.21493295427895165, -0.22690717199241517, 0.12968284511598235, 0.08475938335123667, -0.07301402362500076, -0.03015006044090704, 0.03972528822384539, 0.00927928971013331, -0.02111590463950705, -0.0017023090193374751, 0.010983468455186487, -0.0006814583231087717, -0.005591212058122939, 0.0011610180440455795, 0.0027651141788543753, -0.0010109960808147023, -0.001326916675406786, 0.0007386139138751968, 0.0006012258118572893, -0.00048367393192482074, -0.0002592454145595954, 0.0003064287448849661, 9.211622931941838e-05, -0.00017783960001830756, -2.8434914735138146e-05, 0.00010701060490334036, -4.8239776349260855e-06, -5.428458590588797e-05, 6.212484369731428e-06, 3.411391810129519e-05, -1.2917316272301924e-05, -1.315737152016161e-05, 5.200160201671596e-06, 1.0618557729949463e-05, -8.439200976939821e-06, -8.167441933326636e-07, 1.3600016303918068e-06, 3.495601312561436e-06, -4.311237527846263e-06, 1.4943750015483648e-06, -3.1832214487412466e-07, 1.7149736653426784e-06, -2.108466375010687e-06, 1.0543699185891486e-06}});
*/

//Infinite-stage schedule

static Chebyshev A = Chebyshev(Eigen::VectorXd{{10.06631359845914, 9.572528519345955, 2.344490791938677, 1.9317599108212864, 0.9352536112872702, 0.4923742163619589, 0.4802386481736466, 0.15670178289541264, 0.2520388615953169, 0.06166229028478701, 0.1283690802396797, 0.0336564810515741, 0.05376197483397199, 0.030483281872071888, 0.013109601563453588, 0.024061407445070293, 0.00321210927016443, 0.008506410135642477, 0.008706341224489331, -0.0063872111681329, 0.015290687679588542, -0.015283804604360088, 0.01832950902352909, -0.017778053364876006, 0.016207129517264066, -0.013881100230743182, 0.009455051214543701, -0.0067971464396239835, 0.0025049590179587952, -0.0015030427539693858, -0.0011916303300252076}});

static Chebyshev B = Chebyshev(Eigen::VectorXd{{10.422009750980182, -14.294698016745322, 6.295492891881257, -3.7785095411623875, 1.7024145296474624, -0.24603787435574676, -0.544653301711775, 1.0038017035015165, -1.131491475120528, 1.083582625447141, -0.9092995620680127, 0.6746102605792779, -0.42975757785917423, 0.21011235955407326, -0.049149444121195235, -0.044299019273784934, 0.0680202602249867, -0.03590685831874209, -0.03237193482921633, 0.12033294201115276, -0.20378338419502026, 0.27797547035329845, -0.3205627765306063, 0.3422252472843311, -0.32319926447961506, 0.2906886858770172, -0.22622564095503872, 0.1700349368984366, -0.09837377757576148, 0.05570380289190164, -0.006108186357898872}});

//Look up what anneal fraction s gives a given gamma
//I return values between -1 and 1 for consistency 
float gamma_lookup(float gamma){
  //g(s) is monotonically decreasing so I just use binary search
  //I could likely replace this with a Chebyshev polynomial too
  float bracket[2] = {-1,1};
  while (bracket[1] - bracket[0] > 1e-5){
    float midpoint = (bracket[1] + bracket[0])/2.0;
    if (gamma > B(midpoint)/A(midpoint)){
      bracket[1] = midpoint;
    } else {
      bracket[0] = midpoint;
    }
  }
  return (bracket[1] + bracket[0])/2.0;
}

//I scale A and B later so keep a copy
static Chebyshev A_orig = A;
static Chebyshev B_orig = B;

using namespace Eigen;

//We calculate this once at the start and have it available everywhere
static Eigen::ArrayXf raw_norms;

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

//Evaulated the n'th shifted Legendre polynomial at x
//Domain has been shifted to [0,1]
float L(int n, float x){
  x = 2*x - 1;
  if (n == 0) return 1.0;
  if (n == 1) return x;

  //Uses Bonnet's recursion formula, but unrolled as a loop
  float p_prev_2 = 1.0; // Represents P_{i-2}(x)
  float p_prev_1 = x;    // Represents P_{i-1}(x)
  float p_current = 0;

  for (int i = 2; i <= n; ++i) {
    // Bonnet’s recurrence: P_i = ((2i-1)*x*P_{i-1} - (i-1)*P_{i-2}) / i
    p_current = ((2.0 * i - 1.0) * x * p_prev_1 - (i - 1.0) * p_prev_2) / (float)i;
        
    p_prev_2 = p_prev_1;
    p_prev_1 = p_current;
  }

  return p_current;
}

// hall_basis returns a 7-element array of approximate operator 1-norms (max over ground/excited state):
// It uses polynomial time by only storing the vector indices that are actually used

using State    = std::unordered_map<int, double>;
using Operator = std::function<State(const State&)>;

static State scale(const State& s, double c){
  State out;
  out.reserve(s.size());
  for (auto& [i, a] : s)
    out[i] = c * a;
  return out;
}

static State sum_states(const State& a, const State& b){
  State out = a;
  for (auto& [i, amp] : b)
    out[i] += amp;
  for (auto it = out.begin(); it != out.end(); )
    it = (it->second == 0.0) ? out.erase(it) : std::next(it);
  return out;
}

static double norm1(const State& s){
  double n = 0.0;
  for (auto& [i, a] : s)
    n += std::abs(a);
  return n;
}

static Operator make_com(Operator f, Operator g){
  return [f, g](const State& v) -> State {
    return sum_states(f(g(v)), scale(g(f(v)), -1.0));
  };
}

static double eval_norm(const Operator& op, const State& psi_ground, const State& psi_excited){
  return std::max(norm1(op(psi_ground)), norm1(op(psi_excited)));
}

Eigen::ArrayXf hall_basis(const Eigen::ArrayXf& hp_diag){
  //Upper bounds the norm of a set of commutators with an approximate 1-norm
  //order of output array is as follows:
  //  [HP, HG]
  //  [HP, [HP, HG]]
  //  [HG, [HP, HG]]
  //  [HP, [HP, [HP, HG]]]
  //  [HG, [HP, [HP, HG]]]
  //  [HG, [HG, [HP, HG]]]
  //  [HP, [HP, [HP, [HP, [HP, HG]]]]]
  const int n = log2(hp_diag.size());

  Eigen::Index ground_idx, excited_idx;
  hp_diag.minCoeff(&ground_idx);
  hp_diag.maxCoeff(&excited_idx);

  Operator hp = [&hp_diag](const State& s) -> State {
    State out;
    out.reserve(s.size());
    for (auto& [i, amp] : s){
      out[i] = amp * hp_diag[i];
    }
    return out;
  };

  Operator hg = [n](const State& s) -> State {
    State out;
    for (auto& [i, amp] : s){
      for (int k = 0; k < n; ++k){
        out[i ^ (1 << k)] += amp;
      }
    }
    return out;
  };

  Operator C1    = make_com(hp, hg);
  Operator DP    = make_com(hp, C1);
  Operator DG    = make_com(hg, C1);
  Operator HP_DP = make_com(hp, DP);
  Operator HG_DP = make_com(hg, DP);
  Operator HG_DG = make_com(hg, DG);

  //This can be used as a rough estimate of the largest 8th order term but it is a massive over-estimate
  //I don't use it currently as a result
  //Operator five  = make_com(hp, make_com(hp, make_com(hp, make_com(hp, C1))));

  const State psi_ground  = {{ground_idx,  1.0}};
  const State psi_excited = {{excited_idx, 1.0}};
  
  Eigen::ArrayXf norms(6);

  norms << eval_norm(C1,    psi_ground, psi_excited),
           eval_norm(DP,    psi_ground, psi_excited),
           eval_norm(DG,    psi_ground, psi_excited),
           eval_norm(HP_DP, psi_ground, psi_excited),
           eval_norm(HG_DP, psi_ground, psi_excited),
           eval_norm(HG_DG, psi_ground, psi_excited);

  return norms;
}

Eigen::ArrayXf commutator_bounds(const Eigen::ArrayXf& raw_bounds, const Eigen::ArrayXf& AB){
  float a1 = AB[0], a2 = AB[1], a3 = AB[2], b1 = AB[3], b2 = AB[4], b3 = AB[5];

  float a12 = abs(a1*b2 - b1*a2);
  float a13 = abs(a1*b3 - b1*a3);
  float a23 = abs(a2*b3 - b2*a3);

  Eigen::ArrayXf commutators(4);
  commutators << a23 * raw_bounds[0],
              a13 * (abs(a1)*raw_bounds[1] + abs(b1)*raw_bounds[2]),
              a12 * (abs(a2)*raw_bounds[1] + abs(b2)*raw_bounds[2]),
              a12 * (a1*a1 * raw_bounds[3] + abs(2*a1*b1) * raw_bounds[4] + b1*b1 * raw_bounds[5]);

  return commutators;
}

float error_f(const Eigen::ArrayXf& E, Eigen::ArrayXXf f){
  float error1 = (f(2,3)/6.0/(1 + f(2,1)) + 1/30.0)*E[0];
  float error2 =-((1.0 + f(2,1))*f(2,3)/24.0 + 1/60.0)*E[1];
  float error3 = (1/60.0 - (1 + 2*f(2,1))/54.0/(1 + f(2,1))/(1 + f(2,1)))*E[2];
  float error4 = (1/1440.0 - f(2,1)*f(2,1) / 288.0)*E[3];

  return abs(error1) + abs(error2) + abs(error3) + abs(error4);
}

//Calculates all integration constants from f_{2,1} and f_{2,3}
//I use a slightly larger matrix than needed to match the paper's 1-based indexing
Eigen::ArrayXXf calc_f(float f21, float f23) {
    Eigen::ArrayXXf f = Eigen::ArrayXXf::Zero(3, 4);

    f(2, 1) = f21;
    f(2, 3) = f23;
    
    f(1, 1) = (1.0 - f(2, 1)) / 2.0;
    
    f(1, 2) = 1.0 / 3.0 / (1.0 + f(2, 1));
    f(1, 3) = -f(2, 3) / 2.0;

    return f;
}

Eigen::ArrayXXf opt_f(const Eigen::ArrayXf& E) {
    //We can optimise f to minimise error
    //Skip for now
    Eigen::ArrayXXf f = calc_f(9/20.0,-7/25.0);

    return f;
}

Eigen::ArrayXf legendre_coeffs(float t, float dt) {
  //First three indices are A_n, last three are B_n
  Eigen::ArrayXf AB = Eigen::ArrayXf::Zero(6);

  //Use gauss-legendre quadrature to find terms via orthogonality integral
  for (int i = 0; i<16; i++){
    float shifted_t = t + dt * gauss_x[i];
    //Shift input to [-1,1]
    shifted_t = 2*shifted_t - 1;
    float A_t = A(shifted_t);
    float B_t = B(shifted_t);
    for (int j = 0; j < 3; j++){
      float temp = gauss_w[i] * (2*j + 1) * dt * L(j, gauss_x[i]);
      AB[j] += A_t * temp;
      AB[j + 3] += B_t * temp;
    }
  }

  return AB;
}

float error(float t, float dt){
  //Calculate first three terms of Legendre expansion of A and B
  Eigen::ArrayXf AB = legendre_coeffs(t, dt);

  Eigen::ArrayXf E = commutator_bounds(raw_norms, AB);
  return error_f(E,opt_f(E));

}

float find_dt(float t, float target_err, float initial_dt){
  //Use fixed-point iteration to find root
  float dt = initial_dt;
  for (int i = 0; i < 100; i++){
    //Typically converges very fast, 2-3 iterations
    float err = error(t, dt);
    if (abs(err - target_err)/target_err < 1e-6) break;
    dt *= pow((target_err / err),0.2);
  }
  return dt;
}

Eigen::ArrayXf greedy_breakpoints(float err_tol){
  std::vector<float> breakpoints;

  float current_t = 0.0;
  float dt = 0.01;
  float remaining = 1.0;
  while (remaining > 1e-10){
    remaining = 1.0 - current_t;
    if (error(current_t, remaining) <= err_tol) break;
    
    dt = find_dt(current_t, err_tol, dt);
    current_t += dt;
    breakpoints.push_back(dt);
  }
  breakpoints.push_back(remaining);
  return Eigen::Map<Eigen::ArrayXf>(breakpoints.data(), breakpoints.size());
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
  int samples = 1;

  if (argc >= 6){
    start = atoi(argv[4]);
    problems = atoi(argv[5]);
  }

  std::string output_dir = (argc >= 7) ? argv[6] : "./results";
  std::string output = output_dir + "/output_" + std::to_string(n) + "_" + std::to_string(m);

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

  for (int problem = 0; problem < problems; problem++){
    float E_abs = 0;
    unsigned int E_loc = 0;

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
      //The <H_P> one has an extra approximation though so I use <H_G>
      float first_t = sqrt(2*dE/delta2);
      float last_t = sqrt(dE/last_denom);
      //No harm in evolving too long so we pick the longest, except the first stage which we know exactly
      if (i == 0){short_t = first_t;}
      else{short_t = std::max(first_t, last_t);}
      //To convert walk time to anneal time, we need to see what A and B fields give our gamma value
      //The walk assumes we can apply H_P + gamma H_G, but we really apply A*H_P + B*H_G
      //So we write as A*(H_P + B/A * H_G) and reduce evolution time by a factor A to match.
      total_t += short_t / A(gamma_lookup(gammas[i]));
    }
    //For QW we take 100 samples on [t,2*t]
    //That's too expensive here, but for fair comparison multiply by 1.5
    total_t *= 1.5;
    //Rescale our fields to change evolution time.
    //This means our anneal time is always between 0 and 1
    A.coeffs *= total_t;
    B.coeffs *= total_t;

    raw_norms = hall_basis(H_P);

    times = greedy_breakpoints(1e-5);

    float current_t = 0;
    for (int i = 0; i < times.size(); i++){
      float dt = times(i);

      //For each time step, we have to do three quantum walks
      //We first have to find the Legendre expansions for A and B
      Eigen::ArrayXf AB = legendre_coeffs(current_t, dt);

      Eigen::ArrayXf E = commutator_bounds(raw_norms, AB);

      Eigen::ArrayXXf f = opt_f(E);

      //First QW is (f(1,1)*(A_n[0]*H_P + B_n[0]*H_G) - \
            f(1,2)*(A_n[1]*H_P + B_n[1]*H_G) + \
            f(1,3)*(A_n[2]*H_P + B_n[2]*H_G))


      float coeff1 = f(1,1)*AB[0] - f(1,2) * AB[1] + f(1,3)*AB[2];
      float coeff2 = f(1,1)*AB[3] - f(1,2) * AB[4] + f(1,3)*AB[5];

      float gamma = coeff2/coeff1;

      float onenorm = (E_abs + gamma*n);
      double scale = abs(coeff1) * onenorm;
      auto scaled_exp1 = [scale](double x) { return sin(scale * x) + cos(scale * x); };
      ArrayXf coeffs = Chebyshev<double>::RCF_odd_even(scaled_exp1, 1e-6).coeffs.cast<float>().array();

      Clenshaw(coeffs, psi, H_P, gamma, onenorm, psi_real);
      psi_real = false;

      //QW 2

      coeff1 = f(2,1)*AB[0] + f(2,3)*AB[2];
      coeff2 = f(2,1)*AB[3] + f(2,3)*AB[5];

      gamma = coeff2/coeff1;

      onenorm = (E_abs + gamma*n);

      scale = abs(coeff1) * onenorm;

      auto scaled_exp2 = [scale](double x) { return sin(scale * x) + cos(scale * x); };
        
      coeffs = Chebyshev<double>::RCF_odd_even(scaled_exp2, 1e-6).coeffs.cast<float>().array();

      Clenshaw(coeffs, psi, H_P, gamma, onenorm, psi_real);

      //QW 3, same as QW1 but with a plus sign on the f(1,2) terms

      coeff1 = f(1,1)*AB[0] + f(1,2) * AB[1] + f(1,3)*AB[2];
      coeff2 = f(1,1)*AB[3] + f(1,2) * AB[4] + f(1,3)*AB[5];

      gamma = coeff2/coeff1;
      onenorm = (E_abs + gamma*n);
      scale = abs(coeff1) * onenorm;
      auto scaled_exp3 = [scale](double x) { return sin(scale * x) + cos(scale * x); };
        
      coeffs = Chebyshev<double>::RCF_odd_even(scaled_exp3, 1e-6).coeffs.cast<float>().array();

      Clenshaw(coeffs, psi, H_P, gamma, onenorm, psi_real);

      //Approximation errors make this method non-unitary so we renormalise
      psi /= psi.matrix().norm();
      current_t += dt;
    }

  A = A_orig;
  B = B_orig;
  float result = psi[E_loc]*psi[E_loc] + psi[E_loc+N]*psi[E_loc+N];
  outFile.seekp((start + problem) * sizeof(float));
  outFile.write(reinterpret_cast<char*>(&result), sizeof(float));
  std::cout << result << "\n";
  }
}

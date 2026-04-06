#include <eigen3/Eigen/Dense>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "common.hpp"

#define PI 3.1415926535897932384626

using namespace Eigen;

// Build hypercube adjacency matrix: H_G[i][j] = 1 if i,j differ by one bit
MatrixXf build_HG(int n){
  int N = 1 << n;
  MatrixXf H_G = MatrixXf::Zero(N, N);
  for (int i = 0; i < N; i++){
    for (int j = 0; j < n; j++){
      H_G(i, i ^ (1 << j)) = 1.0f;
    }
  }
  return H_G;
}

// Infinite-time success probability for m stages.
// eigs[i] is (N x N) with rows as eigenvectors for stage i.
float P_inf(const std::vector<MatrixXf>& eigs, const VectorXf& Psi0, int e0){
  int m = eigs.size();

  if (m == 1){
    // This is the formula from the Callison paper
    // P_inf = sum_k |<eig_k|e0> * <eig_k|Psi0>|^2
    VectorXf overlaps  = eigs[0] * Psi0;        // <eig_k|Psi0> for each k
    VectorXf e0_comps  = eigs[0].col(e0);       // <e0|eig_k> for each k
    return e0_comps.cwiseProduct(overlaps).squaredNorm();
  }

  // New multi-stage formula from my paper
  // First stage: vals[k] = |<eig_k|Psi0>|^2, mat = eigs^T diag(vals) eigs
  VectorXf vals = (eigs[0] * Psi0).cwiseAbs2();
  MatrixXf mat  = eigs[0].transpose() * (vals.asDiagonal() * eigs[0]);

  // Intermediate stages
  for (int i = 1; i < m - 1; i++){
    vals = eigs[i].cwiseProduct(eigs[i] * mat.transpose()).rowwise().sum();
    mat  = eigs[i].transpose() * (vals.asDiagonal() * eigs[i]);
  }

  // Final stage
  vals = eigs[m-1].cwiseProduct(eigs[m-1] * mat.transpose()).rowwise().sum();
  return vals.dot(eigs[m-1].col(e0).cwiseAbs2());
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

  if (argc >= 6){
    start    = atoi(argv[4]);
    problems = atoi(argv[5]);
  }

  std::string output_dir = (argc >= 7) ? argv[6] : "./results";
  std::string output = output_dir + "/output_" + std::to_string(n) + "_" + std::to_string(m);

  std::ofstream outFile(output, std::ios::binary | std::ios::in | std::ios::out);
  std::ifstream file(filename, std::ios::binary);

  //Seek to beginning, each problem has (n+1)*n/2 parameters, and a double has 8 bytes.
  file.seekg(4*(n+1)*n*start, file.beg);

  //Use actual matrices and vectors so we can use Eigen's functions
  MatrixXf H_G = build_HG(n);
  VectorXf Psi0 = VectorXf::Constant(N, 1.0f / std::sqrt((float)N));

  ArrayXf  H_P(N);
  ArrayXXf J(n, n);
  ArrayXf h(n);

  ArrayXf gammas(m);

  for (unsigned int problem = 0; problem < problems; problem++){
    float E_abs = 0;
    unsigned int E_loc = 0;

    load_problem(file, n, H_P, J, h, E_loc, E_abs);
    
    Psi0.setConstant(1/sqrt(N));

    float HP2 = 2*(J*J).sum() + (h*h).sum();

    gammas = compute_gammas(n,m,HP2);

    // Diagonalise H = (H_P + gamma*H_G) / sqrt(1 + gamma^2) for each stage
    std::vector<MatrixXf> eigs(m);
    MatrixXf H_dense = H_G;  // reuse allocation
    for (int i = 0; i < m; i++){
      float gamma = gammas[i];
      float scale = std::sqrt(1.0f + gamma*gamma);
      H_dense = (H_P.matrix().asDiagonal().toDenseMatrix() - gamma * H_G) / scale;
      SelfAdjointEigenSolver<MatrixXf> solver(H_dense);
      eigs[i] = solver.eigenvectors().transpose();  // rows = eigenvectors
    }

    float result = P_inf(eigs, Psi0, E_loc);
    outFile.seekp((start + problem) * sizeof(float));
    outFile.write(reinterpret_cast<char*>(&result), sizeof(float));
    std::cout << result << "\n\n";
  }
}
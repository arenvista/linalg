#pragma once

// Umbrella header.

#include "linalg/core/Traits.hpp"
#include "linalg/core/Exceptions.hpp"
#include "linalg/core/MatrixView.hpp"
#include "linalg/core/Matrix.hpp"
#include "linalg/core/Vector.hpp"

#include "linalg/ops/Kernels.hpp"
#include "linalg/ops/Norm.hpp"
#include "linalg/ops/Householder.hpp"
#include "linalg/ops/Givens.hpp"

#include "linalg/decomp/LU.hpp"
#include "linalg/decomp/Cholesky.hpp"
#include "linalg/decomp/QR.hpp"
#include "linalg/decomp/Hessenberg.hpp"
#include "linalg/decomp/SVD.hpp"
#include "linalg/decomp/Eigen.hpp"

#include "linalg/solve/TriangularSolver.hpp"
#include "linalg/solve/LinearSolver.hpp"
#include "linalg/solve/LeastSquares.hpp"

#include "linalg/iterative/Preconditioner.hpp"
#include "linalg/iterative/KrylovSolver.hpp"
#include "linalg/iterative/EigenSolver.hpp"

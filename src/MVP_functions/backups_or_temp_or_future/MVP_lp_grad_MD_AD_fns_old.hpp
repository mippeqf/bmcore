
#pragma once


 
 

#include <stan/math/rev.hpp>


#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <stan/math/prim/fun/value_of_rec.hpp>
#include <stan/math/prim/err/check_pos_definite.hpp>
#include <stan/math/prim/err/check_square.hpp>
#include <stan/math/prim/err/check_symmetric.hpp>


#include <stan/math/prim/fun/cholesky_decompose.hpp>
#include <stan/math/prim/fun/sqrt.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/transpose.hpp>
#include <stan/math/prim/fun/dot_product.hpp>
#include <stan/math/prim/fun/norm2.hpp>
#include <stan/math/prim/fun/diagonal.hpp>
#include <stan/math/prim/fun/cholesky_decompose.hpp>
#include <stan/math/prim/fun/eigenvalues_sym.hpp>
#include <stan/math/prim/fun/diag_post_multiply.hpp>




#include <stan/math/prim/prob/multi_normal_cholesky_lpdf.hpp>
#include <stan/math/prim/prob/lkj_corr_cholesky_lpdf.hpp>
#include <stan/math/prim/prob/weibull_lpdf.hpp>
#include <stan/math/prim/prob/gamma_lpdf.hpp>
#include <stan/math/prim/prob/beta_lpdf.hpp>






/// #include <RcppEigen.h>
#include <Eigen/Dense>
 



#include <unsupported/Eigen/SpecialFunctions>


 
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

 
 

 
using namespace Eigen;





#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE



using std_vec_of_EigenVecs_dbl = std::vector<Eigen::Matrix<double, -1, 1>>;
using std_vec_of_EigenVecs_int = std::vector<Eigen::Matrix<int, -1, 1>>;

using std_vec_of_EigenMats_dbl = std::vector<Eigen::Matrix<double, -1, -1>>;
using std_vec_of_EigenMats_int = std::vector<Eigen::Matrix<int, -1, -1>>;

using two_layer_std_vec_of_EigenVecs_dbl =  std::vector<std::vector<Eigen::Matrix<double, -1, 1>>>;
using two_layer_std_vec_of_EigenVecs_int = std::vector<std::vector<Eigen::Matrix<int, -1, 1>>>;

using two_layer_std_vec_of_EigenMats_dbl = std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>;
using two_layer_std_vec_of_EigenMats_int = std::vector<std::vector<Eigen::Matrix<int, -1, -1>>>;


using three_layer_std_vec_of_EigenVecs_dbl =  std::vector<std::vector<std::vector<Eigen::Matrix<double, -1, 1>>>>;
using three_layer_std_vec_of_EigenVecs_int =  std::vector<std::vector<std::vector<Eigen::Matrix<int, -1, 1>>>>;

using three_layer_std_vec_of_EigenMats_dbl = std::vector<std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>>; 
using three_layer_std_vec_of_EigenMats_int = std::vector<std::vector<std::vector<Eigen::Matrix<int, -1, -1>>>>;








 



//////////////////////////////////////////////////----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


void                             fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process(    Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat ,
                                                                                               const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                                               const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                                               const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                                               const std::string &grad_option,
                                                                                               const Model_fn_args_struct &Model_args_as_cpp_struct
                                                                                               //MVP_ThreadLocalWorkspace &MVP_workspace
) { 
  
  
   out_mat.setZero(); //// set log_prob and grad vec to zero at the start (only do this on outer fns, not inner/likelihood fns)
 
  //// important params
  const int N = y_ref.rows();
  const int n_tests = y_ref.cols();
  const int n_us = theta_us_vec_ref.rows()  ;
  const int n_params_main =  theta_main_vec_ref.rows()  ;
  const int n_params = n_params_main + n_us;

  //////////////  access elements from struct and read
  const std::vector<std::vector<Eigen::Matrix<double, -1, -1>>>  &X =  Model_args_as_cpp_struct.Model_args_2_layer_vecs_of_mats_double[0];

  const bool exclude_priors = Model_args_as_cpp_struct.Model_args_bools(0);
  const bool CI =             Model_args_as_cpp_struct.Model_args_bools(1);
  const bool corr_force_positive = Model_args_as_cpp_struct.Model_args_bools(2);
  const bool corr_prior_beta = Model_args_as_cpp_struct.Model_args_bools(3);
  const bool corr_prior_norm = Model_args_as_cpp_struct.Model_args_bools(4);
  const bool handle_numerical_issues = Model_args_as_cpp_struct.Model_args_bools(5);
  const bool skip_checks_exp =   Model_args_as_cpp_struct.Model_args_bools(6);
  const bool skip_checks_log =   Model_args_as_cpp_struct.Model_args_bools(7);
  const bool skip_checks_lse =   Model_args_as_cpp_struct.Model_args_bools(8);
  const bool skip_checks_tanh =  Model_args_as_cpp_struct.Model_args_bools(9);
  const bool skip_checks_Phi =  Model_args_as_cpp_struct.Model_args_bools(10);
  const bool skip_checks_log_Phi = Model_args_as_cpp_struct.Model_args_bools(11);
  const bool skip_checks_inv_Phi = Model_args_as_cpp_struct.Model_args_bools(12);
  const bool skip_checks_inv_Phi_approx_from_logit_prob = Model_args_as_cpp_struct.Model_args_bools(13);
  const bool debug = Model_args_as_cpp_struct.Model_args_bools(14);

  const int n_cores = Model_args_as_cpp_struct.Model_args_ints(0);
  const int n_class = Model_args_as_cpp_struct.Model_args_ints(1);
  const int ub_threshold_phi_approx = Model_args_as_cpp_struct.Model_args_ints(2);
  const int n_chunks = Model_args_as_cpp_struct.Model_args_ints(3);

  const double prev_prior_a = Model_args_as_cpp_struct.Model_args_doubles(0);
  const double prev_prior_b = Model_args_as_cpp_struct.Model_args_doubles(1);
  const double overflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(2);
  const double underflow_threshold = Model_args_as_cpp_struct.Model_args_doubles(3);

  std::string vect_type = Model_args_as_cpp_struct.Model_args_strings(0);
  const std::string &Phi_type = Model_args_as_cpp_struct.Model_args_strings(1);
  const std::string &inv_Phi_type = Model_args_as_cpp_struct.Model_args_strings(2);
  std::string vect_type_exp = Model_args_as_cpp_struct.Model_args_strings(3);
  std::string vect_type_log = Model_args_as_cpp_struct.Model_args_strings(4);
  std::string vect_type_lse = Model_args_as_cpp_struct.Model_args_strings(5);
  std::string vect_type_tanh = Model_args_as_cpp_struct.Model_args_strings(6);
  std::string vect_type_Phi = Model_args_as_cpp_struct.Model_args_strings(7);
  std::string vect_type_log_Phi = Model_args_as_cpp_struct.Model_args_strings(8);
  std::string vect_type_inv_Phi = Model_args_as_cpp_struct.Model_args_strings(9);
  std::string vect_type_inv_Phi_approx_from_logit_prob = Model_args_as_cpp_struct.Model_args_strings(10);
  // const std::string grad_option =  Model_args_as_cpp_struct.Model_args_strings(11);
  const std::string nuisance_transformation =   Model_args_as_cpp_struct.Model_args_strings(12);

  ///// load vectors / matrices
  const Eigen::Matrix<double, -1, 1>  &lkj_cholesky_eta =   Model_args_as_cpp_struct.Model_args_col_vecs_double[0];

  const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct.Model_args_mats_int[0];

  // const Eigen::Matrix<double, -1, -1> &LT_b_priors_shape  = Model_args_as_cpp_struct.Model_args_mats_double[0];
  // const Eigen::Matrix<double, -1, -1> &LT_b_priors_scale  = Model_args_as_cpp_struct.Model_args_mats_double[1];
  // const Eigen::Matrix<double, -1, -1> &LT_known_bs_indicator = Model_args_as_cpp_struct.Model_args_mats_double[2];
  // const Eigen::Matrix<double, -1, -1> &LT_known_bs_values = Model_args_as_cpp_struct.Model_args_mats_double[3];

  const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_coeffs_mean  = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[0];
  const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_coeffs_sd   =  Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[1];
  const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_for_corr_a   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[2];
  const std::vector<Eigen::Matrix<double, -1, -1>>   &prior_for_corr_b   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[3];
  const std::vector<Eigen::Matrix<double, -1, -1>>   &lb_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[4];
  const std::vector<Eigen::Matrix<double, -1, -1>>   &ub_corr   = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[5];
  const std::vector<Eigen::Matrix<double, -1, -1>>   &known_values    = Model_args_as_cpp_struct.Model_args_vecs_of_mats_double[6];

  const std::vector<Eigen::Matrix<int, -1, -1 >> &known_values_indicator = Model_args_as_cpp_struct.Model_args_vecs_of_mats_int[0];

  //////////////
  const int n_corrs =  n_class * n_tests * (n_tests - 1) * 0.5;

  int n_covariates_total_nd, n_covariates_total_d, n_covariates_total;
  int n_covariates_max_nd, n_covariates_max_d, n_covariates_max;

  if (n_class > 1)  {

    n_covariates_total_nd = n_covariates_per_outcome_vec.row(0).sum();
    n_covariates_total_d = n_covariates_per_outcome_vec.row(1).sum();
    n_covariates_total = n_covariates_total_nd + n_covariates_total_d;

    n_covariates_max_nd = n_covariates_per_outcome_vec.row(0).maxCoeff();
    n_covariates_max_d = n_covariates_per_outcome_vec.row(1).maxCoeff();
    n_covariates_max = std::max(n_covariates_max_nd, n_covariates_max_d);

  } else {

    n_covariates_total = n_covariates_per_outcome_vec.sum();
    n_covariates_max = n_covariates_per_outcome_vec.array().maxCoeff();

  }

  const double sqrt_2_pi_recip = 1.0 / sqrt(2.0 * M_PI);
  const double sqrt_2_recip = 1.0 / stan::math::sqrt(2.0);
  const double minus_sqrt_2_recip = -sqrt_2_recip;
  const double a = 0.07056;
  const double b = 1.5976;
  const double a_times_3 = 3.0 * 0.07056;
  const double s = 1.0 / 1.702;

  // //// ---- determine chunk size --------------------------------------------------
  // const int desired_n_chunks = n_chunks;
  // 
  // int vec_size;
  // if (vect_type == "AVX512") {
  //   vec_size = 8;
  // } else  if (vect_type == "AVX2") {
  //   vec_size = 4;
  // } else  if (vect_type == "AVX") {
  //   vec_size = 2;
  // } else {
  //   vec_size = 1;
  // }
  // 
  // ChunkSizeInfo chunk_size_info = calculate_chunk_sizes(N, vec_size, desired_n_chunks);
  // 
  // int chunk_size = chunk_size_info.chunk_size;
  // int chunk_size_orig = chunk_size_info.chunk_size_orig;
  // int normal_chunk_size = chunk_size_info.normal_chunk_size;
  // int last_chunk_size = chunk_size_info.last_chunk_size;
  // int n_total_chunks = chunk_size_info.n_total_chunks;
  // int n_full_chunks = chunk_size_info.n_full_chunks;
  
  
  //// ---- determine chunk size --------------------------
  const int desired_n_chunks = n_chunks;

  int vec_size;
  if (vect_type == "AVX512") {
    vec_size = 8;
  } else  if (vect_type == "AVX2") {
    vec_size = 4;
  } else  if (vect_type == "AVX") {
    vec_size = 2;
  } else {
    vec_size = 1;
  }

  const double N_double = static_cast<double>(N);
  const double vec_size_double =   static_cast<double>(vec_size);
  const double desired_n_chunks_double = static_cast<double>(desired_n_chunks);

  int normal_chunk_size = vec_size_double * std::floor(N_double / (vec_size_double * desired_n_chunks_double));    // Make sure main chunks are divisible by 8
  int n_full_chunks = std::floor(N_double / static_cast<double>(normal_chunk_size));    ///  How many complete chunks we can have
  int last_chunk_size = N_double - (static_cast<double>(n_full_chunks) * static_cast<double>(normal_chunk_size));  //// remainder

  int n_total_chunks;
  if (last_chunk_size == 0) {
    n_total_chunks = n_full_chunks;
  } else {
    n_total_chunks = n_full_chunks + 1;
  }

  int chunk_size = normal_chunk_size; // set initial chunk_size (this may be modified later so non-const)
  int chunk_size_orig = normal_chunk_size;     // store original chunk size for indexing

  if (desired_n_chunks == 1) {
    chunk_size = N;
    chunk_size_orig = N;
    normal_chunk_size = N;
    last_chunk_size = N;
    n_total_chunks = 1;
    n_full_chunks = 1;
  }
  
  
  //////////////  ---------------------------------------------------------------------------------------------------------------------------------
  // corrs
  const Eigen::Matrix<double, -1, 1>  Omega_raw_vec_double = theta_main_vec_ref.head(n_corrs); // .cast<double>();

  // coeffs
  std::vector<Eigen::Matrix<double, -1, -1>> beta_double_array = vec_of_mats_double(n_covariates_max, n_tests, n_class);

  {
    int i = n_corrs;
    for (int c = 0; c < n_class; ++c) {
      for (int t = 0; t < n_tests; ++t) {
        for (int k = 0; k < n_covariates_per_outcome_vec(c, t); ++k) {
          beta_double_array[c](k, t) = theta_main_vec_ref(i);
          i += 1;
        }
      }
    }
  }

  // prev
  double u_prev_diseased = theta_main_vec_ref(n_params_main - 1);

  /////// for autodiff
  double prior_densities_L_Omega_double = 0.0;
  double log_det_J_L_Omega_double = 0.0;
  Eigen::Matrix<double, -1, 1>  grad_Omega_raw_priors_and_log_det_J(n_corrs);

  double prior_densities_prev_double = 0.0;
  double log_det_J_prev_double = 0.0;
  double grad_prev_raw_priors_and_log_det_J = 0.0;

  int dim_choose_2 = n_tests * (n_tests - 1) * 0.5 ;
  std::vector<Eigen::Matrix<double, -1, -1>> deriv_L_wrt_unc_full = vec_of_mats_double(dim_choose_2 + n_tests, dim_choose_2, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_double = vec_of_mats_double(n_tests, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>> L_Omega_recip_double = L_Omega_double;

  
  
  
  {

 
  {     ////////////////////////// local AD block

          stan::math::start_nested();  ////////////////////////

          //// these need to be outside the AD block
          Eigen::Matrix<stan::math::var, -1, 1  >  Omega_raw_vec_var =  stan::math::to_var(Omega_raw_vec_double) ;
          std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  Omega_unconstrained_var = fn_convert_std_vec_of_corrs_to_3d_array_var(Eigen_vec_to_std_vec_var(Omega_raw_vec_var),  n_tests, n_class);
          std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  L_Omega_var = vec_of_mats_var(n_tests, n_tests, n_class);
          stan::math::var target_AD = 0.0;
          std::vector<Eigen::Matrix<stan::math::var, -1, -1 > >  Omega_var   = vec_of_mats_var(n_tests, n_tests, n_class);

          {

                stan::math::var log_det_J_L_Omega = 0.0;

                for (int c = 0; c < n_class; ++c) {

                      Eigen::Matrix<stan::math::var, -1, -1>  ub = stan::math::to_var(ub_corr[c]);
                      Eigen::Matrix<stan::math::var, -1, -1>  lb = stan::math::to_var(lb_corr[c]);
                      Eigen::Matrix<stan::math::var, -1, -1>  Chol_Schur_outs =  Pinkney_LDL_bounds_opt(n_tests, lb, ub, Omega_unconstrained_var[c], known_values_indicator[c], known_values[c]) ;
                      L_Omega_var[c]   =  Chol_Schur_outs.block(1, 0, n_tests, n_tests);
                      Omega_var[c] =   L_Omega_var[c] * L_Omega_var[c].transpose() ;

                      log_det_J_L_Omega +=   Chol_Schur_outs(0, 0); // now can set prior directly on Omega (as this is Jacobian adjustment)

                }

                log_det_J_L_Omega_double += log_det_J_L_Omega.val();
                target_AD += log_det_J_L_Omega;

           }

          {

                stan::math::var prior_densities_L_Omega = 0.0;

                    for (int c = 0; c < n_class; ++c) {

                          if ( (corr_prior_beta == false)   &&  (corr_prior_norm == false) ) {
                            prior_densities_L_Omega +=  stan::math::lkj_corr_cholesky_lpdf(L_Omega_var[c], lkj_cholesky_eta(c)) ;
                          } else if ( (corr_prior_beta == true)   &&  (corr_prior_norm == false) ) {
                            for (int i = 1; i < n_tests; i++) {
                              for (int j = 0; j < i; j++) {
                                prior_densities_L_Omega +=  stan::math::beta_lpdf(  (Omega_var[c](i, j) + 1)/2, prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
                              }
                            }
                            //  Jacobian for  Omega -> L_Omega transformation for prior log-densities (since both LKJ and truncated normal prior densities are in terms of Omega, not L_Omega)
                            Eigen::Matrix<stan::math::var, -1, 1 >  jacobian_diag_elements(n_tests);
                            for (int i = 0; i < n_tests; ++i)     jacobian_diag_elements(i) = ( n_tests + 1 - (i + 1) ) * stan::math::log(L_Omega_var[c](i, i));
                            prior_densities_L_Omega  += + (n_tests * stan::math::log(2) + jacobian_diag_elements.sum());  //  L -> Omega
                          } else if  ( (corr_prior_beta == false)   &&  (corr_prior_norm == true) ) {
                            for (int i = 1; i < n_tests; i++) {
                              for (int j = 0; j < i; j++) {
                                prior_densities_L_Omega +=  stan::math::normal_lpdf(  Omega_var[c](i, j), prior_for_corr_a[c](i, j), prior_for_corr_b[c](i, j));
                              }
                            }
                            Eigen::Matrix<stan::math::var, -1, 1 >  jacobian_diag_elements(n_tests);
                            for (int i = 0; i < n_tests; ++i)     jacobian_diag_elements(i) = ( n_tests + 1 - (i + 1) ) * stan::math::log(L_Omega_var[c](i, i));
                            prior_densities_L_Omega  += + (n_tests * stan::math::log(2) + jacobian_diag_elements.sum());  //  L -> Omega
                          }

                    }

                    target_AD += prior_densities_L_Omega;
                    prior_densities_L_Omega_double += prior_densities_L_Omega.val();

            }


            {
                ///////////////////////
                target_AD.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
                grad_Omega_raw_priors_and_log_det_J =  Omega_raw_vec_var.adj();    // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
                out_mat.segment(1 + n_us, n_corrs) =  grad_Omega_raw_priors_and_log_det_J ;   //// add grad constribution to output vec
                stan::math::set_zero_all_adjoints();
                ////////////////////////////////////////////////////////////
            }


          /////////////  prev stuff  ---- vars
        {
            if (n_class > 1) {  //// if latent class

              std::vector<stan::math::var> 	 u_prev_var_vec_var(n_class, 0.0);
              std::vector<stan::math::var> 	 prev_var_vec_var(n_class, 0.0);
              std::vector<stan::math::var> 	 tanh_u_prev_var(n_class, 0.0);
              Eigen::Matrix<stan::math::var, -1, -1>	 prev_var = Eigen::Matrix<stan::math::var, -1, -1>::Zero(1, 2);
              stan::math::var tanh_pu_deriv_var = 0.0;
              stan::math::var deriv_p_wrt_pu_var = 0.0;
              stan::math::var tanh_pu_second_deriv_var = 0.0;
              stan::math::var log_jac_p_deriv_wrt_pu_var = 0.0;
              stan::math::var log_det_J_prev_var = 0.0;
              stan::math::var target_AD_prev = 0.0;

              u_prev_var_vec_var[1] =  stan::math::to_var(u_prev_diseased);
              tanh_u_prev_var[1] =  stan::math::tanh(u_prev_var_vec_var[1]); /// ( stan::math::exp(2.0*u_prev_var_vec_var[1] ) - 1.0) / ( stan::math::exp(2*u_prev_var_vec_var[1] ) + 1.0) ;
              u_prev_var_vec_var[0] =   0.5 *  stan::math::log( (1.0 + ( (1.0 - 0.5 * ( tanh_u_prev_var[1] + 1.0))*2.0 - 1.0) ) / (1.0 - ( (1.0 - 0.5 * ( tanh_u_prev_var[1] + 1.0))*2.0 - 1.0) ) )  ;
              tanh_u_prev_var[0] =  stan::math::tanh(u_prev_var_vec_var[0]); ///  (stan::math::exp(2.0*u_prev_var_vec_var[0] ) - 1.0) / ( stan::math::exp(2*u_prev_var_vec_var[0] ) + 1.0) ;

              prev_var_vec_var[1] =  0.5 * ( tanh_u_prev_var[1] + 1.0);
              prev_var_vec_var[0] =  0.5 * ( tanh_u_prev_var[0] + 1.0);
              prev_var(0,1) =  prev_var_vec_var[1];
              prev_var(0,0) =  prev_var_vec_var[0];

              tanh_pu_deriv_var = ( 1.0 - (tanh_u_prev_var[1] * tanh_u_prev_var[1])  );
              deriv_p_wrt_pu_var = 0.5 *  tanh_pu_deriv_var;
              tanh_pu_second_deriv_var  = -2.0 * tanh_u_prev_var[1]  * tanh_pu_deriv_var;
              log_jac_p_deriv_wrt_pu_var  = ( 1.0 / deriv_p_wrt_pu_var) * 0.5 * tanh_pu_second_deriv_var; // for gradient of u's

              log_det_J_prev_var =    stan::math::log( deriv_p_wrt_pu_var );
              log_det_J_prev_double =  log_det_J_prev_var.val() ;

              stan::math::var prior_densities_prev = beta_lpdf(prev_var(0, 1), prev_prior_a, prev_prior_b)  ;  // weakly informative prior - helps avoid boundaries with slight negative skew (for lower N)
              prior_densities_prev_double = prior_densities_prev.val();

              //  target_AD_prev += log_det_J_prev_var ; /// done manually later
              target_AD  +=  prior_densities_prev;

              ///////////////////////
              prior_densities_prev.grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
              grad_prev_raw_priors_and_log_det_J  =  u_prev_var_vec_var[1].adj() - u_prev_var_vec_var[0].adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
              out_mat(1 + n_us + n_corrs + n_covariates_total) = grad_prev_raw_priors_and_log_det_J; //// add grad constribution to output vec
              stan::math::set_zero_all_adjoints();

            }
        }

          ////////////////////////////////////////////////////////////
          for (int c = 0; c < n_class; ++c) {
            int cnt_1 = 0;
            for (int k = 0; k < n_tests; k++) {
              for (int l = 0; l < k + 1; l++) {
                (  L_Omega_var[c](k, l)).grad() ;   // differentiating this (i.e. NOT wrt this!! - this is the subject)
                int cnt_2 = 0;
                for (int i = 1; i < n_tests; i++) {
                  for (int j = 0; j < i; j++) {
                    deriv_L_wrt_unc_full[c](cnt_1, cnt_2)  =   Omega_unconstrained_var[c](i, j).adj();     // differentiating WRT this - Note: theta_var_std is the parameter vector - a std::vector of stan::math::var's
                    cnt_2 += 1;
                  }
                }
                stan::math::set_zero_all_adjoints();
                cnt_1 += 1;
              }
            }
          }


          ///////////////// get cholesky factor's (lower-triangular) of corr matrices
          // convert to 3d var array
          for (int c = 0; c < n_class; ++c) {
            for (int t2 = 0; t2 < n_tests; ++t2) { //// col-major storage
              for (int t1 = 0; t1 < n_tests; ++t1) {
                L_Omega_double[c](t1, t2) =   L_Omega_var[c](t1, t2).val()  ;
                L_Omega_recip_double[c](t1, t2) =   1.0 / L_Omega_double[c](t1, t2) ;
              }
            }
          }

          stan::math::recover_memory_nested();  //////////////////////////////////////////

  }   //////////////////////////  end of local AD block
   
  /////////////  prev stuff
  std::vector<double> 	 u_prev_var_vec(n_class, 0.0);
  std::vector<double> 	 prev_var_vec(n_class, 0.0);
  std::vector<double> 	 tanh_u_prev(n_class, 0.0);
  Eigen::Matrix<double, -1, -1>	 prev = Eigen::Matrix<double, -1, -1>::Zero(1, n_class);
  double tanh_pu_deriv = 0.0;
  double deriv_p_wrt_pu_double = 0.0;
  double tanh_pu_second_deriv = 0.0;
  double log_jac_p_deriv_wrt_pu = 0.0;
  double log_jac_p = 0.0;

  if (n_class > 1) {  //// if latent class

        u_prev_var_vec[1] =  (double) u_prev_diseased ;
        tanh_u_prev[1] = stan::math::tanh(u_prev_var_vec[1]);//  ( exp(2.0*u_prev_var_vec[1] ) - 1.0) / ( exp(2.0*u_prev_var_vec[1] ) + 1.0) ;
        u_prev_var_vec[0] =   0.5 *  stan::math::log( (1.0 + ( (1.0 - 0.5 * ( tanh_u_prev[1] + 1.0))*2.0 - 1.0) ) / (1.0 - ( (1.0 - 0.5 * ( tanh_u_prev[1] + 1.0))*2.0 - 1.0) ) )  ;
        tanh_u_prev[0] = stan::math::tanh(u_prev_var_vec[0]); //  (exp(2.0*u_prev_var_vec[0] ) - 1.0) / ( exp(2.0*u_prev_var_vec[0] ) + 1.0) ;

        prev_var_vec[1] =  0.5 * ( tanh_u_prev[1] + 1.0);
        prev_var_vec[0] =  0.5 * ( tanh_u_prev[0] + 1.0);
        prev(0,1) =  prev_var_vec[1];
        prev(0,0) =  prev_var_vec[0];

        tanh_pu_deriv = ( 1.0 - (tanh_u_prev[1] * tanh_u_prev[1])  );
        deriv_p_wrt_pu_double = 0.5 *  tanh_pu_deriv;
        tanh_pu_second_deriv  = -2.0 * tanh_u_prev[1]  * tanh_pu_deriv;
        log_jac_p_deriv_wrt_pu  = ( 1.0 / deriv_p_wrt_pu_double) * 0.5 * tanh_pu_second_deriv; // for gradient of u's
        log_jac_p =    stan::math::log( deriv_p_wrt_pu_double );

  }

  ///////////////////////////////////////////////////////////////////////// prior densities
  double prior_densities = 0.0;

  if (exclude_priors == false) {

        ///////////////////// priors for coeffs
        double prior_densities_coeffs_double = 0.0;
        for (int c = 0; c < n_class; c++) {
          for (int t = 0; t < n_tests; t++) {
            for (int k = 0; k < n_covariates_per_outcome_vec(c, t); k++) {
              prior_densities_coeffs_double  += stan::math::normal_lpdf(beta_double_array[c](k, t), prior_coeffs_mean[c](k, t), prior_coeffs_sd[c](k, t));
            }
          }
        }

        prior_densities += prior_densities_coeffs_double;
        prior_densities += prior_densities_L_Omega_double;
        prior_densities += prior_densities_prev_double;

  }

  ////////  ------- likelihood ("inner") function  --------------------------------------------------------------------------------------------------------------------------
  // Jacobian adjustments (none needed for coeffs as unconstrained - so only for L_Omega -> Omega and u_prev -> prev, and the one for u's is computed in the likelihood)
  const double log_det_J_main = log_det_J_prev_double + log_det_J_L_Omega_double;

  double log_prob_out = 0.0;
  double log_prob = 0.0;

  const Eigen::Matrix<double, -1, -1>  log_prev = stan::math::log(prev);

  //// define unconstrained nuisance parameter vec
  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> u_unc_vec = theta_us_vec_ref;
  
  ///////////////////////////////////////////////
  Eigen::Matrix<double , -1, 1>   beta_grad_vec   =  Eigen::Matrix<double, -1, 1>::Zero(n_covariates_total);  //
  std::vector<Eigen::Matrix<double, -1, -1>> beta_grad_array =  vec_of_mats<double>(n_covariates_max, n_tests, n_class);
  std::vector<Eigen::Matrix<double, -1, -1>> U_Omega_grad_array =  vec_of_mats<double>(n_tests, n_tests, n_class);
  Eigen::Matrix<double, -1, 1> L_Omega_grad_vec(n_corrs + (2 * n_tests));
  Eigen::Matrix<double, -1, 1> U_Omega_grad_vec(n_corrs);
  Eigen::Matrix<double, -1, 1>  prev_unconstrained_grad_vec =   Eigen::Matrix<double, -1, 1>::Zero(n_class); //
  Eigen::Matrix<double, -1, 1>  prev_grad_vec =   Eigen::Matrix<double, -1, 1>::Zero(n_class); //
  Eigen::Matrix<double, -1, -1>  prev_unconstrained_grad_vec_out =   Eigen::Matrix<double, -1, -1>::Zero(n_class - 1, 1); //
  ////////////////////////////////////////////////
   
  {
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::vector<Eigen::Matrix<double, -1, -1>> Z_std_norm = vec_of_mats<double>(chunk_size, n_tests, n_class);
    std::vector<Eigen::Matrix<double, -1, -1>> Bound_Z = Z_std_norm;
    std::vector<Eigen::Matrix<double, -1, -1>> Bound_U_Phi_Bound_Z = Z_std_norm;
    std::vector<Eigen::Matrix<double, -1, -1>> prob = Z_std_norm;
    std::vector<Eigen::Matrix<double, -1, -1>> Phi_Z = Z_std_norm;
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> y1_log_prob =   Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_tests);
    Eigen::Matrix<double, -1, -1> phi_Z_recip = y1_log_prob;
    Eigen::Matrix<double, -1, -1> phi_Bound_Z = y1_log_prob;
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> u_grad_array_CM_chunk = y1_log_prob;
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> common_grad_term_1 = y1_log_prob;
    Eigen::Matrix<double, -1, -1> y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip = y1_log_prob;
    Eigen::Matrix<double, -1, -1> y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip = y1_log_prob;
    Eigen::Matrix<double, -1, -1> prob_rowwise_prod_temp = y1_log_prob;
    Eigen::Matrix<double, -1, -1> prob_recip_rowwise_prod_temp = y1_log_prob;
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1> prod_container_or_inc_array =    Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    Eigen::Matrix<double, -1, 1> derivs_chain_container_vec =  prod_container_or_inc_array;
    Eigen::Matrix<double, -1, 1> prob_rowwise_prod_temp_all =  prod_container_or_inc_array;
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> grad_prob =   y1_log_prob;
    Eigen::Matrix<double, -1, -1> z_grad_term = y1_log_prob;
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> y_chunk = y1_log_prob;
    Eigen::Matrix<double, -1, -1> u_array = y1_log_prob;
    Eigen::Matrix<double, -1, -1> y_sign =  y1_log_prob;
    Eigen::Matrix<double, -1, -1> y_m_y_sign_x_u = y1_log_prob;
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> u_grad_array_CM_chunk_block = y1_log_prob;
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1> u_unc_vec_chunk =   Eigen::Matrix<double, -1, 1>::Zero(chunk_size * n_tests);
    Eigen::Matrix<double, -1, 1> u_vec_chunk =       u_unc_vec_chunk;
    Eigen::Matrix<double, -1, 1> du_wrt_duu_chunk =  u_unc_vec_chunk;
    Eigen::Matrix<double, -1, 1> d_J_wrt_duu_chunk = u_unc_vec_chunk;
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, -1> lp_array = Eigen::Matrix<double, -1, -1>::Zero(chunk_size, n_class);
    ///////////////////////////////////////////////
    double log_jac_u = 0.0;
    ///////////////////////////////////////////////
    Eigen::Matrix<double, -1, 1> log_sum_result =        Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    Eigen::Matrix<double, -1, 1> container_max_logs =    Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    Eigen::Matrix<double, -1, 1> prob_n       =          Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    Eigen::Matrix<double, -1, 1> prob_n_recip       =    Eigen::Matrix<double, -1, 1>::Zero(chunk_size);
    ///////////////////////////////////////////////

    { // start of big local block
       
      for (int nc = 0; nc < n_total_chunks; nc++) {
        
        int chunk_counter = nc; 
        
        if ((chunk_counter == n_full_chunks) && (n_chunks > 1) && (last_chunk_size > 0)) { // Last chunk (remainder - don't use AVX / SIMD for this)

                        chunk_size = last_chunk_size;  //// update chunk_size

                        //// use either Loop (i.e. double fn's) or Stan's vectorisation for the remainder (i.e. last) chunk, regardless of input
                        vect_type = "Stan";
                        vect_type_exp = "Stan";
                        vect_type_log = "Stan";
                        vect_type_lse = "Stan";
                        vect_type_tanh = "Stan";
                        vect_type_Phi =  "Stan";
                        vect_type_log_Phi = "Stan";
                        vect_type_inv_Phi = "Stan";
                        vect_type_inv_Phi_approx_from_logit_prob = "Stan";


                        ///////////////////////////////////////////////
                        for (int c = 0; c < n_class; c++) {
                          Z_std_norm[c].resize(last_chunk_size, n_tests);
                          Bound_Z[c].resize(last_chunk_size, n_tests);
                          Bound_U_Phi_Bound_Z[c].resize(last_chunk_size, n_tests);
                          prob[c].resize(last_chunk_size, n_tests);
                          Phi_Z[c].resize(last_chunk_size, n_tests);
                        }
                        ///////////////////////////////////////////////
                        y1_log_prob.resize(last_chunk_size, n_tests);
                        phi_Z_recip.resize(last_chunk_size, n_tests);
                        phi_Bound_Z.resize(last_chunk_size, n_tests);
                        ///////////////////////////////////////////////
                        u_grad_array_CM_chunk.resize(last_chunk_size, n_tests);
                        ///////////////////////////////////////////////
                        common_grad_term_1.resize(last_chunk_size, n_tests);
                        y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip.resize(last_chunk_size, n_tests);
                        y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip.resize(last_chunk_size, n_tests);
                        prob_rowwise_prod_temp.resize(last_chunk_size, n_tests);
                        prob_recip_rowwise_prod_temp.resize(last_chunk_size, n_tests);
                        ///////////////////////////////////////////////
                        prod_container_or_inc_array.resize(last_chunk_size);
                        derivs_chain_container_vec.resize(last_chunk_size);
                        prob_rowwise_prod_temp_all.resize(last_chunk_size);
                        ///////////////////////////////////////////////
                        grad_prob.resize(last_chunk_size, n_tests);
                        z_grad_term.resize(last_chunk_size, n_tests);
                        ///////////////////////////////////////////////
                        y_chunk.resize(last_chunk_size, n_tests);
                        u_array.resize(last_chunk_size, n_tests);
                        y_sign.resize(last_chunk_size, n_tests);
                        y_m_y_sign_x_u.resize(last_chunk_size, n_tests);
                        ///////////////////////////////////////////////
                        u_grad_array_CM_chunk_block.resize(last_chunk_size, n_tests);
                        ///////////////////////////////////////////////
                        u_unc_vec_chunk.resize(last_chunk_size * n_tests);
                        u_vec_chunk.resize(last_chunk_size * n_tests);
                        du_wrt_duu_chunk.resize(last_chunk_size * n_tests);
                        d_J_wrt_duu_chunk.resize(last_chunk_size * n_tests);
                        ///////////////////////////////////////////////
                        lp_array.resize(last_chunk_size, n_class);
                        ///////////////////////////////////////////////
                        log_sum_result.resize(last_chunk_size);
                        container_max_logs.resize(last_chunk_size);
                        prob_n.resize(last_chunk_size);
                        prob_n_recip.resize(last_chunk_size);
                        ///////////////////////////////////////////////

        }
        
        u_grad_array_CM_chunk.setZero(); //// reset to 0

        y_chunk = y_ref.middleRows( chunk_size_orig * chunk_counter , chunk_size).array().cast<double>() ;

        //// Nuisance parameter transformation step
        u_unc_vec_chunk = u_unc_vec.segment( chunk_size_orig * n_tests * chunk_counter , chunk_size * n_tests);

        fn_MVP_compute_nuisance(    u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);
        log_jac_u +=    fn_MVP_compute_nuisance_log_jac_u(   u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);

        u_array  =  u_vec_chunk.reshaped(chunk_size, n_tests).array();
        y_sign =    ( (y_chunk.array()  + (y_chunk.array() - 1.0)) ).matrix();
        y_m_y_sign_x_u = ( y_chunk.array()  - y_sign.array() * u_array.array() ).matrix();
        
        {
          //// START of c loop
          for (int c = 0; c < n_class; c++) {

                  prod_container_or_inc_array.setZero(); //// reset to 0
      
                  //// start of t loop
                  for (int t = 0; t < n_tests; t++) {
      
                          if (n_covariates_max > 1) {
                              Eigen::Matrix<double, -1, 1>    Xbeta_given_class_c_col_t = X[c][t].block(chunk_size_orig * chunk_counter, 0, chunk_size, n_covariates_per_outcome_vec(c, t)).cast<double>()  *
                              beta_double_array[c].col(t).head(n_covariates_per_outcome_vec(c, t));
                              Bound_Z[c].col(t).array() =     L_Omega_recip_double[c](t, t) * (  - ( Xbeta_given_class_c_col_t.array()    +      prod_container_or_inc_array.array()   )  ) ;
                          } else {  // intercept-only
                              Bound_Z[c].col(t).array() =     L_Omega_recip_double[c](t, t) * (  - ( beta_double_array[c](0, t) +      prod_container_or_inc_array.array()   )  ) ;
                          }
                          
                          //// compute/update important log-lik quantities for GHK-MVP
                          fn_MVP_compute_lp_GHK_cols(  t,
                                                       Bound_U_Phi_Bound_Z[c],
                                                       Phi_Z[c],
                                                       Z_std_norm[c],
                                                       prob[c],
                                                       y1_log_prob,
                                                       Bound_Z[c],
                                                       y_chunk,
                                                       u_array,
                                                       Model_args_as_cpp_struct);
                        
                          if (t < n_tests - 1)       prod_container_or_inc_array.array()  =   ( Z_std_norm[c].leftCols(t + 1)  *   ( L_Omega_double[c].row(t+1).head(t+1).transpose()  ) ) ;
      
                  }   /// / end of t loop
      
                  if (n_class > 1) { // if latent class
                    lp_array.col(c).array() =     y1_log_prob.rowwise().sum().array() + log_prev(0, c) ;
                  } else {
                    lp_array.col(0).array() =     y1_log_prob.rowwise().sum().array();
                  }

          }  //// end of c loop

        }
          
        if (n_class > 1) {
          
                // log_sum_exp_general(lp_array, 
                //                     vect_type_exp, 
                //                     vect_type_log, 
                //                     log_sum_result,
                //                     container_max_logs);
                // 
                // out_mat.tail(N).segment(chunk_size_orig * chunk_counter, chunk_size) = log_sum_result;
                
                out_mat.tail(N).segment(chunk_size_orig * chunk_counter, chunk_size).array()   = fn_log_sum_exp_2d_double(lp_array,  vect_type_lse).array() ;
         
        } else {
          
            out_mat.tail(N).segment(chunk_size_orig * chunk_counter, chunk_size).array()   = lp_array.col(0);
          
        }
        
        // const Eigen::Matrix<double, -1, 1> &prob_n  =  fn_EIGEN_double(out_mat.tail(N).segment(chunk_size_orig * chunk_counter, chunk_size), "exp",  vect_type_exp);
        prob_n  =  fn_EIGEN_double(out_mat.tail(N).segment(chunk_size_orig * chunk_counter, chunk_size), "exp",  vect_type_exp);
        prob_n_recip  = 1.0 / prob_n.array(); // this CANNOT be a temporary otherwise it created a "dangling reference", since prob_n is a temporary!!
        
        /////////////////  ------------------------- compute grad  ---------------------------------------------------------------------------------
        for (int c = 0; c < n_class; c++) {
          
          const Eigen::Matrix<double, -1, -1> &prob_recip = 1.0 / prob[c].array(); // this should be fine since prob[c] isn't a temporary
          
          //// compute/update important log-lik quantities for GHK-MVP
          for (int t = 0; t < n_tests; t++) {

                fn_MVP_compute_phi_Z_recip_cols(   t,
                                                   phi_Z_recip, // computing this
                                                   Phi_Z[c], Z_std_norm[c], Model_args_as_cpp_struct);

                fn_MVP_compute_phi_Bound_Z_cols(   t,
                                                   phi_Bound_Z, // computing this
                                                   Bound_U_Phi_Bound_Z[c], Bound_Z[c], Model_args_as_cpp_struct);
          }
          
          
          if (grad_option != "none") { // not the issue

            fn_MVP_grad_prep(       prob[c],
                                    y_sign,
                                    y_m_y_sign_x_u,
                                    L_Omega_recip_double[c],
                                    prev(0, c),
                                    prob_n_recip,
                                    phi_Z_recip,
                                    phi_Bound_Z,
                                    prob_recip,
                                    prob_rowwise_prod_temp,
                                    prob_recip_rowwise_prod_temp,
                                    prob_rowwise_prod_temp_all,
                                    common_grad_term_1,
                                    y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                    y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                    Model_args_as_cpp_struct) ;

          }
          
          ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of nuisance parameters / u's (manual)
          if ( (grad_option == "us_only") || (grad_option == "all") ) {

            u_grad_array_CM_chunk_block =  u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests);

            fn_MVP_compute_nuisance_grad_v2(  u_grad_array_CM_chunk_block,
                                              phi_Z_recip,
                                              common_grad_term_1,
                                              L_Omega_double[c],
                                              prob[c],
                                              prob_recip,
                                              prob_rowwise_prod_temp,
                                              y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                              y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                              z_grad_term,
                                              grad_prob,
                                              prod_container_or_inc_array,
                                              derivs_chain_container_vec,
                                              Model_args_as_cpp_struct);

            u_grad_array_CM_chunk.block(0, 0, chunk_size, n_tests).matrix() += u_grad_array_CM_chunk_block;

            //// update output vector once all u_grad computations are done
            out_mat.segment(1, n_us).segment(chunk_size_orig * n_tests * chunk_counter , chunk_size * n_tests).array()  =  u_grad_array_CM_chunk.reshaped();

            //// account for unconstrained -> constrained transformations and Jacobian adjustments
            fn_MVP_nuisance_first_deriv(  du_wrt_duu_chunk,
                                          u_vec_chunk, u_unc_vec_chunk, Model_args_as_cpp_struct);

            fn_MVP_nuisance_deriv_of_log_det_J(    d_J_wrt_duu_chunk,
                                                   u_vec_chunk, u_unc_vec_chunk, du_wrt_duu_chunk, Model_args_as_cpp_struct);

            out_mat.segment(1, n_us).segment(chunk_size_orig * n_tests * chunk_counter , chunk_size * n_tests).array() =
                     out_mat.segment(1, n_us).segment(chunk_size_orig * n_tests * chunk_counter , chunk_size * n_tests).array() * du_wrt_duu_chunk.array() + d_J_wrt_duu_chunk.array();

          }
          
          
          /////////////////////////////////////////////////////////////////////////// Grad of intercepts / coefficients (beta's)
          if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "coeff_only") ) {

            //// Eigen::Matrix<int, -1, 1> n_covariates_per_outcome_vec_temp =   n_covariates_per_outcome_vec.row(c).transpose();

            fn_MVP_compute_coefficients_grad_v3(     c,
                                                     beta_grad_array[c],
                                                     chunk_counter,
                                                     n_covariates_max,
                                                     common_grad_term_1,
                                                     L_Omega_double[c],
                                                     prob[c],
                                                     prob_recip,
                                                     prob_rowwise_prod_temp,
                                                     y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                     y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                     z_grad_term,
                                                     grad_prob,
                                                     prod_container_or_inc_array,
                                                     derivs_chain_container_vec,
                                                     true,  ///   compute_final_scalar_grad,
                                                     Model_args_as_cpp_struct);

          }
          
          /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Grad of L_Omega ('s)
          if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "corr_only") ) {

            fn_MVP_compute_L_Omega_grad_v3(      U_Omega_grad_array[c],
                                                 common_grad_term_1,
                                                 L_Omega_double[c],
                                                 prob[c],
                                                 prob_recip,
                                                 Bound_Z[c],
                                                 Z_std_norm[c],
                                                 prob_rowwise_prod_temp,
                                                 y_sign_chunk_times_phi_Bound_Z_x_L_Omega_diag_recip,
                                                 y_m_ysign_x_u_array_times_phi_Z_times_phi_Bound_Z_times_L_Omega_diag_recip,
                                                 z_grad_term,
                                                 grad_prob,
                                                 prod_container_or_inc_array,
                                                 derivs_chain_container_vec,
                                                 true,  ///   compute_final_scalar_grad,
                                                 Model_args_as_cpp_struct);


          }
          
          if (n_class > 1) { /// prevelance only estimated for latent class models

            if ( (grad_option == "main_only") || (grad_option == "all") || (grad_option == "prev_only" ) ) {
              
              Eigen::Matrix<double, -1, -1> abs_vals(prob[c].rows(), prob[c].cols());
              Eigen::Matrix<double, -1, -1> log_vals(prob[c].rows(), prob[c].cols());
              Eigen::Matrix<double, -1, 1> log_prod_prob(prob[c].rows());
              Eigen::Matrix<double, -1, 1> log_prev_grad_n(prob[c].rows());
              Eigen::Matrix<double, -1, 1> prev_grad_n(prob[c].rows());
              const double eps = 1e-10;
              
              abs_vals  = (prob[c].array().abs() + eps);
              log_vals = abs_vals.log();
              log_prod_prob = log_vals.rowwise().sum();
              log_prev_grad_n = prob_n_recip + log_prod_prob;
              prev_grad_n =   fn_EIGEN_double(log_prev_grad_n, "exp",  vect_type_exp);
              prev_grad_vec(c)  +=  prev_grad_n.sum();
              
               //     Eigen::Matrix<double, -1, 1> log_prod_prob = prob[c].array().abs().log().rowwise().sum();
                //  Eigen::Matrix<double, -1, 1> log_prev_grad_n = prob_n_recip + log_prod_prob;
                //  Eigen::Matrix<double, -1, 1> prev_grad_n =   fn_EIGEN_double(log_prev_grad_n, "exp",  vect_type_exp);
                //  prev_grad_vec(c)  +=  prev_grad_n.sum()  ;

            }

          }
          
        }
        
      }
      
    }
    
    

    //////////////////////// gradients for latent class membership probabilitie(s) (i.e. disease prevalence)
    if (n_class > 1) {
      for (int c = 0; c < n_class; c++) {
        prev_unconstrained_grad_vec(c)  =   prev_grad_vec(c)   * deriv_p_wrt_pu_double ;
      }
      prev_unconstrained_grad_vec(0) = prev_unconstrained_grad_vec(1) - prev_unconstrained_grad_vec(0) - 2 * tanh_u_prev[1];
      prev_unconstrained_grad_vec_out(0, 0) = prev_unconstrained_grad_vec(0);
    }

    log_prob_out +=  out_mat.tail(N).sum();       //  log_lik
    log_prob_out +=  log_jac_u;

    if (exclude_priors == false)  log_prob_out += prior_densities;

    log_prob_out +=  log_det_J_main ; // log_jac_p_double;

    log_prob = (double) log_prob_out;
    
    {

      int i = 0;
      for (int c = 0; c < n_class; c++ ) {
        for (int t = 0; t < n_tests; t++) {
          for (int k = 0; k <  n_covariates_per_outcome_vec(c, t); k++) {
            beta_grad_vec(i) = beta_grad_array[c](k, t);
            i += 1;
          }
        }
      }

    }
    
    {

      int i = 0;
      for (int c = 0; c < n_class; c++) {
        for (int t1 = 0; t1 < n_tests; t1++) {
          for (int t2 = 0; t2 <  t1 + 1; t2++) {
            L_Omega_grad_vec(i) = U_Omega_grad_array[c](t1,t2);
            i += 1;
          }
        }
      }

    }
    
    Eigen::Matrix<double, -1, 1>  grad_wrt_L_Omega_nd(dim_choose_2 + n_tests);
    Eigen::Matrix<double, -1, 1>  grad_wrt_L_Omega_d(dim_choose_2 + n_tests);

    if (n_class > 1) {
      grad_wrt_L_Omega_nd =   L_Omega_grad_vec.segment(0, dim_choose_2 + n_tests);
      grad_wrt_L_Omega_d =   L_Omega_grad_vec.segment(dim_choose_2 + n_tests, dim_choose_2 + n_tests);
      U_Omega_grad_vec.head(dim_choose_2) =  ( grad_wrt_L_Omega_nd.transpose()  *  deriv_L_wrt_unc_full[0]  ).transpose() ;
      U_Omega_grad_vec.segment(dim_choose_2, dim_choose_2) =   ( grad_wrt_L_Omega_d.transpose()  *  deriv_L_wrt_unc_full[1] ).transpose()  ;
    } else {
      grad_wrt_L_Omega_nd =   L_Omega_grad_vec.segment(0, dim_choose_2 + n_tests);
      U_Omega_grad_vec.head(dim_choose_2) =  ( grad_wrt_L_Omega_nd.transpose()  *  deriv_L_wrt_unc_full[0] ).transpose() ;
    }
    
  }
  

  {   ////////////////////////////  outputs // add log grad and sign stuff';///////////////
    out_mat(0) =  log_prob;
    out_mat.segment(1 + n_us, n_corrs) += U_Omega_grad_vec ;
    out_mat.segment(1 + n_us + n_corrs, n_covariates_total) += beta_grad_vec ;  /// no Jacobian needed
    out_mat(1 + n_us + n_corrs + n_covariates_total) += prev_unconstrained_grad_vec_out(0, 0) ;  
  }

  
  }


  // add derivative of normal prior density to beta / coeffs gradient
  {
    int i = n_us + n_corrs + 1; /// + 1 because first element is the log_prob !!
    for (int c = 0; c < n_class; c++) {
      for (int t = 0; t < n_tests; t++) {
        for (int k = 0; k <  n_covariates_per_outcome_vec(c, t); k++) {
          if (exclude_priors == false) {
              out_mat(i) += - ((beta_double_array[c](k, t) - prior_coeffs_mean[c](k, t)) / prior_coeffs_sd[c](k, t) ) * (1.0 / prior_coeffs_sd[c](k, t) ) ;
              i += 1;
          }
        }
      }
    }
  }

 
 
  
}





























// Internal function using Eigen::Ref as inputs for matrices
void     fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_InPlace(    Eigen::Matrix<double, -1, 1> &&out_mat_R_val,
                                                               const Eigen::Matrix<double, -1, 1> &&theta_main_vec_R_val,
                                                               const Eigen::Matrix<double, -1, 1> &&theta_us_vec_R_val,
                                                               const Eigen::Matrix<int, -1, -1> &&y_R_val,
                                                               const std::string &grad_option,
                                                               const Model_fn_args_struct &Model_args_as_cpp_struct
                                                              // MVP_ThreadLocalWorkspace &MVP_workspace




) {


  Eigen::Ref<Eigen::Matrix<double, -1, 1>> out_mat_ref(out_mat_R_val);
  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref(theta_main_vec_R_val);  // create Eigen::Ref from R-value
  const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref(theta_us_vec_R_val);  // create Eigen::Ref from R-value
  const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref(y_R_val);  // create Eigen::Ref from R-value


  fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process(  out_mat_ref,
                                                              theta_main_vec_ref,
                                                              theta_us_vec_ref,
                                                              y_ref,
                                                              grad_option,
                                                              Model_args_as_cpp_struct
                                                             // MVP_workspace
                                                              );


}








// Internal function using Eigen::Ref as inputs for matrices
void     fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_InPlace(    Eigen::Matrix<double, -1, 1> &out_mat_ref,
                                                               const Eigen::Matrix<double, -1, 1> &theta_main_vec_ref,
                                                               const Eigen::Matrix<double, -1, 1> &theta_us_vec_ref,
                                                               const Eigen::Matrix<int, -1, -1> &y_ref,
                                                               const std::string &grad_option,
                                                               const Model_fn_args_struct &Model_args_as_cpp_struct
                                                           //    MVP_ThreadLocalWorkspace &MVP_workspace




) {


  fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process(  out_mat_ref,
                                                              theta_main_vec_ref,
                                                              theta_us_vec_ref,
                                                              y_ref,
                                                              grad_option,
                                                              Model_args_as_cpp_struct
                                                              //MVP_workspace
                                                              );


}





// Internal function using Eigen::Ref as inputs for matrices
template <typename MatrixType>
void     fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_InPlace(    Eigen::Ref<Eigen::Block<MatrixType, -1, 1>>  &out_mat_ref,
                                                               const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_main_vec_ref,
                                                               const Eigen::Ref<const Eigen::Block<MatrixType, -1, 1>>  &theta_us_vec_ref,
                                                               const Eigen::Matrix<int, -1, -1> &y_ref,
                                                               const std::string &grad_option,
                                                               const Model_fn_args_struct &Model_args_as_cpp_struct
                                                            //   MVP_ThreadLocalWorkspace &MVP_workspace




) {


  fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_Inplace_process(  out_mat_ref,
                                                              theta_main_vec_ref,
                                                              theta_us_vec_ref,
                                                              y_ref,
                                                              grad_option,
                                                              Model_args_as_cpp_struct
                                                             // MVP_workspace
                                                              );


}














// Internal function using Eigen::Ref as inputs for matrices
Eigen::Matrix<double, -1, 1>    fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD(   const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_main_vec_ref,
                                                                             const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> theta_us_vec_ref,
                                                                             const Eigen::Ref<const Eigen::Matrix<int, -1, -1>> y_ref,
                                                                             const std::string &grad_option,
                                                                             const Model_fn_args_struct &Model_args_as_cpp_struct
                                                                         //    MVP_ThreadLocalWorkspace &MVP_workspace




) {

  int n_params_main = theta_main_vec_ref.rows();
  int n_us = theta_us_vec_ref.rows();
  int n_params = n_us + n_params_main;
  int N = y_ref.rows();

  Eigen::Matrix<double, -1, 1> out_mat = Eigen::Matrix<double, -1, 1>::Zero(1 + N + n_params);

  fn_lp_grad_MVP_LC_Pinkney_NoLog_MD_and_AD_InPlace(  out_mat,
                                                          theta_main_vec_ref,
                                                          theta_us_vec_ref,
                                                          y_ref,
                                                          grad_option,
                                                          Model_args_as_cpp_struct
                                                        //  MVP_workspace
                                                          );

  return out_mat;

}


 








  
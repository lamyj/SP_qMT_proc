#ifndef _c7e42f6f_c110_43e2_8076_f7ae3e40af0f
#define _c7e42f6f_c110_43e2_8076_f7ae3e40af0f

#include "expm.h"

#include <cmath>
#include <xtensor/xbuilder.hpp>

template<typename T>
auto expm_2_2(T const & A)
{
    auto const Delta = 
        std::pow(A.unchecked(0, 0)-A.unchecked(1, 1), 2)
        + 4*A.unchecked(0, 1)*A.unchecked(1, 0);
    
    auto result = xt::empty<typename T::value_type>(A.shape());
    
    static auto const I = xt::eye<typename T::value_type>({2, 2}, 0);
    
    // The Cayley-Hamilton theorem states that every square matrix satisfies its
    // own characteristic polynomial.
    // 
    // Moreover, for any scalar k, exp(At) = exp(-kt) exp((A+kI)t) [1]
    // Proof: since k is a scalar, then exp(kI) = I exp(k) and 
    // exp((A+kI)t) = exp(At)exp(ktI) = exp(At)exp(kt)
    // 
    // For each case of the spectrum of A, there is a value of k which allows us
    // to use [1] in order to compute the matrix exponential.
    // 
    // Source: https://smashmath.github.io/math/2x2ezmatrixexp/
    
    auto const a = (A.unchecked(0, 0)+A.unchecked(1, 1))/2.;
    
    if(Delta == 0.)
    {
        // A has a defective eigenvalue. Its characteristic polynomial is then
        // (t-λ)². By Cayley-Hamilton theorem, (A-λI)²=0. All terms of order ≥ 2
        // in the power series (A-λI) are 0 and exp(t(A-λI)) = I + t(A-λI)
        // Let k=-λ in [1], we get 
        // exp(At) = exp(λt) (I + t(A-λI)) = exp(λt) (tA + t(1-λ)I)
        auto const lambda = a;
        result = std::exp(lambda) * (A+(1-lambda)*I);
    }
    else if(Delta < 0)
    {
        // A has two conjugate complex eigenvalues a±bi.
        // Characteristic polynomial: t² - 2at + a² + b² = (t-a)² + b²
        // By CH: (A - aI)² = -b²I
        // Let B = A - aI
        // Even powers of B take the form B²ᵏ = -1ᵏb²ᵏI
        // Odd powers of B take the form B²ᵏ⁺¹ = -1ᵏb²ᵏB
        // By splitting the power series in its even and odd terms, we get
        // exp(Bt) = Σ -1ᵏb²ᵏt²ᵏ/((2k)!) I + Σ -1ᵏb²ᵏt²ᵏ⁺¹/((2k+1)!) B
        // or exp(Bt) = Σ -1ᵏ(bt)²ᵏ/((2k)!) I + 1/b Σ -1ᵏ(bt)²ᵏ⁺¹/((2k+1)!) B
        // The left term is the power series of cos, while the second is the
        // power series of sin, hence
        // exp(Bt) = cos(bt) I + 1/b sin(bt) B
        // Let k=-a in [1], we get 
        // exp(At) = exp(at)(cos(bt)I + 1/b sin(bt)(A-aI))
        //         = exp(at)((cos(bt)-a sin(bt)/b)I + sin(bt)/bA )
        auto const b = std::sqrt(-Delta)/2.;
        
        result = std::exp(a)*((std::cos(b)-a*std::sin(b)/b)*I + std::sin(b)/b*A);
    }
    else
    {
        // A has two real eigenvalues
        auto const b = std::sqrt(Delta)/2.;
        auto const lambda_1 = a + b;
        auto const lambda_2 = a - b;
        
        if(std::fabs(a) < 1e-6)
        {
            // If Tr(A) = 0, then λ1 = -λ2 = λ
            // Characteristic polynomial (t-λ)(t+λ) = t²-λ²
            // By CH: A² = λ²I
            // As above, we get 
            // exp(At) = Σ λ²ᵏt²ᵏ/((2k)!) I + Σ λ²ᵏt²ᵏ⁺¹/((2k+1)!) A
            // or exp(At) = Σ λ²ᵏt²ᵏ/((2k)!) I + 1/λ Σ (λt)²ᵏ⁺¹/((2k+1)!) A
            // The left term is the power series of cosh, while the second is the
            // power series of sinh, hence
            // exp(At) = cosh(λt) I + 1/λ sinh(λt) A
            result = std::cosh(lambda_1)*I + 1/lambda_1*std::sinh(lambda_1)*A;
        }
        else if(lambda_1 * lambda_2 == 0)
        {
            // If Tr(A) ≠ 0 and A has rank 1, then only one of λ₁,λ₂ is 0, called
            // λ and Tr(A) = λ₁+λ₂ = λ
            // Characteristic polynomial: (t-λ)⋅t
            // By CH: (A-λ)⋅A = 0 ⇔ A² = λ⋅A
            // Hence A³ = A²⋅A = λ⋅A⋅A = λ⋅A² = λ²⋅A and
            // for n≥1, Aⁿ = λⁿ⁻¹⋅A
            // The power series can then be written as
            // exp(At) = I + Σ λⁿ⁻¹⋅tⁿ / n! A = I + A Σ λⁿ⁻¹⋅tⁿ / n!
            //         = I + A/λ Σ (λ⋅t)ⁿ / n! 
            //         = I + A/λ (exp(λ⋅t) - 1)
            auto const lambda = 2.*a;
            result = I + A/lambda*(std::exp(lambda)-1.);
        }
        else
        {
            // Generic case with two real eigenvalues λ₁, λ₂
            // Let B = A-λ₁I, which is by definition singular, with eigenvalue 
            // λ₂-λ₁, called λ. Rewriting exp(At) = exp(λ₁t) exp(Bt), we can 
            // apply the previous result to get
            // exp(Bt) = I + B/λ (exp(λt) - 1)
            // and finally
            // exp(At) = exp(λ₁t) (I + B/λ (exp(λt) - 1))
            // Rewrite using definition of B and λ:
            //         = exp(λ₁t) (I + (A-λ₁I)/(λ₂-λ₁) (exp((λ₂-λ₁) t) - 1))
            // "Factorize" 1/(λ₂-λ₁) outside the right factor:
            //         = exp(λ₁t)/(λ₂-λ₁) ( (λ₂-λ₁)I + (A-λ₁I) (exp((λ₂-λ₁) t) - 1) )
            // Expand exp((λ₂-λ₁) t):
            //         = exp(λ₁t)/(λ₂-λ₁) ( (λ₂-λ₁)I + (A-λ₁I) (exp(λ₂t)exp(-λ₁t) - 1) )
            // Distribute exp(λ₁t) inside the right factor:
            //         = 1/(λ₂-λ₁) ( exp(λ₁t) (λ₂-λ₁)I + (A-λ₁I) (exp(λ₂t) - exp(λ₁t)) )
            // Distribute the terms inside the right factor:
            //         = 1/(λ₂-λ₁) ( 
            //             λ₂exp(λ₁t)I - λ₁exp(λ₁t)I 
            //             + exp(λ₂t)A - exp(λ₁t)A - λ₁exp(λ₂t)I + λ₁exp(λ₁t)I )
            // The two terms λ₁exp(λ₁t)I cancel:
            //         = 1/(λ₂-λ₁) ( λ₂exp(λ₁t)I + exp(λ₂t)A - exp(λ₁t)A - λ₁exp(λ₂t)I )
            // Factorize I and A inside the right factor:
            //         = 1/(λ₂-λ₁) ( (λ₂exp(λ₁t) - λ₁exp(λ₂t))I + (exp(λ₂t)-exp(λ₁t))A )
            
            result = 
                1/(lambda_2-lambda_1)
                * (
                    (lambda_2*std::exp(lambda_1) - lambda_1*std::exp(lambda_2))*I
                    + (std::exp(lambda_2)-std::exp(lambda_1))*A);
        }
    }
    
    return result;
}

#endif // _c7e42f6f_c110_43e2_8076_f7ae3e40af0f

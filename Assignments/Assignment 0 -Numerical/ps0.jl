################### Problem Set 0 for Chris Conlon's Empirical IO course ###################

# Deadline: Tuesday, September 9, 2025


# Part 0: Logit Inclusive Value

# 1. Show that the this function is everywhere convex if $x_0=0$.

# See Latex answers for quick proof.


# 2. A common problem in practice is that if one of the $x_i > 600$ that we have an ``overflow'' error
# on a computer. In this case $\exp[600] \approx 10^{260}$ which is too large to store with any real precision,
#  especially if another $x$ has a different scale (say $x_2=10$). A common ``trick'' is to subtract off
#  $m_i = \max_i x_i$ from all $x_i$.  Show how to implement the trick and get the correct value of $IV$. 
# If you get stuck take a look at Wikipedia.

# Suppose:
x = [10, 11, 12, 1000]


# Naive version:
IV = log(sum(exp.(x)))
println("Naive version: $IV") #exp(1000) overflows to Inf, so whole thing is Inf

# Trick version:
m = maximum(x)
IV = log(sum(exp.(x .- m))) + m
println("Trick version: $IV") # avoids overflow; we get to right answer


# 3. Compare your function to $\mathtt{scipy.special.logsumexp}$.
#    Does it appear to suffer from underflow/overflow? Does it use the $\max$ trick?

# I'll compare to the Julia equivalent.

using StatsFuns

IV_statsfuns = logsumexp(x)
println("StatsFuns version: $IV_statsfuns") #matches trick version


# Part 1: Markov Chains

using LinearAlgebra, Printf

"""
    stationary_dist_via_eig(P; atol=1e-10)

Compute the stationary distribution π for a row-stochastic transition matrix P
by extracting the left eigenvector associated with eigenvalue 1 and normalizing.
Returns a probability vector π (1×n).
"""
function stationary_dist_via_eig(P::AbstractMatrix{<:Real}; atol=1e-10)
    # Basic sanity checks
    size(P, 1) == size(P, 2) || error("P must be square")
    rowsums = sum(P, dims=2)
    maximum(abs.(rowsums .- 1)) < 1e-8 || @warn "P rows do not sum to 1 within tolerance"

    # Left eigenvector of P for λ=1 is the eigenvector of P' for λ=1
    F = eigen(Matrix(transpose(P)))  # real matrix → possibly complex eigensystem

    # Find eigenvalue closest to 1
    idx = argmin(abs.(F.values .- 1))
    v = real.(F.vectors[:, idx])     # take the corresponding eigenvector, real part

    # Ensure a proper probability vector: nonnegative and sums to 1
    # If vector has mixed signs due to arbitrary scaling, flip by dominant sign
    if sum(v) < 0
        v .= -v
    end
    v .-= minimum(v) .* (minimum(v) < 0)  # shift up if any tiny negatives
    s = sum(v)
    s > atol || error("Eigenvector for λ≈1 is numerically degenerate")
    π = (v ./ s)'

    # Small cleanup: clamp tiny negatives to zero, renormalise
    π .= max.(π, 0.0)
    π ./= sum(π)

    return π
end

"""
    compare_to_power(P; k=100)

Compute stationary distribution π and P^k, and report how close each row of P^k is to π.
Returns (π, Pk, errs) where errs[j] = norm(Pk[j, :] - π, 1).
"""
function compare_to_power(P::AbstractMatrix{<:Real}; k::Integer=100)
    π = stationary_dist_via_eig(P)
    Pk = Matrix(I, size(P,1), size(P,2))
    Pk = P^k  # matrix power (not elementwise)

    errs = [norm(view(Pk, j, :) .- π, 1) for j in 1:size(P,1)]
    return π, Pk, errs
end

# ---- Example matrix in problem ----
P = [0.2 0.4 0.4;
     0.1 0.3 0.6;
     0.5 0.1 0.4]

π, P100, errs = compare_to_power(P; k=100)

println("Stationary distribution π (via eigenvector method):")
@printf("[%.6f, %.6f, %.6f]\n", π[1], π[2], π[3])

println("\nEach row of P^100 (should be ~ π):")
for j in 1:size(P,1)
    @printf("Row %d of P^100: [%.6f, %.6f, %.6f]   L1 err vs π = %.3e\n",
            j, P100[j,1], P100[j,2], P100[j,3], errs[j])
end




# Part 2: Numerical Integration

# Setup:

using Random, Distributions, QuadGK, FastGaussQuadrature, LinearAlgebra

Random.seed!(99)

# Common logistic helper
σlogit(x) = 1 / (1 + exp(-x))

# parameters
μ1D  = 0.5
σ1D  = 2.0
X1D  = 0.5
dist = Normal(μ1D, σ1D)


# 1. Create the function called binomiallogit

function binomiallogit(β; X = X1D, μ = μ1D, σ = σ1D, fpdf = (b->pdf(Normal(μ,σ), b)))
    return σlogit(β * X) * fpdf(β)
end

# 2. Integrate the function numerically, setting the tolerance to 1e-14.

true_1D, evals_1D = quadgk(β -> binomiallogit(β; X=X1D, μ=μ1D, σ=σ1D),
                           -Inf, Inf; rtol=1e-14, atol=1e-14, maxevals=10^7)

# 3. Integrate the function by taking 20 and 400 Monte Carlo draws from f and computing the sample mean.

function mc_estimate_1D(n; X=X1D, μ=μ1D, σ=σ1D)
    β = rand(Normal(μ,σ), n)
    return mean(σlogit.(β .* X))
end

Fmc_20  = mc_estimate_1D(20)
Fmc_400 = mc_estimate_1D(400)

# 4. Integrate the function using Gauss-Hermite quadrature for k=4, 12 (Try some odd ones too).
# Obtain the quadrature points and nodes from the internet.
# Gauss-Hermite quadrature assumes a weighting function of exp[-x^2], you will need a change of variables to integrate over a normal density.
# You also need to pay attention to the constant of integration.

# Gauss–Hermite quadrature expects ∫ e^{-x^2} g(x) dx.
# For β ~ N(μ,σ), we change variables so β = μ + σ√2 x,
# giving weights = w/√π and nodes = μ + σ√2 * x. 
# See latex answers.

function gh_estimate_1D(k; X=X1D, μ=μ1D, σ=σ1D)
    x, w = gausshermite(k)                 # weight exp(-x^2)
    βnodes = @. μ + σ * sqrt(2) * x
    weights = w ./ sqrt(pi)               # normalise to sum ≈ 1
    # Sanity check weight sum
    @assert sum(weights) ≈ 1
    return weights ⋅ σlogit.(βnodes .* X)
end

FGH_4   = gh_estimate_1D(4)
FGH_12  = gh_estimate_1D(12)

FGH_9   = gh_estimate_1D(9) #odd one

# 5. Compare results to the Monte Carlo results. Make sure your quadrature weights sum to 1!

println("True value: $true_1D")
println("MC 20: $Fmc_20")
println("MC 400: $Fmc_400")
println("GH 4: $FGH_4")
println("GH 12: $FGH_12")
println("GH 9: $FGH_9")

# note my function checks that weights sum to 1 and throws up an error if they don't

# 6. Repeat the exercise in two dimensions where μ = (0.5,1), σ = (2,1), and X=(0.5,1).

μ2  = (0.5, 1.0)
σ2  = (2.0, 1.0)
X2  = (0.5, 1.0)
dist2_1 = Normal(μ2[1], σ2[1])
dist2_2 = Normal(μ2[2], σ2[2])

# 2D integrand with density
function binomiallogit2D(β1, β2; X=X2, μ=μ2, σ=σ2)
    val = σlogit(β1 * X[1] + β2 * X[2])
    return val * pdf(Normal(μ[1],σ[1]), β1) * pdf(Normal(μ[2],σ[2]), β2)
end

# True 2D (nested integration)
true_2D = quadgk(β1 ->
    quadgk(β2 -> binomiallogit2D(β1, β2; X=X2, μ=μ2, σ=σ2),
           -Inf, Inf; rtol=1e-14, atol=1e-14, maxevals=10^7)[1],
    -Inf, Inf; rtol=1e-14, atol=1e-14, maxevals=10^7)[1]

# Monte Carlo 2D
function mc_estimate_2D(n; X=X2, μ=μ2, σ=σ2)
    β1 = rand(Normal(μ[1],σ[1]), n)
    β2 = rand(Normal(μ[2],σ[2]), n)
    return mean(σlogit.(β1 .* X[1] .+ β2 .* X[2]))
end

Fmc2_20  = mc_estimate_2D(20)
Fmc2_400 = mc_estimate_2D(400)

# Gauss–Hermite tensor grid 2D
# (x_i, w_i) per dim with exp(-x^2) weight; scale nodes and normalise weights per dim
# Effective 2D weights = (w1/√π) ⊗ (w2/√π) = (w1*w2)/π (i.e. tensor product)
function gh_estimate_2D(k1, k2; X=X2, μ=μ2, σ=σ2)
    x1, w1 = gausshermite(k1)
    x2, w2 = gausshermite(k2)
    β1_nodes = @. μ[1] + σ[1] * sqrt(2) * x1
    β2_nodes = @. μ[2] + σ[2] * sqrt(2) * x2
    weights1 = w1 ./ sqrt(pi)
    weights2 = w2 ./ sqrt(pi)
    # Tensor product rule evaluation
    # Build all pairwise sums for β1*X1 + β2*X2 and all pairwise weight products
    S = @. (β1_nodes .* X[1])     # length k1
    T = @. (β2_nodes .* X[2])     # length k2
    # Evaluate σlogit on the k1×k2 grid via broadcasting
    vals = σlogit.(S .+ T')       # (k1 × k2)
    W = weights1 * weights2'      # (k1 × k2), outer product, sums to ~1
    return sum(vals .* W)
end

FGH2_4x4  = gh_estimate_2D(4,4)
FGH2_12x12  = gh_estimate_2D(12,12)
FGH2_9x9  = gh_estimate_2D(9,9)



# 7. Put everything into two tables (one for the 1-D integral, one for the 2-D integral). 
# Showing the error from the ``true'' value and the number of points used in the evaluation.

fmt_err(x) = "\\num{" * @sprintf("%.3e", x) * "}"

function latex_tables_side_by_side(true1D, true2D)
    rows1 = [
        ("Monte Carlo",   "20",  fmt_err(abs(Fmc_20    - true1D))),
        ("Monte Carlo",   "400", fmt_err(abs(Fmc_400   - true1D))),
        ("Gauss--Hermite","4",   fmt_err(abs(FGH_4     - true1D))),
        ("Gauss--Hermite","9",   fmt_err(abs(FGH_9     - true1D))),
        ("Gauss--Hermite","12",  fmt_err(abs(FGH_12    - true1D))),
    ]
    rows2 = [
        ("Monte Carlo",             "20",  fmt_err(abs(Fmc2_20    - true2D))),
        ("Monte Carlo",             "400", fmt_err(abs(Fmc2_400   - true2D))),
        ("Gauss--Hermite (4×4)",    "16",  fmt_err(abs(FGH2_4x4   - true2D))),
        ("Gauss--Hermite (9×9)",    "81",  fmt_err(abs(FGH2_9x9   - true2D))),
        ("Gauss--Hermite (12×12)", "144",  fmt_err(abs(FGH2_12x12 - true2D))),
    ]

    cap1 = @sprintf("1-D Results\\\\(true value: %.6f)", true1D)
    cap2 = @sprintf("2-D Results\\\\(true value: %.6f)", true2D)

    io = IOBuffer()
    println(io, "\\begin{table}[H]")
    println(io, "\\centering")
    println(io, "\\begin{threeparttable}")
    println(io, "\\caption{Numerical Integration Results}")
    println(io, "\\begin{tabular}{@{}p{0.48\\linewidth} p{0.48\\linewidth}@{}}")

    # LEFT (1-D)
    println(io, "{\\centering \\textbf{", cap1, "}\\\\[0.5ex]")
    println(io, "\\begingroup\\setlength{\\tabcolsep}{6pt}")
    println(io, "\\begin{tabular}{l S[table-format=4.0] S}")
    println(io, "\\toprule")
    println(io, "Method & {Points} & {Error}\\\\")
    println(io, "\\midrule")
    for (m,p,e) in rows1
        println(io, m, " & ", p, " & ", e, " \\\\")
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\endgroup")
    println(io, "} &")

    # RIGHT (2-D)
    println(io, "{\\centering \\textbf{", cap2, "}\\\\[0.5ex]")
    println(io, "\\begingroup\\setlength{\\tabcolsep}{6pt}")
    println(io, "\\begin{tabular}{l S[table-format=4.0] S}")
    println(io, "\\toprule")
    println(io, "Method & {Points} & {Error}\\\\")
    println(io, "\\midrule")
    for (m,p,e) in rows2
        println(io, m, " & ", p, " & ", e, " \\\\")
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\endgroup")
    println(io, "} \\\\")
    println(io, "\\end{tabular}")

    println(io, "\\begin{tablenotes}")
    println(io, "\\footnotesize Note: For Monte Carlo, ``Points'' = random draws; for Gauss--Hermite, ``Points'' = quadrature nodes.")
    println(io, "\\end{tablenotes}")

    println(io, "\\end{threeparttable}")
    println(io, "\\end{table}")
    String(take!(io))
end

println(latex_tables_side_by_side(true_1D, true_2D))


# 8. Now Construct a new function called binomiallogitmixture that takes a vector for X and returns
#  a vector of binomial probabilities (appropriately integrated over f(beta_i | theta) for the 1-D 
# mixture). It should be obvious that Gauss-Hermite is the most efficient way to do this. Do NOT use loops.


# binomiallogitmixture: vectorised GH expectation Eβ[σ(β * X_j)] for β ~ N(μ,σ)
# Input:  X_vec :: AbstractVector  (returns one probability per X entry)
# Options: μ, σ (normal params); k (GH nodes)
function binomiallogitmixture(X_vec::AbstractVector; μ=μ1D, σ=σ1D, k::Int=12)
    x, w = gausshermite(k)                     # GH nodes/weights for ∫ e^{-x^2} g(x) dx
    βnodes = @. μ + σ * sqrt(2) * x            # map to Normal nodes
    weights = w ./ sqrt(pi)                    # effective GH weights under Normal; sum ≈ 1
    M = βnodes .* (X_vec')                     # k × m matrix of β_i * X_j
    P = 1 ./(1 .+ exp.(-M))                    # σlogit applied elementwise
    return (weights' * P)[:]                   # length-m vector of probabilities
end


# e.g.
X_test = [-2.0, -0.5, 0.0, 0.5, 2.0]
probs = binomiallogitmixture(X_test; μ=μ1D, σ=σ1D, k=12)

println("X values:   ", X_test)
println("Probabilities: ", probs)

# obviously, this scale pretty well 
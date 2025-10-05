################### Problem Set 1 for Chris Conlon's Empirical IO course ###################

# Deadline: Friday, September 19th, 2025

using CSV, DataFrames, Statistics, LinearAlgebra, Optim, Random, FastGaussQuadrature, Printf, Plots, ForwardDiff
using Base.Threads: @threads, nthreads   # light parallelism across households
using StatsFuns: logsumexp

cd("Grad-IO/Assignments/Assignment 1-MLE and GMM")
mkpath("outputs")

data = CSV.read("schools_dataset.csv", DataFrame)
sort!(data, [:household_id, :school_id])  # sorting for reshape to (J × H)

J = maximum(data.school_id) + 1 # number of schools;add 1 as starts at 0
H = maximum(data.household_id) + 1 # number of schools;add 1 as starts at 0

@assert nrow(data) == J * H # one observation per school-household
@assert all(x -> x in (0,1), data.y_ij) # choice indicator only has 0 and 1

# Reshape to matrices (J × H), household-major ordering
tests = reshape(data.test_scores, J, H)
sports = reshape(data.sports, J, H)
dist  = reshape(data.distance,J, H)
ymat  = reshape(data.y_ij,J, H)

# Assert exactly one choice per household
@assert all(sum(ymat[:, h]) == 1 for h in 1:H)

# Chosen index per household (assumes exactly one 1 per household)

# iterate through households and find the index of the chosen school (yields a vector of length H)
chosen_idx = vec(map(col -> findfirst(==(1), col), eachcol(ymat)))

# Q1. Plot distance distributions
xmin, xmax = extrema(dist) # to get the same x-axis scale for comparison

plt1 = histogram(dist[:], bins=30, xlabel="Distance", ylabel="Count",
                 title="Distance to Schools (All Options)",
                 xlims=(xmin, xmax))
savefig(plt1, "outputs/Q1_dist_all.png")

plt2 = histogram(dist[ymat .== 1], bins=30, xlabel="Distance", ylabel="Count",
                 title="Distance to Chosen School",
                 xlims=(xmin, xmax))
savefig(plt2, "outputs/Q1_dist_chosen.png")

# people choose closer schools!


# Q2. Write down the market share and log-likelihood for a plain logit model.

# See Latex


# Q3. Write down the score and the gradient of your log-likelihood.

# see Latex


# Q4. Estimate the plain logit model by maximimum likelihood.

# start with some helper functions
function unpack_plain(θ::AbstractVector, J::Int) #unpacks parameters
    β1 = θ[1]
    β2 = θ[2]
    α  = θ[3]              # utility uses −α* distance (α > 0 expected)
    ξ  = zeros(J) # zero vector now ignored in this setup as collinear
    return β1, β2, α, ξ
end

# deterministic utility V (J × H): V_{j,h} = β1*tests + β2*sports − α*dist + ξ_j
function V_plain(β1::Float64, β2::Float64, α::Float64, ξ::Vector{Float64})
    # tests, sports: length-J (or J×1); dist: J×H; ξ: length-J
    # broadcasting adds the J-vectors across columns of the J×H matrix
    return @. β1*tests + β2*sports - α*dist
end

# per-household log-likelihood contribution using the overflow trick from last pset
@inline function ll_contrib_stable(Vcol::AbstractVector{<:Real}, jstar::Int)
    m = maximum(Vcol)
    return (Vcol[jstar] - m) - log(sum(exp.(Vcol .- m)))
end

# Plain logit log-likelihood
function ll_plain(θ::AbstractVector)
    β1, β2, α, ξ = unpack_plain(θ, J)
    V = V_plain(β1, β2, α, ξ)
    s = 0.0
    @inbounds for h in 1:H
        s += ll_contrib_stable(view(V, :, h), chosen_idx[h])
    end
    return s
end

# Gradient for plain logit (using overflow trick)
function grad_plain!(g::AbstractVector, θ::AbstractVector)
    β1, β2, α, ξ = unpack_plain(θ, J)
    V = V_plain(β1, β2, α, ξ)
    Vcmax = maximum(V, dims=1)
    P = exp.(V .- Vcmax)                # stabilize
    P ./= sum(P, dims=1)

    Ey_tests  = sum(tests  .* ymat);  E_tests  = sum(tests  .* P)
    Ey_sports = sum(sports .* ymat);  E_sports = sum(sports .* P)
    Ey_dist   = sum(dist   .* ymat);  E_dist   = sum(dist   .* P)

    g[1] = Ey_tests  - E_tests
    g[2] = Ey_sports - E_sports
    g[3] = -(Ey_dist - E_dist)          # utility uses −α*dist

    return g
end


# Diversion ratio from school 1 to school 2
function diversion_1to2(β1::Float64, β2::Float64, α::Float64, ξ::Vector{Float64})
    V = V_plain(β1, β2, α, ξ)
    Vcmax = maximum(V, dims=1)
    P = exp.(V .- Vcmax)
    P ./= sum(P, dims=1)
    P1 = vec(P[1, :]); P2 = vec(P[2, :])
    vals = P2 ./ clamp.(1 .- P1, 1e-12, Inf)
    return mean(vals)
end

# Average own elasticity of chosen school wrt distance
function avg_own_dist_elasticity(β1::Float64, β2::Float64, α::Float64, ξ::Vector{Float64})
    V = V_plain(β1, β2, α, ξ)
    Vcmax = maximum(V, dims=1)
    P = exp.(V .- Vcmax)
    P ./= sum(P, dims=1)
    els = Float64[]
    @inbounds for h in 1:H
        j = chosen_idx[h]
        pj = P[j, h]
        deriv = -α * pj * (1 - pj)
        push!(els, deriv * (dist[j, h] / max(pj, 1e-12)))
    end
    return mean(els)
end


θ0 = vcat([0.0, 0.0, 0.1])  # [β1, β2, α]
od = OnceDifferentiable(θ -> -ll_plain(θ),
                        (g, θ) -> (grad_plain!(g, θ); g .= -g),
                        θ0; inplace=true)
res_plain = optimize(od, θ0, LBFGS(), Optim.Options(; iterations=10_000))

θ̂_plain = Optim.minimizer(res_plain)
β1̂, β2̂, α̂, ξ̂ = unpack_plain(θ̂_plain, J)
nll_plain = -ll_plain(θ̂_plain)
D12_plain = diversion_1to2(β1̂, β2̂, α̂, ξ̂)
elas_plain = avg_own_dist_elasticity(β1̂, β2̂, α̂, ξ̂)

## quick architecture for printing extra sets of results to table

rf(x) = @sprintf("%.2f", x) #rounds

# container for rows
const results = []

# push a new row
function add_result!(model; nll, β1=NaN, β2=NaN, α=NaN, D12=NaN, elas=NaN)
    push!(results, (model=model, nll=nll, β1=β1, β2=β2, α=α, D12=D12, elas=elas))
end

# format helper: number if defined, "--" if NaN
function fmt(x)
    isnan(x) ? "--" : rf(x)
end

function latex_table(results)
    rows = String[]
    for r in results
        push!(rows,
              @sprintf("%s & %s & %s & %s & %s & %s & %s \\\\",
                       r.model,
                       fmt(r.nll),
                       fmt(r.β1),
                       fmt(r.β2),
                       fmt(r.α),
                       fmt(r.D12),
                       fmt(r.elas)))
    end
    return """
\\begin{table}[H]
\\centering
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{lcccccc}
\\toprule
Model & NLL & \$\\beta_1\$ & \$\\beta_2\$ & \$\\alpha\$ & \$D_{12}\$ & Avg. own dist. elas. \\\\
\\midrule
$(join(rows, "\\n"))
\\bottomrule
\\end{tabular}
}
\\caption{Estimation results for logit models.}
\\end{table}
"""
end

# q4 results
add_result!("Plain Logit", nll=nll_plain, β1=β1̂, β2=β2̂, α=α̂, D12=D12_plain, elas=elas_plain)
println(latex_table(results))

# Q5. Estimate a restricted model with only $\xi_j$ parameters. Add that to your table.
function ll_only_xi(ϕ::AbstractVector)
    ξ = zeros(J)
    ξ[1:J-1] .= ϕ
    ξ[J] = -sum(ξ[1:J-1])
    V = ξ .+ zeros(J, H) # replicate ξ across columns
    s = 0.0
    @inbounds for h in 1:H
        s += ll_contrib_stable(V[:, h], chosen_idx[h])
    end
    return s
end

function grad_only_xi!(g::AbstractVector, ϕ::AbstractVector)
    ξ = zeros(J)
    ξ[1:J-1] .= ϕ
    ξ[J] = -sum(ξ[1:J-1])
    V = ξ .+ zeros(J, H)
    Vcmax = maximum(V, dims=1)
    P = exp.(V .- Vcmax)
    P ./= sum(P, dims=1)
    @inbounds for j in 1:J-1
        g[j] = sum(ymat[j, :]) - sum(P[j, :])
    end
    return g
end

ϕ0 = zeros(J-1)
od_xi = OnceDifferentiable(ϕ -> -ll_only_xi(ϕ),
                           (g, ϕ) -> (grad_only_xi!(g, ϕ); g .= -g),
                           ϕ0; inplace=true)
res_xi = optimize(od_xi, ϕ0, LBFGS(), Optim.Options(; iterations=10_000))
ϕ̂     = Optim.minimizer(res_xi)
nll_xi = -ll_only_xi(ϕ̂)

add_result!(raw"$\text{Only-}\xi$", nll=nll_xi)

println(latex_table(results))



# Q6. Now allow for parents to have different preferences for \text{test scores}_j so that \beta_{1i} \sim \mathcal{N}(\beta_1, \sigma_b). Write down the (simulated) market share and gradient expressions.

# see Latex


# Q7. Estimate this expanded model via maximum likelihood: 
# (a) Using 100 Monte Carlo Draws from an appropriately transformed standard normal. 
# (b) Using a Gauss Hermite quadrature rule.

### (a) Monte Carlo ###


#fixed base draws
const R = 100
Random.seed!(1999)
const Z = randn(H, R)

# Reusable buffers (shape fixed by J,H)
const VBUF = zeros(Float64, J, H)
const PBUF = similar(VBUF)
const ST   = zeros(Float64, H)  # s_tests (expected tests by household)
const SS   = zeros(Float64, H)  # s_sports (expected sports by household)
const SD   = zeros(Float64, H)  # s_dist (expected distance by household)
const CD   = zeros(Float64, H)  # chosen_dist (household-specific chosen distance)

# Extra reusable accumulators (avoid per-call allocations)
const PCBUF = zeros(Float64, H) # p_i accumulator: sum over draws/nodes of P_{j(i)}
const S1BUF = zeros(Float64, H) # gradient accumulator for β1
const S2BUF = zeros(Float64, H) # gradient accumulator for β2
const SABUF = zeros(Float64, H) # gradient accumulator for α
const SσBUF = zeros(Float64, H) # gradient accumulator for ℓσ


# Collapse to J×1 vectors since tests/sports do not vary across households
tests_j  = ndims(tests)  == 1 ? tests  : vec(tests[:, 1])
sports_j = ndims(sports) == 1 ? sports : vec(sports[:, 1])
@assert length(tests_j) == J && length(sports_j) == J

# Unpack parameters: θ = [β1, β2, α, ℓσ]
# Enforce σ_b > 0 via σ_b = exp(ℓσ), and impose sum-to-zero constraint on ξ
function unpack_mixed(θ::AbstractVector, J::Int)
    T  = eltype(θ)
    β1 = θ[1]
    β2 = θ[2]
    α  = θ[3]
    σb = exp(θ[4])
    ξ  = zeros(T, J)
    return β1, β2, α, σb, ξ
end

# Column-wise softmax with stabilisation
function softmax_cols!(P::AbstractMatrix, V::AbstractMatrix)
    @inbounds for i in 1:size(V,2)
        m = -Inf
        @inbounds for j in 1:size(V,1)
            v = V[j,i]; if v > m; m = v; end
        end
        s = 0.0
        @inbounds for j in 1:size(V,1)
            x = exp(V[j,i] - m)
            P[j,i] = x
            s += x
        end
        invs = 1.0 / s
        @inbounds for j in 1:size(V,1)
            P[j,i] *= invs
        end
    end
    return P
end

function softmax_cols(V::AbstractMatrix)
    P = similar(V)
    return softmax_cols!(P, V)
end

# Simulated log-likelihood with R Monte Carlo draws
function ll_mixed_mc(θ::AbstractVector)
    T = eltype(θ)
    β1, β2, α, σb, ξ = unpack_mixed(θ, J)
    pchosen = zeros(T, H)
    base = β2 .* sports_j .+ ξ
    for r in 1:R
        β1i = β1 .+ σb .* Z[:, r] # household-specific β1 draws
        V = tests_j * permutedims(β1i) # J×H utility from test scores
        V .+= base # add sports and ξ
        V .-= α .* dist # subtract distance disutility
        P = softmax_cols(V) # choice probabilities
        for i in 1:H
            pchosen[i] += P[chosen_idx[i], i] # accumulate prob of chosen option
        end
    end
    return sum(log.(max.(pchosen ./ T(R), eps(T))))          # average over draws
end

ll_mixed_mc_neg(θ) = -ll_mixed_mc(θ) #as minimises

# MC gradient (preallocated, BLAS for expectations, no per-i allocations)
function grad_mixed_mc!(g::AbstractVector, θ::AbstractVector)
    β1, β2, α, σb, _ = unpack_mixed(θ, J)
    g .= 0.0

    # gather chosen distances once
    @inbounds for i in 1:H
        CD[i] = dist[chosen_idx[i], i]
    end

    # per-household accumulators
    # reuse global buffers to reduce allocations
    PC = PCBUF; fill!(PC, 0.0)
    Sβ1 = S1BUF; fill!(Sβ1, 0.0)
    Sβ2 = S2BUF; fill!(Sβ2, 0.0)
    Sα  = SABUF; fill!(Sα, 0.0)
    Sσ  = SσBUF; fill!(Sσ, 0.0)

    for r in 1:R
        # V = β1i*tests + β2*sports - α*dist
        @inbounds @simd for i in 1:H
            β1i = β1 + σb * Z[i, r]
            for j in 1:J
                VBUF[j,i] = β1i * tests_j[j] + β2 * sports_j[j] - α * dist[j,i]
            end
        end

        softmax_cols!(PBUF, VBUF)

        # Expectations across alternatives for each i
        mul!(ST, transpose(PBUF), tests_j)      # ST[i] = ∑_k P_{k,i} * tests_k
        mul!(SS, transpose(PBUF), sports_j)     # SS[i] = ∑_k P_{k,i} * sports_k

        # SD[i] = ∑_k P_{k,i} * d_{k,i}
        Threads.@threads for i in 1:H        # parallel across households
            s = 0.0
            for j in 1:J
                s += PBUF[j,i] * dist[j,i]
            end
            SD[i] = s
        end

        # Accumulate derivatives of the averaged probability p_i
        Threads.@threads for i in 1:H        # parallel across households
            ji = chosen_idx[i]
            pc = PBUF[ji, i]                     # P_{j(i)}^{(r)}
            PC[i] += pc
            dtests   = tests_j[ji]  - ST[i]
            dsports  = sports_j[ji] - SS[i]
            ddist    = -(CD[i] - SD[i])
            dσterm   = (σb * Z[i, r]) * dtests

            Sβ1[i] += pc * dtests
            Sβ2[i] += pc * dsports
            Sα[i]  += pc * ddist
            Sσ[i]  += pc * dσterm
        end
    end

    # Chain rule for log of the average: ∂ℓ/∂θ = ∑_i (1 / p_i) * ∂p_i/∂θ, with p_i = PC[i]/R
    @inbounds for i in 1:H
        denom = max(PC[i], 1e-300)       # avoid divide-by-zero; (R cancels)
        g[1] += Sβ1[i] / denom
        g[2] += Sβ2[i] / denom
        g[3] += Sα[i]  / denom
        g[4] += Sσ[i]  / denom
    end

    g .*= -1.0   # Optim minimizes the negative log-likelihood
    return g
end

# Fused objective+gradient for MC MLE (avoids recomputing probabilities twice)
function fg_mixed_mc!(g::AbstractVector, θ::AbstractVector)
    β1, β2, α, σb, _ = unpack_mixed(θ, J)
    g .= 0.0

    # chosen distances
    @inbounds for i in 1:H
        CD[i] = dist[chosen_idx[i], i]
    end

    # zero accumulators
    PC = PCBUF; fill!(PC, 0.0)
    Sβ1 = S1BUF; fill!(Sβ1, 0.0)
    Sβ2 = S2BUF; fill!(Sβ2, 0.0)
    Sα  = SABUF; fill!(Sα, 0.0)
    Sσ  = SσBUF; fill!(Sσ, 0.0)

    # accumulate over Monte Carlo draws (households in parallel where safe)
    for r in 1:R
        @inbounds @simd for i in 1:H
            β1i = β1 + σb * Z[i, r]
            for j in 1:J
                VBUF[j,i] = β1i * tests_j[j] + β2 * sports_j[j] - α * dist[j,i]
            end
        end
        softmax_cols!(PBUF, VBUF)
        mul!(ST, transpose(PBUF), tests_j)
        mul!(SS, transpose(PBUF), sports_j)

        Threads.@threads for i in 1:H
            s = 0.0
            for j in 1:J
                s += PBUF[j,i] * dist[j,i]
            end
            SD[i] = s
        end

        Threads.@threads for i in 1:H
            ji = chosen_idx[i]
            pc = PBUF[ji, i]
            PC[i] += pc
            dtests   = tests_j[ji]  - ST[i]
            dsports  = sports_j[ji] - SS[i]
            ddist    = -(CD[i] - SD[i])
            dσterm   = (σb * Z[i, r]) * dtests
            Sβ1[i] += pc * dtests
            Sβ2[i] += pc * dsports
            Sα[i]  += pc * ddist
            Sσ[i]  += pc * dσterm
        end
    end

    # gradient
    ll = 0.0
    @inbounds for i in 1:H
        denom = max(PC[i], 1e-300)
        g[1] += Sβ1[i] / denom
        g[2] += Sβ2[i] / denom
        g[3] += Sα[i]  / denom
        g[4] += Sσ[i]  / denom
        ll += log(max(PC[i] / R, 1e-300))
    end
    g .*= -1.0
    return -ll
end


θ0_mixed = vcat([0.0, 0.0, 0.1, log(2)])  # [β1, β2, α, ℓσ]


od_mc = OnceDifferentiable(ll_mixed_mc_neg, grad_mixed_mc!, fg_mixed_mc!, θ0_mixed; inplace=true)  # fused value+grad speeds up
res_mc = optimize(od_mc, θ0_mixed, LBFGS(), Optim.Options(; iterations=10_000))

θ̂_mc    = Optim.minimizer(res_mc)
β1̂_mc, β2̂_mc, α̂_mc, σ̂b_mc, ξ̂_mc = unpack_mixed(θ̂_mc, J)
nll_mc   = -ll_mixed_mc(θ̂_mc)

# Diversion ratio from school 1 to school 2 under mixed logit (MC average over draws and households)
function diversion_1to2_mc(β1, β2, α, σb, ξ)
    base = β2 .* sports_j .+ ξ                     # terms common across households
    acc = 0.0
    count = 0
    @inbounds for r in 1:R
        β1i = β1 .+ σb .* Z[:, r]                  # household-specific β1 draws
        V = tests_j * permutedims(β1i)             # J×H utility from test scores
        V .+= base                                  # add sports and ξ
        V .-= α .* dist                             # subtract distance disutility
        P = softmax_cols(V)                         # choice probabilities
        P1 = @view P[1, :]
        P2 = @view P[2, :]
        vals = P2 ./ clamp.(1 .- P1, 1e-12, Inf)    # D_{12} = P2 / (1 - P1)
        acc += sum(vals)
        count += length(vals)
    end
    return acc / count
end

# Average own-distance elasticity of the chosen school under mixed logit (MC average)
function avg_own_dist_elasticity_mc(β1, β2, α, σb, ξ)
    base = β2 .* sports_j .+ ξ
    acc = 0.0
    count = 0
    @inbounds for r in 1:R
        β1i = β1 .+ σb .* Z[:, r]
        V = tests_j * permutedims(β1i)
        V .+= base
        V .-= α .* dist
        P = softmax_cols(V)
        for i in 1:H
            j = chosen_idx[i]
            pij = P[j, i]
            deriv = -α * pij * (1 - pij)           # ∂P_{ij}/∂d_{ij}
            acc += deriv * (dist[j, i] / max(pij, 1e-12))
            count += 1
        end
    end
    return acc / count
end

D12_mc  = diversion_1to2_mc(β1̂_mc, β2̂_mc, α̂_mc, σ̂b_mc, ξ̂_mc)
elas_mc = avg_own_dist_elasticity_mc(β1̂_mc, β2̂_mc, α̂_mc, σ̂b_mc, ξ̂_mc)

add_result!(raw"Expanded Logit (MC, R=100)",
    nll=nll_mc, β1=β1̂_mc, β2=β2̂_mc, α=α̂_mc, D12=D12_mc, elas=elas_mc)


### (b) Gauss Hermite ###

# Gauss–Hermite nodes/weights for ∫ e^{-x^2} g(x) dx as in last pset
function gh_nodes_weights(M::Int)
    a = zeros(M)
    b = sqrt.(collect(1:M-1) ./ 2)
    T = SymTridiagonal(a, b)
    vals, vecs = eigen(T)
    x = vals
    w = vecs[1, :].^2 .* sqrt(pi)
    return x, w
end

# Transform to standard-normal expectation: E[f(Z)] with Z~N(0,1)
const M_GH = 20
const x_GH, w_GH = gh_nodes_weights(M_GH)
const z_GH  = x_GH ./ sqrt(2)
const wn_GH = w_GH ./ sqrt(pi)


# Simulated log-likelihood using Gauss–Hermite quadrature
function ll_mixed_gh(θ::AbstractVector)
    T = eltype(θ)
    β1, β2, α, σb, _ = unpack_mixed(θ, J)  # ignore ξ
    pchosen = zeros(T, H)

    @inbounds for m in 1:M_GH
        β1m = β1 + σb * z_GH[m]                  # scalar at node m
        # V[j,i] = β1m*tests_j[j] + β2*sports_j[j] - α*dist[j,i]
        V = @. β1m*tests_j + β2*sports_j - α*dist
        P = softmax_cols(V)                      # J×H probabilities
        w = wn_GH[m]
        @inbounds for i in 1:H
            pchosen[i] += w * P[chosen_idx[i], i]
        end
    end
    return sum(log.(max.(pchosen, eps(T))))
end

ll_mixed_gh_neg(θ) = -ll_mixed_gh(θ)

function grad_mixed_gh!(g::AbstractVector, θ::AbstractVector)
    β1, β2, α, σb, _ = unpack_mixed(θ, J)  # ignore ξ
    g .= 0.0

    # per-household accumulators
    PC = zeros(Float64, H)   # p_i = ∑_m w_m * P_{j(i)}(node m)
    Sβ1 = zeros(Float64, H)  # ∑_m w_m * P_{j(i)} * (tests_{j(i)} - E_P[tests])
    Sβ2 = zeros(Float64, H)  # ∑_m w_m * P_{j(i)} * (sports_{j(i)} - E_P[sports])
    Sα  = zeros(Float64, H)  # ∑_m w_m * P_{j(i)} * (-(d_{j(i),i} - E_P[d]))
    Sσ  = zeros(Float64, H)  # ∑_m w_m * P_{j(i)} * (z_m * (tests_{j(i)} - E_P[tests]))

    s_tests  = zeros(Float64, H)   # E_P[tests]_i at each node
    s_sports = zeros(Float64, H)   # E_P[sports]_i at each node
    s_dist   = zeros(Float64, H)   # E_P[distance]_i at each node

    @inbounds for m in 1:M_GH
        w  = wn_GH[m]
        z  = z_GH[m]
        β1m = β1 + σb * z

        # V[j,i] = β1m*tests_j[j] + β2*sports_j[j] - α*dist[j,i]
        V = @. β1m*tests_j + β2*sports_j - α*dist
        P = softmax_cols(V)   # J×H

        # E_P[tests]_i and E_P[sports]_i: (H×J)*(J) -> (H)
        s_tests  .= vec(transpose(P) * tests_j)
        s_sports .= vec(transpose(P) * sports_j)

        # E_P[distance]_i
        @inbounds for i in 1:H
            s = 0.0
            @inbounds for j in 1:J
                s += P[j,i] * dist[j,i]
            end
            s_dist[i] = s
        end

        # accumulate ∂p_i/∂θ (note we multiply by P_{j(i)} here)
        @inbounds for i in 1:H
            ji = chosen_idx[i]
            pc = P[ji, i]                       # P_{j(i)} at node m
            PC[i] += w * pc
            dtests   = tests_j[ji]  - s_tests[i]
            dsports  = sports_j[ji] - s_sports[i]
            ddist    = -(dist[ji, i] - s_dist[i])
            dσterm   = (σb * z) * dtests

            Sβ1[i] += w * pc * dtests
            Sβ2[i] += w * pc * dsports
            Sα[i]  += w * pc * ddist
            Sσ[i]  += w * pc * dσterm
        end
    end

    # Chain rule for log of simulated prob: ∂ℓ/∂θ = ∑_i (1 / p_i) * ∂p_i/∂θ
    @inbounds for i in 1:H
        wi = 1.0 / max(PC[i], 1e-300)
        g[1] += wi * Sβ1[i]
        g[2] += wi * Sβ2[i]
        g[3] += wi * Sα[i]
        g[4] += wi * Sσ[i]
    end

    g .*= -1.0   # Optim minimises the negative log-likelihood
    return g
end

# Fused objective+gradient for GH MLE (compute value and gradient together)
function fg_mixed_gh!(g::AbstractVector, θ::AbstractVector)
    β1, β2, α, σb, _ = unpack_mixed(θ, J)
    g .= 0.0

    PC = PCBUF; fill!(PC, 0.0)
    Sβ1 = S1BUF; fill!(Sβ1, 0.0)
    Sβ2 = S2BUF; fill!(Sβ2, 0.0)
    Sα  = SABUF; fill!(Sα, 0.0)
    Sσ  = SσBUF; fill!(Sσ, 0.0)

    @inbounds for m in 1:M_GH
        w  = wn_GH[m]
        z  = z_GH[m]
        β1m = β1 + σb * z
        V = @. β1m*tests_j + β2*sports_j - α*dist
        P = softmax_cols(V)
        ST .= vec(transpose(P) * tests_j)
        SS .= vec(transpose(P) * sports_j)

        Threads.@threads for i in 1:H
            s = 0.0
            for j in 1:J
                s += P[j,i] * dist[j,i]
            end
            SD[i] = s
        end

        Threads.@threads for i in 1:H
            ji = chosen_idx[i]
            pc = P[ji, i]
            PC[i] += w * pc
            dtests   = tests_j[ji]  - ST[i]
            dsports  = sports_j[ji] - SS[i]
            ddist    = -(dist[ji, i] - SD[i])
            dσterm   = (σb * z) * dtests
            Sβ1[i] += w * pc * dtests
            Sβ2[i] += w * pc * dsports
            Sα[i]  += w * pc * ddist
            Sσ[i]  += w * pc * dσterm
        end
    end

    ll = 0.0
    @inbounds for i in 1:H
        wi = 1.0 / max(PC[i], 1e-300)
        g[1] += wi * Sβ1[i]
        g[2] += wi * Sβ2[i]
        g[3] += wi * Sα[i]
        g[4] += wi * Sσ[i]
        ll += log(max(PC[i], 1e-300))
    end
    g .*= -1.0
    return -ll
end

# Optimise
θ0_gh = vcat([0.0, 0.0, 0.1, log(0.1)])
od_gh = OnceDifferentiable(ll_mixed_gh_neg, grad_mixed_gh!, fg_mixed_gh!, θ0_gh; inplace=true)   # fused value+grad speeds up
res_gh = optimize(od_gh, θ0_gh, LBFGS(), Optim.Options(; iterations=10_000))

θ̂_gh    = Optim.minimizer(res_gh)
β1̂_gh, β2̂_gh, α̂_gh, σ̂b_gh, ξ̂_gh = unpack_mixed(θ̂_gh, J)
nll_gh   = -ll_mixed_gh(θ̂_gh)

# Diversion ratio from school 1 to 2 (GH)
function diversion_1to2_gh(β1, β2, α, σb)
    acc = 0.0; cnt = 0
    @inbounds for m in 1:M_GH
        β1m = β1 + σb * z_GH[m]
        V = @. β1m*tests_j + β2*sports_j - α*dist
        P = softmax_cols(V)
        vals = wn_GH[m] .* (P[2, :] ./ clamp.(1 .- P[1, :], 1e-12, Inf))
        acc += sum(vals); cnt += length(vals)
    end
    return acc / H
end

# Average own-distance elasticity of the chosen school (GH)
function avg_own_dist_elasticity_gh(β1, β2, α, σb)
    acc = 0.0; cnt = 0
    @inbounds for m in 1:M_GH
        β1m = β1 + σb * z_GH[m]
        V = @. β1m*tests_j + β2*sports_j - α*dist
        P = softmax_cols(V)
        w = wn_GH[m]
        @inbounds for i in 1:H
            j = chosen_idx[i]
            pij = P[j, i]
            deriv = -α * pij * (1 - pij)                    # ∂P_{ij}/∂d_{ij}
            acc += w * deriv * (dist[j, i] / max(pij, 1e-12))
            cnt += 1
        end
    end
    return acc / H
end

D12_gh  = diversion_1to2_gh(β1̂_gh, β2̂_gh, α̂_gh, σ̂b_gh)
elas_gh = avg_own_dist_elasticity_gh(β1̂_gh, β2̂_gh, α̂_gh, σ̂b_gh)

add_result!(raw"Mixed Logit (GH, M=20)",
    nll=nll_gh, β1=β1̂_gh, β2=β2̂_gh, α=α̂_gh, D12=D12_gh, elas=elas_gh)


println(latex_table(results))




# Q8. Read Chapter 10 in Train and write down the MSM estimator for the expanded model. What are your ``instruments''?

# See Latex


# Q9. Calculate the Jacobian of the MSM estimator.

# See Latex


# Q10. Estimate the Parameters of the MSM estimator.

function msm_moments_and_jacobian(θ::AbstractVector)
    # Moments g(θ) and analytic Jacobian G(θ) under Gauss–Hermite averaging
    T = eltype(θ)
    β1, β2, α, σb, _ = unpack_mixed(θ, J)

    # Instruments z_{ij} = (tests_j, sports_j, d_{ij})
    Z_tests  = repeat(reshape(tests_j,  J, 1), 1, H)   # J×H
    Z_sports = repeat(reshape(sports_j, J, 1), 1, H)   # J×H
    Z_dist   = dist                                    # J×H

    P̂ = zeros(T, J, H)
    p = 4
    G = zeros(T, 3, p)

    V  = zeros(T, J, H)
    Pm = similar(V)
    oneH = T(1) / T(H)

    # Add one Jacobian column given X = ∂V/∂θ (J×H) and node weight w
    function accumulate_param!(col::Int, X, w)
        s = sum(Pm .* X, dims=1)        # Σ_k P_ik x_ik,θ
        M = Pm .* (X .- s)              # ∂P^{(node)}/∂θ at this node
        G[1, col] += -(w * oneH) * sum(Z_tests  .* M)
        G[2, col] += -(w * oneH) * sum(Z_sports .* M)
        G[3, col] += -(w * oneH) * sum(Z_dist   .* M)
        nothing
    end

    @inbounds for m in 1:M_GH
        w = wn_GH[m]
        β1m = β1 + σb * z_GH[m]
        @. V = (β1m * tests_j + β2 * sports_j)
        V .-= α .* dist
        Pm .= softmax_cols(V)
        P̂ .+= w .* Pm

        Xβ1 = repeat(reshape(tests_j,  J, 1), 1, H)
        Xβ2 = repeat(reshape(sports_j, J, 1), 1, H)
        Xα  = .-dist
        Xσ  = Xβ1 .* (σb * z_GH[m])     # ∂V/∂ℓσ = (∂V/∂σ_b)·σ_b

        accumulate_param!(1, Xβ1, w)    # β1
        accumulate_param!(2, Xβ2, w)    # β2
        accumulate_param!(3, Xα,  w)    # α
        accumulate_param!(4, Xσ,  w)    # ℓσ
    end

    resid = ymat .- P̂
    g1 = oneH * sum(resid .* Z_tests)
    g2 = oneH * sum(resid .* Z_sports)
    g3 = oneH * sum(resid .* Z_dist)
    g = T[g1, g2, g3]

    return g, G
end


# MSM criterion Q(θ) = g(θ)' W g(θ); here W = I (one-step MSM)
function msm_objective_and_gradient(θ::AbstractVector)
    g, G = msm_moments_and_jacobian(θ)
    Q = dot(g, g)
    grad = 2 .* (G' * g)
    return Q, grad
end

msm_value(θ) = msm_objective_and_gradient(θ)[1]
function msm_gradient!(gvec, θ)
    _, grad = msm_objective_and_gradient(θ)
    gvec .= grad
    gvec
end

θ0_msm = vcat([0.0, 0.0, 0.1, log(2)])
function msm_fg!(gvec, θ)
    Q, grad = msm_objective_and_gradient(θ)    # compute both once
    gvec .= grad
    return Q
end
od_msm = OnceDifferentiable(msm_value, msm_gradient!, msm_fg!, θ0_msm; inplace=true)  # fused value+grad
res_msm = optimize(od_msm, θ0_msm, LBFGS(), Optim.Options(; iterations=10_000))

θ̂_msm = Optim.minimizer(res_msm)
β1̂_msm, β2̂_msm, α̂_msm, σ̂b_msm, _ = unpack_mixed(θ̂_msm, J)

D12_msm  = diversion_1to2_gh(β1̂_msm, β2̂_msm, α̂_msm, σ̂b_msm)
elas_msm = avg_own_dist_elasticity_gh(β1̂_msm, β2̂_msm, α̂_msm, σ̂b_msm)

add_result!(raw"MSM (GH, M=20)",
    nll=NaN, β1=β1̂_msm, β2=β2̂_msm, α=α̂_msm, D12=D12_msm, elas=elas_msm)

println(latex_table(results))


# Q11. Bonus: Using your initial MSM estimates as a starting point, explain how to construct an ``efficient'' MSM estimator, and produce ``efficient'' estimates.


# GH-averaged probabilities P̂_{ij}(θ) (J×H)
function probs_gh(β1, β2, α, σb)
    T = eltype(dist)
    P̂ = zeros(T, J, H)
    base = β2 .* sports_j
    @inbounds for m in 1:M_GH
        β1m = β1 + σb * z_GH[m]
        V = (tests_j .* β1m) .+ base .- α .* dist     # J×H
        P = softmax_cols(V)                           # J×H
        P̂ .+= wn_GH[m] .* P
    end
    P̂
end


# Per-household moment vectors m_i(θ) (H×K) and their average g(θ) (K)
# Instruments are (tests_j, sports_j, d_{ij}); K=3
function household_moments(θ::AbstractVector)
    T = eltype(θ)
    β1, β2, α, σb, _ = unpack_mixed(θ, J)
    P̂ = probs_gh(β1, β2, α, σb)
    resid = ymat .- P̂                                # J×H
    M = zeros(T, H, 3)
    @inbounds for i in 1:H
        M[i, 1] = dot(resid[:, i], tests_j)          # tests_j instrument
        M[i, 2] = dot(resid[:, i], sports_j)         # sports_j instrument
        M[i, 3] = dot(resid[:, i], dist[:, i])       # distance instrument
    end
    g = vec(sum(M, dims=1)) ./ T(H)                   # length-3
    M, g
end



# Build optimal weight from initial MSM estimate
function optimal_weight(θ::AbstractVector; ridge::Float64 = 1e-10)
    M, g = household_moments(θ) # H×3 and length-3
    Hloc = size(M, 1)
    U = M .- reshape(g, 1, :)
    S = (U' * U) / Hloc     # covariance of moments
    W = inv(S + ridge * I)  # small ridge for stability
    W
end

# One-step objective/gradient with generic weight W (3×3)
function msm_Q_grad(θ::AbstractVector, W::AbstractMatrix)
    g, G = msm_moments_and_jacobian(θ) # g: 3×1, G: 3×p
    Q = g' * W * g
    grad = 2 .* (G' * (W * g))
    Q[], grad
end

# Efficient MSM: re-optimise starting from θ̂_msm with W = Ŝ^{-1}
function efficient_msm(θ_start::AbstractVector)
    W = optimal_weight(θ_start)
    msm_val(θ) = msm_Q_grad(θ, W)[1]
    function msm_grad!(gvec, θ)
        _, grad = msm_Q_grad(θ, W)
        gvec .= grad
        gvec
    end
    od = OnceDifferentiable(msm_val, msm_grad!, θ_start; inplace=true)
    res = optimize(od, θ_start, LBFGS(), Optim.Options(; iterations=10_000))
    θ̂ = Optim.minimizer(res)
    θ̂, W, res
end

# Run two-step (efficient) MSM from initial one-step MSM estimate θ̂_msm
θ̂_eff, Ŵ, res_eff = efficient_msm(θ̂_msm)

β1̂_eff, β2̂_eff, α̂_eff, σ̂b_eff, _ = unpack_mixed(θ̂_eff, J)

D12_eff  = diversion_1to2_gh(β1̂_eff, β2̂_eff, α̂_eff, σ̂b_eff)
elas_eff = avg_own_dist_elasticity_gh(β1̂_eff, β2̂_eff, α̂_eff, σ̂b_eff)


add_result!(raw"Efficient MSM (GH, M=20))",
    nll=NaN, β1=β1̂_eff, β2=β2̂_eff, α=α̂_eff, D12=D12_eff, elas=elas_eff)

println(latex_table(results))
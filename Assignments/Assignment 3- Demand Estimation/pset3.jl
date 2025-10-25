# Pull in the helper that wires Julia to PyBLP via PyCall
include(joinpath(@__DIR__, "ensure_pyblp.jl"))

using Random, Distributions, DataFrames, LinearAlgebra, Plots, NLsolve, Printf

Random.seed!(1999)

# 1. Generate the data

T = 600 # number of markets
J = 4 # four inside goods per market
N = T * J # total product-market observations

market_id  = repeat(1:T, inner=J)
product_id = repeat(1:J,  T)

# Technology indicators: goods 1-2 are satellite; goods 3-4 are wired
satellite = Int.(product_id .<= 2)
wired     = Int.(product_id .>= 3)

# Exogenous product characteristic and cost shifter: absolute value of N(0,1)
x = abs.(randn(N))
w = abs.(randn(N))

# Demand and cost unobservables drawn jointly:
# (ξ_{jt}, ω_{jt})' ~ N(0, Σ) with Σ = [1 0.25; 0.25 1]
Σ = [1.0 0.25; 0.25 1.0]
Σ_chol = cholesky(Symmetric(Σ)).L # to draw multivariate normal
Z = randn(2, N)
ξω = Σ_chol * Z
ξ = vec(ξω[1, :])
ω = vec(ξω[2, :])

# Structural parameters
β1     = 1.0       # coefficient on quality x
α      = -2.0      # price coefficient
γ0     = 0.5       # intercept in log marginal cost
γ1     = 0.25      # slope on w in log marginal cost
β_rc_μ = 4.0       # mean of random coefficients on satellite and wired
β_rc_σ = 1.0       # std dev of those random coefficients

# Assemble product-market data
products = DataFrame(
    market_id = market_id,
    product_id = product_id,
    satellite = satellite,
    wired     = wired,
    x         = x,
    w         = w,
    xi        = ξ,
    omega     = ω
)

# 2(a)Start by writing a procedure to approximate the derivatives of market shares with respect to prices

# approximate integral over shares with simulated average over draws

# conditional shares s_t(p | b)
function conditional_shares_matrix(p, x, sat, wired, xi, α, β1, beta2_draws, beta3_draws)
    J = length(p)
    M = length(beta2_draws)
    S = Matrix{Float64}(undef, J, M)
    V_base = β1 .* x .+ α .* p .+ xi
    @inbounds for m in 1:M
        V = V_base .+ beta2_draws[m] .* sat .+ beta3_draws[m] .* wired
        vmax = max(0.0, maximum(V))
        logden = log1p(sum(exp.(V .- vmax))) + vmax
        S[:, m] = exp.(V .- logden)
    end
    return S
end

# Jacobian ∂s/∂p via simulated average over draws
# Returns J×J average derivative; optionally J×J MC standard errors
function jacobian_dsdp_market(p, x, sat, wired, xi, α, β1, beta2_draws, beta3_draws; return_se::Bool=false)
    S = conditional_shares_matrix(p, x, sat, wired, xi, α, β1, beta2_draws, beta3_draws)
    J, M = size(S)
    sumJ  = zeros(J, J)
    sumJ2 = return_se ? zeros(J, J) : nothing
    @inbounds for m in 1:M
        sm = @view S[:, m]
        Jm = α .* (Diagonal(sm) .- (sm * sm'))
        sumJ .+= Jm
        if return_se
            sumJ2 .+= Jm .^ 2
        end
    end
    Jbar = sumJ ./ M
    if return_se
        EX2 = sumJ2 ./ M
        var = EX2 .- Jbar .^ 2
        var .= max.(var, 0.0)
        SE  = sqrt.(var ./ M)
        return Jbar, SE
    else
        return Jbar
    end
end

# Step 4: Experiment to see how many simulation draws you need to get precise approximations.

# How does the average Jacobian entry stablisie as M increases?

function experiment_draws(p, x, sat, wired, xi, α, β1, beta2_draws_master, beta3_draws_master;
    Ms=[50, 100, 250, 500, 1000, 2000])

avg_diag = Float64[]
avg_off  = Float64[]

for M in Ms
J_M = jacobian_dsdp_market(p, x, sat, wired, xi, α, β1,
      view(beta2_draws_master, 1:M),
      view(beta3_draws_master, 1:M))
push!(avg_diag, mean(diag(J_M)))                # average own-price derivative
off_vals = [J_M[i,j] for i in 1:size(J_M,1), j in 1:size(J_M,2) if i != j]
push!(avg_off, mean(off_vals))   # average cross-price derivative
end

plot(Ms, avg_diag, lw=2, marker=:circle, label="Mean own-price effect",
xlabel="Number of draws (M)", ylabel="Average derivative", legend = :topright)
plot!(Ms, avg_off, lw=2, marker=:square, label="Mean cross-price effect")
end


# Experiment for a specific market id
function experiment_draws_market(t, products, α, β1, beta2_draws_master, beta3_draws_master;
    p::Union{Nothing,Vector{<:Real}}=nothing,
    Ms=[50, 100, 250, 500, 1000, 2000])

df_t = filter(row -> row.market_id == t, products)

# if no prices provided, use temporary random ones
if p === nothing
p = rand(length(df_t.x)) .+ 5
end

x     = df_t.x
sat   = df_t.satellite
wired = df_t.wired
xi    = df_t.xi

experiment_draws(p, x, sat, wired, xi, α, β1, beta2_draws_master, beta3_draws_master; Ms=Ms)
end

# for market 1 using random placeholder prices
M_master = 5000  # number of random-coefficient draws for all markets
beta2_draws_master = rand(Normal(β_rc_μ, β_rc_σ), M_master)
beta3_draws_master = rand(Normal(β_rc_μ, β_rc_σ), M_master)

experiment_draws_market(1, products, α, β1, beta2_draws_master, beta3_draws_master)

savefig("Assignments/Assignment 3- Demand Estimation/experiment_draws_convergence_random_prices.png")

# 2(c) Solve for equilibrium prices

# Mean shares s_t(p) via simulation
function mean_shares_market(p, x, sat, wired, xi, α, β1, beta2_draws, beta3_draws)
    S = conditional_shares_matrix(p, x, sat, wired, xi, α, β1, beta2_draws, beta3_draws)
    return vec(mean(S; dims=2))
end

# 1. Root-finding solution to the FOCs for a single market
function solve_prices_root_market(t, products, α, β1, beta2_draws_master, beta3_draws_master, γ0, γ1;
                                  M::Int=500, p0::Union{Nothing,Vector{<:Real}}=nothing)

    df_t = filter(row -> row.market_id == t, products)
    mc = exp.(γ0 .+ γ1 .* df_t.w .+ df_t.omega ./ 8)
    J  = length(df_t.x)
    p0 === nothing && (p0 = mc .* 1.25)

    d2 = view(beta2_draws_master, 1:M)
    d3 = view(beta3_draws_master, 1:M)

    function F!(F, p)
        s = mean_shares_market(p, df_t.x, df_t.satellite, df_t.wired, df_t.xi, α, β1, d2, d3)
        Δ = jacobian_dsdp_market(p, df_t.x, df_t.satellite, df_t.wired, df_t.xi, α, β1, d2, d3)
        μ = -Δ \ s
        @inbounds F .= p .- mc .- μ
    end

    res = NLsolve.nlsolve(F!, copy(p0))
    return (prices=collect(res.zero), converged=NLsolve.converged(res),
        iterations=res.iterations, method="root")
end

# 2. Morrow–Skerlos ζ-iteration (single-product firms: H = I)
function solve_prices_ms_market(t, products, α, β1, beta2_draws_master, beta3_draws_master, γ0, γ1;
    M::Int=500, tol::Float64=1e-8, maxit::Int=20_000,
    p0::Union{Nothing,Vector{<:Real}}=nothing)

df_t = filter(row -> row.market_id == t, products)
mc = exp.(γ0 .+ γ1 .* df_t.w .+ df_t.omega ./ 8)
J  = length(df_t.x)
p  = p0 === nothing ? mc .* 1.25 : copy(p0)
ζ  = p .- mc  # initialize markups

d2 = view(beta2_draws_master, 1:M)
d3 = view(beta3_draws_master, 1:M)

converged = false
it = 0
while it < maxit
it += 1

# shares across draws at current prices
S = conditional_shares_matrix(p, df_t.x, df_t.satellite, df_t.wired, df_t.xi, α, β1, d2, d3)
s = vec(mean(S; dims=2))              # length-J, E[s(b)]
Λ = α .* s                            # length-J (diagonal stored as vector)
Γ = α .* (S * S' ./ M)                # J×J, E[s(b) s(b)']

# ζ update: ζ ← Λ^{-1}(Γ ζ − s)
ζ_new = (Γ * ζ .- s) ./ Λ
p_new = mc .+ ζ_new

# FOC residual: r = s + (Λ − Γ) ζ_new  (∞-norm stopping)
r = s .+ Λ .* ζ_new .- (Γ * ζ_new)
if maximum(abs.(r)) < tol
converged = true
ζ = ζ_new
p = p_new
break
end

ζ = ζ_new
p = p_new
end

return (prices=p, converged=converged, iterations=it, method="MS")
end


# wrapper that runs a given per-market solver across all markets
function solve_all_markets(products, α, β1, beta2_draws_master, beta3_draws_master, γ0, γ1;
    M=500, method::Symbol=:root)

markets = unique(products.market_id)
out = Vector{Any}(undef, length(markets))

for (i, t) in enumerate(markets)
if method == :root
res = solve_prices_root_market(t, products, α, β1, beta2_draws_master, beta3_draws_master, γ0, γ1; M=M)
elseif method == :ms
res = solve_prices_ms_market(t, products, α, β1, beta2_draws_master, beta3_draws_master, γ0, γ1; M=M)
else
error("Unknown method: $method. Choose :root or :ms.")
end
out[i] = (market=t, prices=res.prices, converged=res.converged, method=res.method)
end
return out
end

results_root = solve_all_markets(products, α, β1, beta2_draws_master, beta3_draws_master, γ0, γ1; M=500, method=:root)
results_ms   = solve_all_markets(products, α, β1, beta2_draws_master, beta3_draws_master, γ0, γ1; M=500, method=:ms)

# check if the two methods converge to the same prices
max_diff = maximum(vcat([abs.(r.prices .- m.prices) for (r,m) in zip(results_root, results_ms)]...))
println("Max |Δp| across markets = ", max_diff)



# 3. Calculate “observed” shares for fake data set 

function write_observed_shares!(products, results, α, β1, beta2_draws_master, beta3_draws_master;
    M::Int=500, price_col::Symbol=:p_eq, share_col::Symbol=:s_obs)
products[!, price_col] = similar(products.x, Float64)
products[!, share_col] = similar(products.x, Float64)
d2 = view(beta2_draws_master, 1:M)
d3 = view(beta3_draws_master, 1:M)
for r in results
idx = findall(products.market_id .== r.market)
p   = r.prices
x   = products.x[idx]
sat = products.satellite[idx]
wir = products.wired[idx]
xi  = products.xi[idx]
s   = mean_shares_market(p, x, sat, wir, xi, α, β1, d2, d3)
products[!, price_col][idx] = p
products[!, share_col][idx] = s
end
return products
end

# Using shares from MS method
write_observed_shares!(products, results_ms,   α, β1, beta2_draws_master, beta3_draws_master; 
                       M=500, price_col=:p_eq_ms,   share_col=:s_obs_ms)

markets     = unique(products.market_id)
sat_share   = [sum(products.s_obs_ms[(products.market_id .== t) .& (products.satellite .== 1)]) for t in markets]
wired_share = [sum(products.s_obs_ms[(products.market_id .== t) .& (products.wired .== 1)])     for t in markets]

# Satellite vs Wired shares by market
scatter(sat_share, wired_share, xlabel="Satellite share", ylabel="Wired share",
        title="'Observed' shares: Satellite vs Wired (by market)", label=false)
plot!([0.0, 1.0], [0.0, 1.0], lw=1, linestyle=:dash, label=false)
savefig("Assignments/Assignment 3- Demand Estimation/q3_sat_vs_wired.png")

# 4) Estimate the plain logit model

# Build Berry outcome and nested-logit within-group logs (from observed shares)
products.y_berry = similar(products.x, Float64)
products.ln_within_sat = zeros(length(products.x))
products.ln_within_wired = zeros(length(products.x))

for t in unique(products.market_id)
    idx = findall(products.market_id .== t)
    s   = products.s_obs_ms[idx]
    s0  = 1.0 - sum(s)
    products.y_berry[idx] .= log.(s) .- log(s0)

    satmask = products.satellite[idx] .== 1
    wirmask = products.wired[idx]     .== 1
    S_sat = sum(s[satmask]);  S_wir = sum(s[wirmask])

    products.ln_within_sat[idx[satmask]]   .= log.(s[satmask]) .- log(S_sat)
    products.ln_within_wired[idx[wirmask]] .= log.(s[wirmask]) .- log(S_wir)
end

# OLS: y = βx + β_sat*satellite + αp  (no intercept in Berry transform)
y  = products.y_berry
Xo = hcat(products.x, products.satellite, products.p_eq_ms)
β_ols = Xo \ y
β1_ols   = β_ols[1]   # coefficient on x
βsat_ols = β_ols[2]   # coefficient on satellite dummy (wired is baseline)
α_ols    = β_ols[3]   # coefficient on price

# 5)
# 2SLS: instrument price with {x, w}; include all exogenous regressors in Z
X  = hcat(products.x, products.satellite, products.p_eq_ms)
Z  = hcat(products.x, products.satellite, products.w)
PZ = Z * inv(Z'Z) * Z'
β_iv = inv(X' * PZ * X) * (X' * PZ * y)
β1_iv   = β_iv[1]      # coefficient on x
βsat_iv = β_iv[2]      # coefficient on satellite dummy
α_iv    = β_iv[3]      # coefficient on price

# Comparison table
estimates = DataFrame(
    Parameter = ["β on x", "β on satellite", "α on price"],
    OLS = [β1_ols, βsat_ols, α_ols],
    TSLS = [β1_iv, βsat_iv, α_iv],
    True = [β1, 0, α]
)

# round for display
estimates.OLS  = round.(estimates.OLS,  digits=3)
estimates.TSLS = round.(estimates.TSLS, digits=3)

println(estimates)

# write LaTeX table for comparison
# write LaTeX table for comparison
open("Assignments/Assignment 3- Demand Estimation/q4_5_estimates.tex", "w") do f
    println(f, "\\begin{table}[H]")
    println(f, "\\centering")
    println(f, "\\caption{Plain logit estimates: OLS vs 2SLS (instrumenting price with x and w).}")
    println(f, "\\begin{tabular}{lccc}")
    println(f, "\\toprule")
    println(f, "Parameter & OLS & 2SLS & True\\\\")
    println(f, "\\midrule")
    println(f, "Beta on x & $(round(β1_ols, digits=3)) & $(round(β1_iv, digits=3)) & $(round(β1, digits=3))\\\\")
    println(f, "Beta on satellite & $(round(βsat_ols, digits=3)) & $(round(βsat_iv, digits=3)) & --\\\\")
    println(f, "Alpha on price & $(round(α_ols, digits=3)) & $(round(α_iv, digits=3)) & $(round(α, digits=3))\\\\")
    println(f, "\\bottomrule")
    println(f, "\\end{tabular}")
    println(f, "\\end{table}")
end



# 6) Nested logit 2SLS: y = β*x + α*p + σ_sat*ln(s|sat) + σ_wir*ln(s|wir)
Xn  = hcat(products.x, products.p_eq_ms, products.ln_within_sat, products.ln_within_wired)
Zn  = hcat(products.x, products.w,       products.ln_within_sat, products.ln_within_wired)  # instrument price only
PZn = Zn * inv(Zn'Zn) * Zn'
β_nl = inv(Xn' * PZn * Xn) * (Xn' * PZn * y)
β1_nl, α_nl, σ_sat, σ_wir = β_nl

open("Assignments/Assignment 3- Demand Estimation/q6_nestedlogit_estimates.tex", "w") do f
    println(f, "\\begin{table}[H]")
    println(f, "\\centering")
    println(f, "\\caption{Nested logit estimates (2SLS).}")
    println(f, "\\begin{tabular}{lc}")
    println(f, "\\toprule")
    println(f, "Parameter & Estimate\\\\")
    println(f, "\\midrule")
    println(f, "Beta on \$x\$ & $(round(β1_nl, digits=3))\\\\")
    println(f, "Alpha on price & $(round(α_nl, digits=3))\\\\")
    println(f, "Sigma (satellite) & $(round(σ_sat, digits=3))\\\\")
    println(f, "Sigma (wired) & $(round(σ_wir, digits=3))\\\\")
    println(f, "\\bottomrule")
    println(f, "\\end{tabular}")
    println(f, "\\end{table}")
end


# 7) Own-price elasticities and diversion ratios: true vs. nested-logit estimates

# Nested logit shares with group-specific σ
function nested_logit_shares(p, x, sat, wired, α̂, β̂, σ_sat, σ_wir)
    v = β̂ .* x .+ α̂ .* p
    idx_sat = findall(sat .== 1); idx_wir = findall(wired .== 1)
    s = zeros(length(p))
    if !isempty(idx_sat)
        den_sat = sum(exp.(v[idx_sat] ./ (1 - σ_sat)))
        s_cond_sat = exp.(v[idx_sat] ./ (1 - σ_sat)) ./ den_sat
        IV_sat = den_sat^(1 - σ_sat)
        # wired placeholder; compute below before S_g
        if !isempty(idx_wir)
            den_wir = sum(exp.(v[idx_wir] ./ (1 - σ_wir)))
            IV_wir = den_wir^(1 - σ_wir)
            denom = 1 + IV_sat + IV_wir
            S_sat = IV_sat / denom; S_wir = IV_wir / denom
            s[idx_sat] = s_cond_sat .* S_sat
            s[idx_wir] = (exp.(v[idx_wir] ./ (1 - σ_wir)) ./ den_wir) .* S_wir
        else
            denom = 1 + IV_sat
            S_sat = IV_sat / denom
            s[idx_sat] = s_cond_sat .* S_sat
        end
    else
        # only wired group present
        den_wir = sum(exp.(v[idx_wir] ./ (1 - σ_wir)))
        IV_wir = den_wir^(1 - σ_wir)
        denom = 1 + IV_wir
        S_wir = IV_wir / denom
        s[idx_wir] = (exp.(v[idx_wir] ./ (1 - σ_wir)) ./ den_wir) .* S_wir
    end
    return s
end

markets = unique(products.market_id)
J = 4
own_true = zeros(J); own_hat = zeros(J)
D_true = zeros(J,J); D_hat = zeros(J,J)

M = 500; h = 1e-4
d2 = view(beta2_draws_master, 1:M)
d3 = view(beta3_draws_master, 1:M)

for t in markets
    idx = findall(products.market_id .== t)
    p   = products.p_eq_ms[idx]
    x   = products.x[idx]
    sat = products.satellite[idx]
    wir = products.wired[idx]
    xi  = products.xi[idx]
    s_true = products.s_obs_ms[idx]

    Δ = jacobian_dsdp_market(p, x, sat, wir, xi, α, β1, d2, d3)
    own_true .+= diag(Δ) .* (p ./ s_true)
    for j in 1:J
        denom = -Δ[j,j]
        if denom != 0.0
            D_true[:, j] .+= Δ[:, j] ./ denom
        end
    end

    s_hat = nested_logit_shares(p, x, sat, wir, α_nl, β1_nl, σ_sat, σ_wir)
    Jhat = zeros(J,J)
    for j in 1:J
        pb = copy(p); pb[j] += h
        s2 = nested_logit_shares(pb, x, sat, wir, α_nl, β1_nl, σ_sat, σ_wir)
        Jhat[:, j] = (s2 .- s_hat) ./ h
    end
    own_hat .+= diag(Jhat) .* (p ./ s_hat)
    for j in 1:J
        denom = -Jhat[j,j]
        if denom != 0.0
            D_hat[:, j] .+= Jhat[:, j] ./ denom
        end
    end
end

m = length(markets)
own_true ./= m; own_hat ./= m
D_true   ./= m; D_hat   ./= m

outpath = "Assignments/Assignment 3- Demand Estimation/q7_results.tex"

open(outpath, "w") do f
    # Elasticities table
    println(f, "\\begin{table}[H]")
    println(f, "\\centering")
    println(f, "\\caption{Average own-price elasticities across markets.}")
    println(f, "\\begin{tabular}{lcc}")
    println(f, "\\hline")
    println(f, "Product & True & Nested logit est.\\\\")
    println(f, "\\hline")
    for j in 1:J
        println(f, "Good ", j, " & ", @sprintf("%.3f", own_true[j]), " & ",
                    @sprintf("%.3f", own_hat[j]), "\\\\")
    end
    println(f, "\\hline")
    println(f, "\\end{tabular}")
    println(f, "\\end{table}")
    println(f)

    # Diversion ratios: True
    println(f, "\\begin{table}[H]")
    println(f, "\\centering")
    println(f, "\\caption{Average diversion ratios across markets (true).}")
    println(f, "\\begin{tabular}{lcccc}")
    println(f, "\\hline")
    println(f, "From\\\\To & 1 & 2 & 3 & 4\\\\")
    println(f, "\\hline")
    for j in 1:J
        row = join([@sprintf("%.3f", D_true[k,j]) for k in 1:J], " & ")
        println(f, j, " & ", row, "\\\\")
    end
    println(f, "\\hline")
    println(f, "\\end{tabular}")
    println(f, "\\end{table}")
    println(f)

    # Diversion ratios: Nested-logit estimate
    println(f, "\\begin{table}[H]")
    println(f, "\\centering")
    println(f, "\\caption{Average diversion ratios across markets (nested-logit estimate).}")
    println(f, "\\begin{tabular}{lcccc}")
    println(f, "\\hline")
    println(f, "From\\\\To & 1 & 2 & 3 & 4\\\\")
    println(f, "\\hline")
    for j in 1:J
        row = join([@sprintf("%.3f", D_hat[k,j]) for k in 1:J], " & ")
        println(f, j, " & ", row, "\\\\")
    end
    println(f, "\\hline")
    println(f, "\\end{tabular}")
    println(f, "\\end{table}")
end

# 8) Report a table with the estimates of the demand parameters and standard errors. Do this three
# times: once when you estimate demand alone, then again when you estimate jointly with supply; and
# again with the “optimal IV”.

product_data = pd.DataFrame(Dict(
    "market_ids"  => products.market_id,
    "product_ids" => products.product_id,
    "firm_ids"    => products.product_id, # single product firms
    "prices"      => products.p_eq_ms,
    "shares"      => products.s_obs_ms,
    "x"           => products.x,
    "satellite"   => products.satellite,
    "wired"       => products.wired,
    "w"           => products.w
))

# Demand instruments:
# BLP rival-sum instruments built from exogenous characteristics (x, satellite, wired)
# Differentiation instruments built from x
# Same-nest quality index: sum of x of the other product in the same technology within a market
# Simple shifters (x, w, satellite, wired)
blp_all  = Array(pyblp.build_blp_instruments(pyblp.Formulation("0 + x + satellite + wired"), product_data))
K        = size(blp_all, 2) ÷ 2
blp_rival = blp_all[:, K+1:end]  # rival-sum

diff_iv = Array(pyblp.build_differentiation_instruments(pyblp.Formulation("0 + x"), product_data))

# Same-nest quality index
same_nest_x = similar(products.x, Float64)
for t in unique(products.market_id)
    idx = findall(products.market_id .== t)
    sat_mask = products.satellite[idx] .== 1
    wir_mask = products.wired[idx]     .== 1
    x_sat = products.x[idx[sat_mask]]
    x_wir = products.x[idx[wir_mask]]
    # sum within nest excluding self
    same_nest_x[idx[sat_mask]] .= (sum(x_sat) .- x_sat)
    same_nest_x[idx[wir_mask]]  .= (sum(x_wir) .- x_wir)
end

xcol  = reshape(collect(products.x), :, 1)
wcol  = reshape(collect(products.w), :, 1)
satc  = reshape(collect(products.satellite), :, 1)
wirc  = reshape(collect(products.wired), :, 1)
sncol = reshape(collect(same_nest_x), :, 1)

Zraw = hcat(blp_rival, diff_iv, sncol, xcol, wcol, satc, wirc)

# Drop near-constant and collinear columns
stds  = std.(eachcol(Zraw))
keep1 = findall(>(1e-10), stds)
Z1    = Zraw[:, keep1]

F = qr(Z1, Val(true)) # rank-revealing QR
r = findlast(x -> abs(x) > 1e-10, diag(F.R))
idx = r === nothing ? Int[] : sort!(collect(F.p[1:r]))
Z   = r === nothing ? zeros(size(Z1,1), 0) : Z1[:, idx]

# Assemble instrument data frames
k_d   = size(Z, 2)
colsD = ["demand_instruments$(j)" for j in 0:(k_d-1)]
df_d  = pd.DataFrame(Z; columns=colsD)

# Supply instruments: just use x (excluded from costs except through w)
df_s = pd.DataFrame(xcol; columns=["supply_instruments0"])

product_data = pd.concat([product_data, df_d, df_s]; axis=1)

# Model formulations:
# - Mean utility: intercept + prices + x + satellite; drop wired to avoid perfect partition
# - One random coefficient on satellite
# - Log marginal cost: intercept + w
X1 = pyblp.Formulation("1 + prices + x + satellite")
X2 = pyblp.Formulation("0 + satellite")
X3 = pyblp.Formulation("1 + w")

integ  = pyblp.Integration("halton", 500)
sigma0 = np.array([[0.5]])

# Demand-only
problem_d = pyblp.Problem((X1, X2), product_data, integration=integ, add_exogenous=false)
res_d = problem_d.solve(sigma=sigma0, method="2s", optimization=pyblp.Optimization("l-bfgs-b"))

# Joint demand + supply (log costs), initialized at demand-only estimates
beta0  = Array(res_d.beta)
sigma0 = Array(res_d.sigma)
problem_js = pyblp.Problem((X1, X2, X3), product_data, integration=integ, costs_type="log", add_exogenous=false)
res_js = problem_js.solve(sigma=sigma0, beta=beta0, method="2s",
                          optimization=pyblp.Optimization("l-bfgs-b"))

# Feasible optimal IV
oi = res_js.compute_optimal_instruments(method="approximate")
problem_opt = oi.to_problem()
res_opt = problem_opt.solve(sigma=Array(res_js.sigma), beta=Array(res_js.beta),
                            method="2s", optimization=pyblp.Optimization("l-bfgs-b"))

# Return (estimate, se) for a named beta; returns (NaN, NaN) if not found
get_beta = function(res, name::AbstractString)
    labs = Vector{String}(res.beta_labels)
    i = findfirst(==(name), labs)
    i === nothing ? (NaN, NaN) : (Array(res.beta)[i], Array(res.beta_se)[i])
end

# Return (estimate, se) for the single sigma
get_sigma = function(res)
    (Array(res.sigma)[1, 1], Array(res.sigma_se)[1, 1])
end

# Pull estimates
βx_d,  se_βx_d    = get_beta(res_d,  "x")
α_d,   se_α_d     = get_beta(res_d,  "prices")
βsat_d, se_βsat_d = get_beta(res_d,  "satellite")
σsat_d, se_σsat_d = get_sigma(res_d)

βx_js,  se_βx_js    = get_beta(res_js,  "x")
α_js,   se_α_js     = get_beta(res_js,  "prices")
βsat_js, se_βsat_js = get_beta(res_js,  "satellite")
σsat_js, se_σsat_js = get_sigma(res_js)

βx_o,  se_βx_o    = get_beta(res_opt,  "x")
α_o,   se_α_o     = get_beta(res_opt,  "prices")
βsat_o, se_βsat_o = get_beta(res_opt,  "satellite")
σsat_o, se_σsat_o = get_sigma(res_opt)

fmt = x -> isnan(x) ? "NA" : @sprintf("%.3f", x)

open("Assignments/Assignment 3- Demand Estimation/q8_pyblp_estimates.tex", "w") do f
    println(f, "\\begin{table}[H]")
    println(f, "\\centering")
    println(f, "\\caption{Random-coefficients logit estimates (PyBLP). Entries are estimate (s.e.).}")
    println(f, "\\begin{tabular}{lccc}")
    println(f, "\\hline")
    println(f, "Parameter & Demand only & Demand+Supply & Optimal IV\\\\")
    println(f, "\\hline")
    println(f, "\$\\beta_x\$ & ",
            "$(fmt(βx_d)) ($(fmt(se_βx_d))) & ",
            "$(fmt(βx_js)) ($(fmt(se_βx_js))) & ",
            "$(fmt(βx_o)) ($(fmt(se_βx_o)))\\\\")
    println(f, "\$\\alpha_{price}\$ & ",
            "$(fmt(α_d)) ($(fmt(se_α_d))) & ",
            "$(fmt(α_js)) ($(fmt(se_α_js))) & ",
            "$(fmt(α_o)) ($(fmt(se_α_o)))\\\\")
    println(f, "Mean on satellite & ",
            "$(fmt(βsat_d)) ($(fmt(se_βsat_d))) & ",
            "$(fmt(βsat_js)) ($(fmt(se_βsat_js))) & ",
            "$(fmt(βsat_o)) ($(fmt(se_βsat_o)))\\\\")
    println(f, "SD on satellite & ",
            "$(fmt(σsat_d)) ($(fmt(se_σsat_d))) & ",
            "$(fmt(σsat_js)) ($(fmt(se_σsat_js))) & ",
            "$(fmt(σsat_o)) ($(fmt(se_σsat_o)))\\\\")
    println(f, "\\hline")
    println(f, "\\end{tabular}")
    println(f, "\\end{table}")
end

# 9. Using your preferred estimates from the prior step (explain your
# preference), provide a table comparing the estimated own-price elasticities
# to the true own-price elasticities. Provide two additional tables showing
# the true matrix of diversion ratios and the diversion ratios implied by your
# estimates.

py_elasticities = Array(res_js.compute_elasticities())

markets = unique(products.market_id)
J = 4
own_est = zeros(J)
D_est   = zeros(J, J)

for t in markets
    idx = findall(products.market_id .== t)     # row indices for market t in product order 1..J
    s_t = products.s_obs_ms[idx]                # market t shares (PyBLP matches these at optimum)
    if ndims(py_elasticities) == 2 && size(py_elasticities, 2) == J
        # E_t[k,j] is elasticity of s_k wrt p_j in market t
        E_t = py_elasticities[idx, :]           # J × J
        # Own-price elasticities: take the diagonal
        own_est .+= diag(E_t)
        # Diversion: D_{kj} = (ε_{kj} * s_k) / (-ε_{jj} * s_j)
        for j in 1:J
            denom = -E_t[j, j] * s_t[j]
            if denom == 0.0
                D_est[:, j] .+= 0.0
            else
                D_est[:, j] .+= (E_t[:, j] .* s_t) ./ denom
                D_est[j, j]  = 0.0
            end
        end
    else
        # Only own elasticities returned; average those. Diversion requires cross terms and is omitted.
        own_est .+= py_elasticities[idx]
    end
end

m = length(markets)
own_est ./= m
if sum(D_est) != 0.0
    D_est ./= m
    for j in 1:J
        D_est[j, j] = 0.0
    end
end

outpath = "Assignments/Assignment 3- Demand Estimation/q9_elasticities_diversion.tex"
open(outpath, "w") do f
    # Own-price elasticities
    println(f, "\\begin{table}[H]")
    println(f, "\\centering")
    println(f, "\\caption{Average own-price elasticities (true vs estimated).}")
    println(f, "\\begin{tabular}{lcc}")
    println(f, "\\hline")
    println(f, "Product & True & Estimated\\\\")
    println(f, "\\hline")
    for j in 1:J
        println(f, "\$", j, "\$ & ",
            @sprintf("%.3f", own_true[j]), " & ",
            @sprintf("%.3f", own_est[j]), "\\\\")
    end
    println(f, "\\hline\\end{tabular}\\end{table}\n")

    # True diversion
    println(f, "\\begin{table}[H]\\centering")
    println(f, "\\caption{Average diversion ratios across markets (true).}")
    println(f, "\\begin{tabular}{lcccc}\\hline From\\\\To & 1 & 2 & 3 & 4\\\\\\hline")
    for j in 1:J
        row = join([@sprintf("%.3f", D_true[k,j]) for k in 1:J], " & ")
        println(f, j, " & ", row, "\\\\")
    end
    println(f, "\\hline\\end{tabular}\\end{table}\n")

    # Estimated diversion (only if cross elasticities were available)
    if sum(D_est) != 0.0
        println(f, "\\begin{table}[H]\\centering")
        println(f, "\\caption{Average diversion ratios across markets (estimated).}")
        println(f, "\\begin{tabular}{lcccc}\\hline From\\\\To & 1 & 2 & 3 & 4\\\\\\hline")
        for j in 1:J
            row = join([@sprintf("%.3f", D_est[k,j]) for k in 1:J], " & ")
            println(f, j, " & ", row, "\\\\")
        end
        println(f, "\\hline\\end{tabular}\\end{table}")
    end
end

# 11) Suppose firms 1 and 2 are proposing to merge. Use the pyBLP merger simulation procedure to
# provide a prediction of the post-merger equilibrium prices

# Product-by-firm ownership (N × 4), single-product firms baseline
OWN_BASE = zeros(Float64, N, 4)
for j in 1:4
    OWN_BASE[products.product_id .== j, j] .= 1.0
end

# Merge products a and b: move ownership of b into a's column
function ownership_after_merge_map(pair::NTuple{2,Int}, base::AbstractMatrix, product_id::AbstractVector{Int})
    a, b = pair
    own = copy(base)
    rows_b = findall(product_id .== b)
    own[rows_b, a] .= 1.0
    own[rows_b, b] .= 0.0
    own
end

# Merger of firms 1 and 2
own_12 = ownership_after_merge_map((1, 2), OWN_BASE, products.product_id)


# Baseline and post-merger prices (no efficiencies)
p_pre_est  = vec(Array(res_js.compute_prices()))
p_post_12  = vec(Array(res_js.compute_prices(; ownership=own_12)))

# 13) Now suppose instead that firms 1 and 3 are the ones to merge. Re-run the merger simulation.
# Provide a table comparing the (average across markets) predicted merger-induced price changes for
# this merger and that in part 11. Interpret the differences between the predictions for the two mergers.

# Merger of firms 1 and 2
own_13 = ownership_after_merge_map((1, 3), OWN_BASE, products.product_id)
p_post_13  = vec(Array(res_js.compute_prices(; ownership=own_13)))


# Average by product and differences
avg_by_product(v) = [mean(v[products.product_id .== j]) for j in 1:4]

avg_p0_est  = avg_by_product(p_pre_est)
avg_p12     = avg_by_product(p_post_12)
avg_p13     = avg_by_product(p_post_13)
avg_dp12    = avg_by_product(p_post_12 .- p_pre_est)
avg_dp13    = avg_by_product(p_post_13 .- p_pre_est)

avg_diff   = [avg_dp12[j] - avg_dp13[j] for j in 1:4]


open("Assignments/Assignment 3- Demand Estimation/q11_q12_mergers.tex", "w") do f
    println(f, raw"""\begin{table}[H]
\centering
\caption{Average post-merger prices by product: merger of firms 1 \& 2}
\begin{tabular}{lcc}
\hline Product & Pre-merger & Post-merger \\ \hline""")
    for j in 1:4
        println(f, "\$", j, "\$ & ",
                @sprintf("%.3f", avg_p0_est[j]), " & ",
                @sprintf("%.3f", avg_p12[j]), " \\\\")
    end
    println(f, raw"""\hline\end{tabular}\end{table}

\begin{table}[H]
\centering
\caption{Average post-merger prices by product: merger of firms 1 \& 3}
\begin{tabular}{lcc}
\hline Product & Pre-merger & Post-merger \\ \hline""")
    for j in 1:4
        println(f, "\$", j, "\$ & ",
                @sprintf("%.3f", avg_p0_est[j]), " & ",
                @sprintf("%.3f", avg_p13[j]), " \\\\")
    end
    println(f, raw"""\hline\end{tabular}\end{table}

\begin{table}[H]
\centering
\caption{Average price changes (post $-$ pre) by product and difference}
\begin{tabular}{lccc}
\hline Product & Merger 1--2 & Merger 1--3 & Difference \\ \hline""")
    for j in 1:4
        println(f, "\$", j, "\$ & ",
                @sprintf("%.3f", avg_dp12[j]), " & ",
                @sprintf("%.3f", avg_dp13[j]), " & ",
                @sprintf("%.3f", avg_diff[j]), " \\\\")
    end
    println(f, raw"\hline\end{tabular}\end{table}")
end

# 14) Consider the merger between firms 1 and 2, and suppose the firms demonstrate that by merging
# they would reduce marginal cost of each of their products by 15%. Furthermore, suppose that
# they demonstrate that this cost reduction could not be achieved without merging. Using the pyBLP
# software, re-run the merger simulation with the 15% cost saving. Show the predicted post-merger
# price changes (again, for each product, averaged across markets). What is the predicted impact of
# the merger on consumer welfare,a assuming that the total measure of consumers Mt is the same in
#each market t?


# Marginal costs from estimated supply (log costs): mc = exp(γ0 + γ1 * w + ω̂)
γ_labels = Vector{String}(res_js.gamma_labels)
γ        = Array(res_js.gamma)
γ0̂ = γ[findfirst(==("1"), γ_labels)]
γ1̂ = γ[findfirst(==("w"), γ_labels)]
ω̂  = vec(Array(res_js.omega))   # ensure N×1 -> N-vector
mc_hat = exp.(γ0̂ .+ γ1̂ .* products.w .+ ω̂)

# 15% cost reduction on products 1 and 2
mc_eff = copy(mc_hat)
for t in 1:T
    ia = (t-1)*J + 1
    ib = (t-1)*J + 2
    mc_eff[ia] *= 0.85
    mc_eff[ib] *= 0.85
end

# Post-merger prices with efficiencies
p_post_12_eff = vec(Array(res_js.compute_prices(; ownership=own_12, costs=mc_eff)))

# Average price changes by product
avg_dp12_eff = [mean((p_post_12_eff .- p_pre_est)[products.product_id .== j]) for j in 1:J]

open("Assignments/Assignment 3- Demand Estimation/q14_merger_efficiencies.tex", "w") do f
    println(f, "\\begin{table}[H]")
    println(f, "\\centering")
    println(f, "\\caption{Merger 1--2 with 15\\% cost reduction: average price changes by product}")
    println(f, "\\begin{tabular}{lc}")
    println(f, "\\hline Product & \$\\Delta p\$ \\\\ \\hline")
    for j in 1:J
        row = string("\$", j, "\$ & ",
                     @sprintf("%.3f", avg_dp12_eff[j]),
                     " \\\\")
        println(f, row)
    end
    println(f, "\\hline\\end{tabular}\\end{table}")
end


# Mean utilities from the estimated joint model
δ_hat = vec(Array(res_js.delta))
β_labels = Vector{String}(res_js.beta_labels)
β        = Array(res_js.beta)
β0̂   = β[findfirst(==("1"),        β_labels)]
α̂    = β[findfirst(==("prices"),   β_labels)]
βx̂   = β[findfirst(==("x"),        β_labels)]
βsat̂ = β[findfirst(==("satellite"),β_labels)]
σ̂    = Array(res_js.sigma)[1, 1]

# Consumer surplus from mean utilities under random-coefficient logit
function cs_from_delta(δ::AbstractVector{<:Real}, sat::AbstractVector{Int}, αp::Real; M::Int=1000, seed::Int=1999)
    rng = MersenneTwister(seed)
    draws = randn(rng, M) .* σ̂
    acc = 0.0
    @inbounds for m in 1:M
        V = δ .+ draws[m] .* sat
        vmax = max(0.0, maximum(V))
        acc += log1p(sum(exp.(V .- vmax))) + vmax
    end
    (acc / M) / (-αp)
end


# Consumer surplus change (per consumer)
ξ̂   = δ_hat .- (β0̂ .+ βx̂ .* products.x .+ βsat̂ .* products.satellite .+ α̂ .* p_pre_est)
δ_cf = β0̂ .+ βx̂ .* products.x .+ βsat̂ .* products.satellite .+ α̂ .* p_post_12_eff .+ ξ̂

ΔCS = 0.0
for t in 1:T
    idx = (t-1)*J + 1 : t*J
    sat = products.satellite[idx]
    ΔCS += cs_from_delta(δ_cf[idx], sat, α̂) - cs_from_delta(δ_hat[idx], sat, α̂)
end
ΔCS_avg = ΔCS / T

open("Assignments/Assignment 3- Demand Estimation/q14_cs_change.tex", "w") do f
    println(f, raw"\begin{table}[H]")
    println(f, raw"\centering")
    println(f, raw"\caption{Average change in consumer surplus (per consumer)}")
    println(f, raw"\begin{tabular}{lc}")
    println(f, raw"\hline $\Delta CS$ & ", @sprintf("%.3f", ΔCS_avg), raw" \\ \hline")
    println(f, raw"\end{tabular}\end{table}")
end

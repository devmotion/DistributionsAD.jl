## Uniform ##

logpdf(d::Uniform, x::TrackedReal) = uniformlogpdf(d.a, d.b, x)
logpdf(d::Uniform{<:TrackedReal}, x::Real) = uniformlogpdf(d.a, d.b, x)
function logpdf(d::Uniform{<:TrackedReal}, x::TrackedReal)
    return uniformlogpdf(d.a, d.b, x)
end
uniformlogpdf(a::Real, b::Real, x::Real) = logpdf(Uniform(a, b), x)
function uniformlogpdf(a::Real, b::Real, x::TrackedReal)
    track(uniformlogpdf, a, b, x)
end
function uniformlogpdf(a::TrackedReal, b::TrackedReal, x::Real)
    track(uniformlogpdf, a, b, x)
end
function uniformlogpdf(
    a::TrackedReal,
    b::TrackedReal,
    x::TrackedReal,
)
    track(uniformlogpdf, a, b, x)
end
@grad function uniformlogpdf(
    a::Real,
    b::Real,
    x::TrackedReal,
)
    xd = data(x)
    T = typeof(x)
    l = logpdf(Uniform(a, b), xd)
    f = isfinite(l)
    n = T(NaN)
    z = zero(T)
    return l, Δ -> (f ? (z, z, z) : (n, n, n))
end
for T in (:TrackedReal, :Real)
    @eval @grad function uniformlogpdf(
        a::TrackedReal,
        b::TrackedReal,
        x::$T,
    )
        ad = data(a)
        bd = data(b)
        T = typeof(a)
        l = logpdf(Uniform(ad, bd), x)
        f = isfinite(l)
        temp = 1/(bd - ad)^2
        dlda = temp
        dldb = -temp
        n = T(NaN)
        z = zero(T)
        return l, Δ -> (f ? (dlda * Δ, dldb * Δ, z) : (n, n, n))
    end
end

## Semicircle ##

function semicircle_dldr(r, x)
    diffsq = r^2 - x^2
    return -2 / r + r / diffsq
end
function semicircle_dldx(r, x)
    diffsq = r^2 - x^2
    return -x / diffsq
end

logpdf(d::Semicircle{<:Real}, x::TrackedReal) = semicirclelogpdf(d.r, x)
logpdf(d::Semicircle{<:TrackedReal}, x::Real) = semicirclelogpdf(d.r, x)
logpdf(d::Semicircle{<:TrackedReal}, x::TrackedReal) = semicirclelogpdf(d.r, x)

semicirclelogpdf(r, x) = logpdf(Semicircle(r), x)
M, f, arity = DiffRules.@define_diffrule DistributionsAD.semicirclelogpdf(r, x) =
    :(semicircle_dldr($r, $x)), :(semicircle_dldx($r, $x))
da, db = DiffRules.diffrule(M, f, :a, :b)
f = :($M.$f)
@eval begin
    @grad $f(a::TrackedReal, b::TrackedReal) = $f(data(a), data(b)), Δ -> (Δ * $da, Δ * $db)
    @grad $f(a::TrackedReal, b::Real) = $f(data(a), b), Δ -> (Δ * $da, _zero(b))
    @grad $f(a::Real, b::TrackedReal) = $f(a, data(b)), Δ -> (_zero(a), Δ * $db)
    $f(a::TrackedReal, b::TrackedReal)  = track($f, a, b)
    $f(a::TrackedReal, b::Real) = track($f, a, b)
    $f(a::Real, b::TrackedReal) = track($f, a, b)
end

## Binomial ##

binomlogpdf(n::Int, p::TrackedReal, x::Int) = track(binomlogpdf, n, p, x)
@grad function binomlogpdf(n::Int, p::TrackedReal, x::Int)
    return binomlogpdf(n, data(p), x),
        Δ->(nothing, Δ * (x / p - (n - x) / (1 - p)), nothing)
end

function binomlogpdf(n::Int, p::ForwardDiff.Dual{T}, x::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val = ForwardDiff.value(p)
    Δ = ForwardDiff.partials(p)
    return FD(binomlogpdf(n, val, x),  Δ * (x / val - (n - x) / (1 - val)))
end

## Negative binomial ##

# Note the definition of NegativeBinomial in Julia is not the same as Wikipedia's.
# Check the docstring of NegativeBinomial, r is the number of successes and
# k is the number of failures
_nbinomlogpdf_grad_1(r, p, k) = k == 0 ? log(p) : sum(1 / (k + r - i) for i in 1:k) + log(p)
_nbinomlogpdf_grad_2(r, p, k) = -k / (1 - p) + r / p

nbinomlogpdf(n::TrackedReal, p::TrackedReal, x::Int) = track(nbinomlogpdf, n, p, x)
nbinomlogpdf(n::Real, p::TrackedReal, x::Int) = track(nbinomlogpdf, n, p, x)
nbinomlogpdf(n::TrackedReal, p::Real, x::Int) = track(nbinomlogpdf, n, p, x)
@grad function nbinomlogpdf(r::TrackedReal, p::TrackedReal, k::Int)
    return nbinomlogpdf(data(r), data(p), k),
        Δ->(Δ * _nbinomlogpdf_grad_1(r, p, k), Δ * _nbinomlogpdf_grad_2(r, p, k), nothing)
end
@grad function nbinomlogpdf(r::Real, p::TrackedReal, k::Int)
    return nbinomlogpdf(data(r), data(p), k),
        Δ->(_zero(r), Δ * _nbinomlogpdf_grad_2(r, p, k), nothing)
end
@grad function nbinomlogpdf(r::TrackedReal, p::Real, k::Int)
    return nbinomlogpdf(data(r), data(p), k),
        Δ->(Δ * _nbinomlogpdf_grad_1(r, p, k), _zero(p), nothing)
end

function nbinomlogpdf(r::ForwardDiff.Dual{T}, p::ForwardDiff.Dual{T}, k::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val_p = ForwardDiff.value(p)
    val_r = ForwardDiff.value(r)
    Δ_r = ForwardDiff.partials(r)
    dr = _nbinomlogpdf_grad_1(val_r, val_p, k)
    Δ_p = ForwardDiff.partials(p)
    dp = _nbinomlogpdf_grad_2(val_r, val_p, k)
    Δ = ForwardDiff._mul_partials(Δ_r, Δ_p, dr, dp)
    return FD(nbinomlogpdf(val_r, val_p, k),  Δ)
end
function nbinomlogpdf(r::Real, p::ForwardDiff.Dual{T}, k::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val_p = ForwardDiff.value(p)
    Δ_p = ForwardDiff.partials(p) * _nbinomlogpdf_grad_2(r, val_p, k)
    return FD(nbinomlogpdf(r, val_p, k),  Δ_p)
end
function nbinomlogpdf(r::ForwardDiff.Dual{T}, p::Real, k::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val_r = ForwardDiff.value(r)
    Δ_r = ForwardDiff.partials(r) * _nbinomlogpdf_grad_1(val_r, p, k)
    return FD(nbinomlogpdf(val_r, p, k),  Δ_r)
end

## Poisson ##

poislogpdf(v::TrackedReal, x::Int) = track(poislogpdf, v, x)
@grad function poislogpdf(v::TrackedReal, x::Int)
      return poislogpdf(data(v), x),
          Δ->(Δ * (x/v - 1), nothing)
end

function poislogpdf(v::ForwardDiff.Dual{T}, x::Int) where {T}
    FD = ForwardDiff.Dual{T}
    val = ForwardDiff.value(v)
    Δ = ForwardDiff.partials(v)
    return FD(poislogpdf(val, x), Δ * (x/val - 1))
end

## PoissonBinomial ##

struct TuringPoissonBinomial{T<:Real, TV<:AbstractVector{T}} <: DiscreteUnivariateDistribution
    p::TV
    pmf::TV
end
function TuringPoissonBinomial(p::AbstractArray{<:Real})
    pb = Distributions.poissonbinomial_pdf_fft(p)
    @assert Distributions.isprobvec(pb)
    TuringPoissonBinomial(p, pb)
end
function logpdf(d::TuringPoissonBinomial{T}, k::Int) where T<:Real
    insupport(d, k) ? log(d.pmf[k + 1]) : -T(Inf)
end
quantile(d::TuringPoissonBinomial, x::Float64) = quantile(Categorical(d.pmf), x) - 1
PoissonBinomial(p::TrackedArray) = TuringPoissonBinomial(p)
Base.minimum(d::TuringPoissonBinomial) = 0
Base.maximum(d::TuringPoissonBinomial) = length(d.p)

poissonbinomial_pdf_fft(x::TrackedArray) = track(poissonbinomial_pdf_fft, x)
@grad function poissonbinomial_pdf_fft(x::TrackedArray)
    x_data = data(x)
    T = eltype(x_data)
    fft = poissonbinomial_pdf_fft(x_data)
    return  fft, Δ -> begin
        ((ForwardDiff.jacobian(x -> poissonbinomial_pdf_fft(x), x_data)::Matrix{T})' * Δ,)
    end
end

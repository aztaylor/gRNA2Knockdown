using Catalyst
using OrdinaryDiffEq
using Plots

rn = @reaction_network begin
    k1, 0 --> GFP
    k2, GFP --> 0
end

u0 = [:GFP => 1.0]
tspan = (0.0, 10.0)
params = [:k1 => 1.0, :k2 => 0.2]

oprob = ODEProblem(rn, u0, tspan, params)
sol = solve(oprob)

plot(sol)

dCasRx_GFP_rn = @reaction_network begin
    #Transcription of GFP (mRNA₁₁) dCasRx (mRNA₂) and the gRNA (mRNA₁₂)
    k₁₁, pl₁ + rnap --> mRNA₁₁
    k₂, pl₂ + rnap --> mRNA₂
    k₁₂, pl₁ + rnap --> mRNA₁₂
    #mRNA degradation
    kd₁₁, mRNA₁₁ --> 0
    kd₂, mRNA₂ --> 0
    kd₁₂, mRNA₁₂ --> 0
    #Translation of dCasRx
    kₜᵣ₂, mRNA₂+ribosome --> dCasRx
    #Binding of dCasRx to gRNA
    kₑ, dCasRx + mRNA₁₂ --> dCasRx_gRNA
    #Binding of dCasRx to GFP mRNA
    (kᵢ, k₋ᵢ), dCasRx + mRNA₁₁ <--> imRNA₁₁
    #Translation of GFP
    kₜᵣ₁₁, mRNA₁₁ + ribosome --> GFP
end

u0 = [:mRNA₁₁ => 0.0, :mRNA₂ => 0.0, :mRNA₁₂ => 0.0, :dCasRx => 0.0, :dCasRx_gRNA => 0.0, :imRNA₁₁ => 0.0, :GFP => 0.0, :rnap => 1500, :ribosome => 10.000, :pl₁ => 15.0, :pl₂ => 15.0]

tspan = (0.0, 50)
params = [:k₁₁ => 0.055, :k₂ => 0.018, :k₁₂ => 1.67, :kd₁₁ => 0.0067, :kd₂ => 0.0067, :kd₁₂ => 0.0067, :kₜᵣ₂ => 0.0086, :kₑ => 0.01, :kᵢ => 0.01, :k₋ᵢ => 0.01, :kₜᵣ₁₁ => 0.0336]


oprob = ODEProblem(dCasRx_GFP_rn, u0, tspan, params)
osol = solve(oprob)
oplot = plot(osol, vars=[:mRNA₁₁, :mRNA₂, :mRNA₁₂, :dCasRx, :dCasRx_gRNA, :imRNA₁₁, :GFP])
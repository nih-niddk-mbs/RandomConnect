### 2PI.jl

G(phi,nu) = 2*(1+cos(phi))/nu
F(phi,I) = 1-cos(phi) +I*(1+cos(phi))
Fu(phi) = 1+cos(phi)

dF(a,a1,I,phi,dphi,) = (F(phi,I)*a - F(phi-dphi,I)*a1)/dphi
dFu(a,a1,phi,dphi,) = (Fu(phi)*a- Fu(phi-dphi)*a1)/dphi


"""
    step_a3!(a3,C31,I,sigma2,h,N,phi,dphi)


"""
function step_a3!(a3,C31,I,sigma2,h,N,dphi)
    for i in 1:N
        phi = mod2pi(i*dphi)
        i1 = i == 1 ? N : i - 1
        a3[i] -= h*(dF(a3[i],a3[i1],I,phi,dphi) - sigma2*dFu(C31[i],C31[i1],phi,dphi))
    end
end

function step_a3!(a3,C31,I,sigma2,h,N,dphi,T)
    for t in 1:T
        step_a3!(a3,C31,I,sigma2,h,N,dphi)
    end
end

function step_C31!(C31,D31,beta,h,N,dphi)
    beta2 = beta^2
    for i in 1:N
        phi = mod2pi(i*dphi)
        i1 = i == 1 ? N : i - 1
        C31[i] -= h*(beta*C31 - beta2*D31)
    end
end

"""
    step_C31!(C31,D31,beta,h,N,dphi,T)

TBW
"""
function step_C31!(C31,D31,beta,h,N,dphi,T)
    for t in 1:T
        step_C31!(C31,D31,beta,h,N,dphi)
    end
end
"""
    step_C31!(a3,C31,D31,C11,I,beta,h,N,dphi)

TBW
"""
function step_C31tt(a3,C11,I,beta,h,N,phi,dphi)
    C31 = a3 .* Fu.(phi) .* C11
    D31 = zeros(N)
    step_C31tt!(C31,a3,C11,D31,I,beta,h,N,dphi)
    return C31
end
function step_C31tt!(C31,a3,C11,D31,I,beta,h,N,dphi)
    for i in 1:N
        phi = mod2pi(i*dphi)
        i1 = i == 1 ? N : i - 1
        C31[i] -= h*(beta*C31[i] - beta^2*D31[i] + dF(C31[i],C31[i1],I,phi,dphi) + dFu(a3[i],a3[i1],phi,dphi)*C11)
    end
end
function update_D31(Q,C32,lag)
    println(size(Q))
    println(size(C32))
    T =
    C32[:,1:length(Q) - lag] * Q[1+lag:end]
end

"""
    step_C32!(C32,a3,I,t,beta,h,N,dphi)

TBW
"""
function step_C32!(C32,a3,I,t,beta,h,N,dphi)
    for i in 1:N
        phi = mod2pi(i*dphi)
        i1 = i == 1 ? N : i - 1
        C32[i] -= h*( dF(C32[i],C32[i1],I,phi,dphi) + dFu(a3[i],a3[i1],phi,dphi)*exp(-beta*t))
    end
end

function step_C32(a3,I,beta,h,N,dphi,T)
    C32 = Matrix{Float64}(undef,N,T)
    C32c = zeros(N)
    for t in 1:T-1
        C32[:,t] = C32c
        step_C32!(C32c,a3,I,t*h,beta,h,N,dphi)
    end
    return C32
end


"""
    step_C33tt(C33,C31,a3,h,N,dphi)

TBW
"""
function step_C33tt!(C33,C31,a3,I,h,N,dphi)
    for i in 1:N
        phi = mod2pi(i*dphi)
        i1 = i == 1 ? N : i - 1
        for j in 1:i
            j1 = j == 1 ? N : j - 1
            C33[i,j] -= h*(dF(C33[i,j],C33[i1,j],I,phi,dphi) + dF(C33[i,j],C33[i,j1],I,phi,dphi)
            + dFu(a3[i],a3[i1],phi,dphi)*C31[j]+dFu(a3[j],a3[j1],phi,dphi)*C31[i])
        end
    end
end

function step_C33(C31,a3,I,h,N,T)
    Q = zeros(N)
    dphi = 2pi/N
    phi = collect(1:N)*dphi
    C33 = (Fu.(phi) .* a3) * (C31./ F.(phi,I))'
    for t in 1:T
        step_C33!(C33,C31,a3,I,h,N,dphi,T)
    end
    return C33
end
function step_C33!(C33,C31,a3,I,h,N,dphi,T)
    for t in 1:T
        step_C33!(C33,C31,a3,I,h,N,dphi)
    end
end
function step_C33!(C33,C31,a3,I,h,N,dphi)
    for i in 1:N
        phi = mod2pi(i*dphi)
        i1 = i == 1 ? N : i - 1
        C33[i] -= h*(dF(C33[i],C33[i1],I,phi,dphi) + dFu(a3[i],a3[i1],phi,dphi)*C31)
    end
end

"""
    steady_state()

TBW
"""

function steady_state(C11,T,N,I,beta,sigma,h,iterations)
    sigma2 = sigma^2
    a3,C32,C31,D31,C33 = initialize(C11,T,N,I,beta,h)
    steady_state!(a3,C11,C31,D31,C32,C33,I,beta,sigma2,h,N,2pi/N,T,iterations)
end
function steady_state!(a3,C11,C31,D31,C32,C33,I,beta,sigma2,h,N,dphi,T,iterations)
    for i in 1:iterations
        step_C31tt!(C31,a3,C11,D31,I,beta,h,N,dphi)
        step_a3!(a3,C31,I,sigma2,h,N,dphi,T)
        # step_C33tt!(C33,C31,a3,I,h,N,dphi,T)
        # step_C33!(C33,C31,a3,I,h,N,dphi,T)
        # step_C32!(C32,a3,I,t,beta,h,N,dphi)
        # D31 = update_D31(4*sigma2*C33,C32,lag)
    end
    return a3,C31,D31,C32,C33
end

function initialize(C11,T,N,I,beta,h)
    dphi = 2pi/N
    phi = dphi*collect(1:N) .- pi
    a3 = 1 ./ F.(phi,I)
    a3 /= sum(a3)
    C31 = step_C31tt(a3,C11,I,beta,h,N,phi,dphi)
    # C33 = step_C33(C31,a3,I,h,N,T)
    # C32 = step_C32(a3,I,beta,h,N,dphi,T)
    # D31 = update_D31(4*sigma*2*C33,C32,0)
    C32 = zeros(T)
    D31 = zeros(N)
    C33 = zeros(N,N)
    return a3,C32,C31,D31,C33
end

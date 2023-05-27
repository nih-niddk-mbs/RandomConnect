### 2PI.jl

G(phi,nu) = 2*(1+cos(phi))/nu
F(phi,I) = 1-cos(phi) + I*(1+cos(phi))
Fu(phi) = 1+cos(phi)
Fprime(phi,I) = (1-I)*I*cos(phi)/(1+cos(phi) + I*cos(phi))

dF(a,a1,I,phi,phi1,dphi) = (F(phi,I)*a - F(phi1,I)*a1)/dphi
dFu(a,a1,phi,phi1,dphi) = (Fu(phi)*a - Fu(phi1)*a1)/dphi

transform(phi,I) =2*atan(sqrt(I)*tan(phi/2))
transform_inv(v,I) =2*atan(tan(v/2)/sqrt(I))

function dFa(a3,phi)


end

function transform_a3(a3,phi,I)
    N = length(a3)
    dphi = 2pi/N
    at = similar(a3)
    for i in eachindex(at)
        p = transform(phi[i],I)
        idx = phase_index(p,N)
        dp = phi[idx] - p
        at[i] = ((dphi-dp)*a3[idx] + dp*a3[idx-1])/dphi
    end
    return at .* jacobian.(phi,I)
end

jacobian(phi,I) = 2*sqrt(I)/(1+cos(phi) +I*(1-cos(phi)))

"""
    step_a3!(a3,C31,I,sigma2,h,N,phi,dphi)


"""

steadystate_a30(I,phi) = sqrt(I)/pi./ F.(phi,I)


function steadystate_a3(a3,C13,I,N)
    dphi = 2pi/N
    phi = collect(1:N)*dphi .- pi .- .5*dphi 
    a30 = 1 ./ F.(phi,I)
    A = Fu.(phi) ./ F.(phi,I) .* C13
    K = (1 + sum(A)*dphi)/(sum(a30)*dphi)
    return K * a30 .- A, F.(phi,I) .* a3 + Fu.(phi) .* C13 .- K, a30*sqrt(I)/pi
end
function step_a3!(a3,C31,I,sigma2,h,N,dphi)
    for i in 1:N
        i1 = i == 1 ? N : i - 1
        a3[i] -= h*(dF(a3[i],a3[i1],I,phi[i],phi[i1],dphi) - sigma2*dFu(C31[i],C31[i1],phi[i],phi[i1],dphi))
    end
end

function step_a3!(a3,C31,I,sigma2,h,N,dphi,T)
    for t in 1:T
        step_a3!(a3,C31,I,sigma2,h,N,dphi)
    end
end

function step_1s(a3,C11,D11,C31,D31,C33,D33,t,sigma2,nu,h,N,dphi,indpi)
    D11 += h*(nu^2*beta2*sigma2*C33[indpi,indpi] - beta*D11)
    C11 += h*(D11 - beta*C11)
    for indchi in 1:N
        phi = mod2pi(indchi*dphi + nu*t)
        indphi = floor(Int,phi/dphi)
        indphi = indphi == 0 ? N : indphi
        indphi1 = indphi == 1 ? N : indphi - 1
        dGa3 = (G(phi,nu)*a3[indphi] - G(phi-dphi,nu)*a3[indphi1])/dphi
        D31[indchi] -= h*dGa3*D11
        C31[indchi] += h*(D31[indchi] - beta*C31[indchi])
        for indchip in 1:N
            chip = mod2pi(indchip*dphi)
            indchip1 = indchip == 1 ? N : indchip - 1
            D33[indchi,indchip] += h*dGa3*(G(chip,nu)*a3[indchip] - G(chip-dphi,nu)*a3[indchip1])/dphi*C11
        end
    end
    return C11,D11,C31,D31,.5*(D33+D33')
end
function step_2s(a3,C11,D11,C31,D31,C33,D33,t,sigma2,nu,h,N,dphi,indpi)

    for indchi in 1:N
        phi = mod2pi(indchi*dphi + nu*t)
        indphi = floor(Int,phi/dphi)
        indphi = indphi == 0 ? N : indphi
        indphi1 = indphi == 1 ? N : indphi - 1
        a3[indchi] -= h*sigma2*(G(phi,nu)*C31[indphi] - G(phi-dphi,nu)*C31[indphi1])/dphi
        for indchip in 1:N
            phip = mod2pi(indchip*dphi + nu*t)
            indphip = floor(Int,phip/dphi)
            indphip = indphip == 0 ? N : indphip
            C33[indchi,indchip] += h*D33[indchi,indphip]
        end
    end
    return a3,.5*(C33+C33')
end


function step_1(a3,C11,D11,C31,D31,C33,D33,t,sigma2,nu,h,N,dphi,indpi)
    D11 += h*(nu^2*beta2*sigma2*C33[indpi,indpi] - beta*D11)
    C11 += h*(D11 - beta*C11)
    for indchi in 1:N
        phi = mod2pi(indchi*dphi + nu*t)
        indphi = floor(Int,phi/dphi)
        indphi = indphi == 0 ? N : indphi
        indphi1 = indphi + 1 > N ? 1 : indphi + 1
        dGa3 = (G(phi+dphi,nu)*a3[indphi1] - G(phi,nu)*a3[indphi])/dphi
        D31[indchi] -= h*dGa3*D11
        C31[indchi] += h*(D31[indchi] - beta*C31[indchi])
        for indchip in 1:N-1
            chip = mod2pi(indchip*dphi)
            D33[indchi,indchip] += h*dGa3*(G(chip+dphi,nu)*a3[indchip+1] - G(chip,nu)*a3[indchip])/dphi*C11
        end
        D33[indchi,N] += h*dGa3*(G(dphi,nu)*a3[1] - G(0,nu)*a3[N])/dphi*C11
    end
    return C11,D11,C31,D31,.5*(D33+D33')
end
function step_2(a3,C11,D11,C31,D31,C33,D33,t,sigma2,nu,h,N,dphi,indpi)

    for indchi in 1:N
        phi = mod2pi(indchi*dphi + nu*t)
        indphi = floor(Int,phi/dphi)
        indphi = indphi == 0 ? N : indphi
        indphi1 = indphi + 1 > N ? 1 : indphi + 1
        a3[indchi] -= h*sigma2*(G(phi+dphi,nu)*C31[indphi1] - G(phi,nu)*C31[indphi])/dphi
        for indchip in 1:N
            phip = mod2pi(indchip*dphi + nu*t)
            indphip = floor(Int,phip/dphi)
            indphip = indphip == 0 ? N : indphip
            C33[indchi,indchip] += h*D33[indchi,indphip]
        end
    end
    return a3,.5*(C33+C33')
end

function evolve(T,sigma2,nu,h,N,f)
    a3 = 2pi/N .+ .2/N .* cos.(2pi*collect(1:N)/N)
    # a3 = ones(N)*2pi/N
    C31 = zeros(N)
    D31 = zeros(N)
    C33 = zeros(N,N)
    # for i in 1:N
    #     C33[i,i] = 1/N/N
    # end
    D33 = zeros(N,N)
    C11 = 0.01
    D11 = 0.
    f(a3,C11,D11,C31,D31,C33,D33,T,sigma2,nu,h,N)
end

function rotateback(v::Vector,t,nu)
    n = floor(Int,length(v)*t*nu/2pi)
    circshift(v,n)
end

function rotateback(m::Matrix,t,nu)
    n = floor(Int,size(m)[1]*t*nu/2pi)
    circshift(m,(n,n))
end


function evolve_C33(a3,C11,D11,sigma2,nu,h,N)
    a3t = a3
    C31t = C31
    dphi = 2pi /N
    tau = 0
    n = 1
    x = [0.]
    for t in 0:h:T
        indpi = floor(Int,mod2pi(pi+nu*t)/dphi)
        indpi = indpi == 0 ? N : indpi
        a3,C11,D11,C31,D31,C33,D33 = step(a3,C11,D11,C31,D31,C33,D33,t,sigma2,nu,h,N,dphi,indpi)
        if tau >= dphi/nu
            a3t = hcat(a3t,circshift(a3,n))
            C31t = hcat(C31t,circshift(C31,n))
            push!(x,t)
            tau -= dphi/nu
            n += 1
        end
        tau += h
    end
    return rotateback(a3,T,nu),C11,D11,rotateback(C31,T,nu),rotateback(D31,T,nu),rotateback(C33,T,nu),rotateback(D33,T,nu),a3t,C31t,x,collect(1:N)
end

function evolves(a3,C11,D11,C31,D31,C33,D33,T,sigma2,nu,h,N)
    a3t = a3
    C31t = C31
    dphi = 2pi /N
    tau = 0
    n = 1
    x = [0.]
    for t in 0:h:T
        indpi = floor(Int,mod2pi(pi+nu*t)/dphi)
        indpi = indpi == 0 ? N : indpi
        a3,C11,D11,C31,D31,C33,D33 = steps(a3,C11,D11,C31,D31,C33,D33,t,sigma2,nu,h,N,dphi,indpi)
        if tau >= dphi/nu
            a3t = hcat(a3t,circshift(a3,n))
            C31t = hcat(C31t,circshift(C31,n))
            push!(x,t)
            tau -= dphi/nu
            n += 1
        end
        tau += h
    end
    return rotateback(a3,T,nu),C11,D11,rotateback(C31,T,nu),rotateback(D31,T,nu),rotateback(C33,T,nu),rotateback(D33,T,nu),a3t,C31t,x,collect(1:N)
end


function evolve_fd(a3,C11,D11,C31,D31,C33,D33,T,sigma2,nu,h,N)
    a3t = a3
    C31t = C31
    dphi = 2pi / N
    indpi = div(N,2)
    tau = 0
    x = [0.]
    for t in 0:h:T
        a3,C11,D11,C31,D31,C33,D33 = step_fd(a3,C11,D11,C31,D31,C33,D33,sigma2,nu,h,N,dphi,indpi)
        if tau >= min(dphi/nu,1.)
            a3t = hcat(a3t,a3)
            C31t = hcat(C31t,C31)
            push!(x,t)
            tau -= dphi/nu
        end
        tau += h
    end
    return a3,C11,D11,C31,D31,C33,D33,a3t,C31t,x,collect(1:N)
end


function evolve_t(a3,C11,D11,C33in,D33in,T,sigma2,nu,h,N)
    C33 = copy(C33in)
    D33 = copy(D33in)
    C11t = [C11]
    dphi = 2pi /N
    tau = 0
    x = [0.]
    for t in 0:h:T
        indpi = floor(Int,mod2pi(pi+nu*t)/dphi)
        indpi = indpi == 0 ? N : indpi
        C11,D11,C33,D33 = step_t(a3,C11,D11,C33,D33,t,sigma2,nu,h,N,dphi,indpi)
        if tau >= dphi/nu
            push!(C11t,C11)
            push!(x,t)
            tau -= dphi/nu
        end
        tau += h
    end
    return C11t,x
end





"""
    step_C31!(C31,D31,beta,h,N,dphi,T)

TBW
"""
function step_C31!(C31,D31,beta,h,N,dphi)
    beta2 = beta^2
    for i in 1:N
        phi = mod2pi(i*dphi)
        i1 = i == 1 ? N : i - 1
        C31[i] -= h*(beta*C31 - beta2*D31)
    end
end
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
        C31[i] -= h*(beta*C31[i] - beta^2*D31[i] + dF(C31[i],C31[i1],I,phi[i],phi[i1],dphi) + dFu(a3[i],a3[i1],phi[i],phi[i1],dphi)*C11)
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


function step_D33tau!(C33,C11,a3,I,h,N,dphi)
        i1 = i == 1 ? N : i - 1
        C33[i] -= h*(dF(C33[i],C33[i1],I,phi[i],phi[i1],dphi) + dFu(a3[i],a3[i1],phi[i],phi[i1],dphi)*C31)
end

function step_D33tau!(C33,C11,a3,I,h,N,dphi)
    for i in 1:N
        for j in 1:i
         phi = mod2pi(i*dphi)
        i1 = i == 1 ? N : i - 1
        C33[i,j] -= h*(dF(C33[i,j],C33[i1,j],I,phi,dphi) + dFu(a3[i],a3[i1],phi,dphi)*C11)
        end
    end
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

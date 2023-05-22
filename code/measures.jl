
# measures.jl


"""
phase_indices(phase,Nphases,u,l)

Assign index on domain 1 to Nphases to phase on domain l to u

"""
phase_indices(phases,Nphases,u,l) = max.(ceil.(Int,(phases .- l)./(u-l)*Nphases),1)

"""
a3!(rho,phases,Nphases)

Find mean phase density
averaged over network and time

- `phase': matrix of phases where rows = neurons, cols = time

"""
function rho!(rho::Matrix,phases,Nphases)
    dN = 1/size(phases,1)
    for t in 1:size(phases,2)
        phase_idx = phase_indices(phases[:,t],Nphases)
        for i in phase_idx
            rho[i,t] += dN
        end
    end
end
function rho!(rho::Vector,phases,Nphases)
    phase_idx = phase_indices(phases,Nphases)
    dN = 1/length(phases)
    for i in phase_idx
        rho[i] += dN
    end
end



"""
a3(phase,Nphases=100)



"""

function rho(phases::Matrix,Nphases)
    rho = zeros(Nphases,size(phases,2))
    rho!(rho,phases,Nphases)
    return rho
end

function rho(phases::Vector,Nphases)
    rho = zeros(Nphases)
    rho!(rho,phases,Nphases)
    return rho
end

"""
covariance(x,y,lags)

Find cross-covariance function of matrices x and y over time lags
covariance is taken over neurons, while function is averaged over time
- `x`,`y`: are matrices where rows are neuron indices and columns are time indices

"""
function covariance(x,y,lags)
    T = size(x,2) -lags[end]
    c = zeros(length(lags))
    for t in 1:T
        for (i,lag) in enumerate(lags)
            c[i] += cov(x[t,:],y[t+lag,:])
        end
    end
    c/T
end


"""
c11(u,lags)

Find covariance function of synaptic input

- `u`: matrix of inputs, rows = neurons, cols = time
"""
c11(u,lags) = covariance(u,u,lags)

"""
m13(phases,u,Nphases)

"""
function m13(phases,u,Nphases)
    rho = zeros(Nphases)
    phase_idx = phase_indices(phases,Nphases)
    for (i,p) in enumerate(phase_idx)
        rho[p] += u[i]/size(phases,2)
    end
    rho/Nphases
end
"""
c13(phases,u,Nphases)

"""
c13(phases,u,Nphases) = m13(phases,u,Nphases) - mean(u)*rho3(phases,Nphases)

function c13(phases,u,lags,Nphases)
    T = size(phases,2) -lags[end]
    c = zeros(Nphases,length(lags))
    a1 = mean(u,dims=1)
    for t in 1:T
        rho = rho3(phases[:,t],Nphases)
        for (ind, lag) in enumerate(lags)
            c[:,ind] .+= m13(phases[:,t],u[:,t+lag],Nphases) - rho*a1[t+lag]
        end
    end
    c/T
end

"""
M33(phases1,phases2,Nphases)


"""

function m33!(c,phases1,phases2,Nphases)
    phase_idx1 = phase_indices(phases1,Nphases)
    phase_idx2 = phase_indices(phases2,Nphases)
    dN =1/size(phases1,2)
    for i in 1:size(phases1,2)
        c[phase_idx1[i],phase_idx2[i]] += dN
    end
end

function m33(phases1::Matrix,phases2::Matrix,lags,Nphases)
    T = size(phases1,1)
    c = zeros(Nphases,Nphases,length(lags))
    for t in 1:T - lags[end]
        for (ind, lag) in enumerate(lags)
            m33!(c[:,:,ind],phases[t,:],phases[t+lag,:],Nphases)
        end
    end
    return c/T
end

function rhorho(phases1::Matrix,phases2::Matrix,lags,Nphases)
    T = size(phases1,1)
    c = zeros(Nphases,Nphases,length(lags))
    rho1 = zeros(Nphases,T)
    rho2 = zeros(Nphases,T)
    rho!(rho1,phases1,Nphases)
    rho!(rho2,phases2,Nphases)
    for t in 1:T - lags[end]
        for (ind, lag) in enumerate(lags)
            c[:,:,ind] .-= rho1[:,t]*rho2[:,t+lag]'
        end
    end
    c/T
end
"""
c33(phase,lags,Nphases)

"""

c33(phases1::Vector,phases2::Vector,Nphases) = M33(phases1,phases2,Nphases) - a3(phases1,Nphases)*a3(phases2,Nphases)'

function c33(phases1::Matrix,phases2::Matrix,lags,Nphases)
    T = size(phases1,1)
    c = zeros(Nphases,Nphases,length(lags))
    rho1 = zeros(Nphases,T)
    rho2 = zeros(Nphases,T)
    rho!(rho1,phases1,Nphases)
    rho!(rho2,phases2,Nphases)
    for t in 1:T - lags[end]
        for (ind, lag) in enumerate(lags)
            b = @view c[:,:,ind]
            m33!(b,phases[t,:],phases[t+lag,:],Nphases)
            c[:,:,ind] .-= rho1[:,t]*rho2[:,t+lag]'
        end
    end
    c/T

    function m33(phases::Matrix,lags,Nphases)
        T = size(phases1,2)
        c = zeros(Nphases,Nphases,length(lags))
        rho = zeros(Nphases,T)
        rho!(rho,phases,Nphases)
        for t in 1:T - lags[end]
            for (ind, lag) in enumerate(lags)
                b = @view c[:,:,ind]
                m33!(b,phases[t,:],phases[t+lag,:],Nphases)
                c[:,:,ind] .-= rho1[:,t]*rho2[:,t+lag]'
            end
        end
        c/T
    end
end

function m33(phases::Matrix,lags,Nphases,bottom=-pi,top=pi)
    T = size(phases,1)
    c = zeros(Nphases,Nphases,length(lags))
    dN =1/size(phases,2)
    rho = rho(phases,Nphases)
    indices = phase_indices(phases,Nphases,bottom,top)
    for t in 1:T - lags[end]
        for (ind, lag) in enumerate(lags)
            c[indices[t,:],indices[t+lag,:],ind] .+= dN
        end
    end
    c/T
end

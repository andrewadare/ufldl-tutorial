using MAT
using Images, ImageView

include("dltUtils.jl")

const nChannels = 3
const patchDim = 8
const nPatches = 100_000
const nv = patchDim^2 * nChannels
const nh = 400
const lambda = 3e-3
const sparsity = 0.035
const beta = 5.0
const epsilon = 0.1

function checkGrad()
    nvDebug, nhDebug = 8, 5
    patches = rand(8,10)
    theta = initWeights(nvDebug, nhDebug)

    # Compute gradient from backprop algorithm
    cost, grad = saeLinCost(theta, nvDebug, nhDebug, lambda, beta, sparsity, patches)

    # Compute gradient directly from cost function
    J(t) = saeLinCost(t, nvDebug, nhDebug, lambda, beta, sparsity, patches)
    numgrad = numericalGradient(J, theta)

    # Compare them
    display([numgrad grad])
    diff = norm(numgrad-grad)/norm(numgrad+grad)
    report = diff < 1e-9 ? "PASS" : "FAIL"
    println("$report: Numerical - analytic gradient norm difference = $diff.")
end

function loadSTL10Images()
    imFile = matopen("data/stlSampledPatches.mat") # From tutorial.
    names(imFile)                       # lists one name: "patches"
    imgs = read(imFile, "patches")      # 192x10_000 Array{Float64,2}:

end

function displayColorNetwork(A::Matrix, saveName::ASCIIString)
    # Shift midpoint if not at zero
    A -= mean(A)

    ncols = int(round(sqrt(size(A,2))))
    nrows = int(ceil(size(A,2)/ncols))
    ppc = int(size(A,1)/3) # pixels per channel
    d = int(sqrt(ppc))
    e = d+1

    B = A[1:ppc,:]
    C = A[ppc+1:2*ppc,:]
    D = A[2*ppc+1:3*ppc,:]

    B = B./(ones(size(B,1),1)*maximum(abs(B),1));
    C = C./(ones(size(C,1),1)*maximum(abs(C),1));
    D = D./(ones(size(D,1),1)*maximum(abs(D),1));

    I = ones(d*nrows+nrows-1, d*ncols+ncols-1, 3)

    for i = 0:nrows-1
        for j = 0:ncols-1
            if i*ncols+j+1 > size(B, 2)
                break
            end

            rows, cols = i*e+1:i*e+d, j*e+1:j*e+d
            I[rows,cols,1] = reshape(B[:,i*ncols+j+1],d, d)
            I[rows,cols,2] = reshape(C[:,i*ncols+j+1],d, d)
            I[rows,cols,3] = reshape(D[:,i*ncols+j+1],d, d)
        end
    end

    I += 1
    I /= 2
    view(I)
    if (length(saveName) > 0)
        imwrite(colorim(I), saveName)
    end
end

function main()
    # checkGrad()

    patches = loadSTL10Images()
    displayColorNetwork(patches[:,1:100], "output/stl10patches.jpg")

    # ZCA whitening
    sigma = patches * patches' / nPatches
    u, s, vt = svd(sigma)
    z = u * (diagm(1 ./ sqrt(s + epsilon)) * u')
    patches = z*patches

    displayColorNetwork(patches[:,1:100], "output/stl10patches_zca.jpg")

    if true
        theta = initWeights(nv, nh)

        # Train autoencoder with linear decoder
        alg = :LD_LBFGS
        npars = length(theta)
        opt = Opt(alg, npars)
        ftol_abs!(opt, 1e-6)
        ftol_rel!(opt, 1e-6)
        maxeval!(opt, 400)
        lower_bounds!(opt, -5.0*ones(npars))
        upper_bounds!(opt, +5.0*ones(npars))
        println("Using ", algorithm_name(opt))

        # Wrap the cost function to match the signature expected by NLopt
        ncalls = 0
        function f(x::Vector, grad::Vector)
            J, grad[:] = saeLinCost(x,nv,nh,lambda,beta,sparsity,patches)
            
            ncalls += 1
            ng = norm(grad)
            println("$ncalls: J = $J, grad = $ng")
            
            J
        end

        min_objective!(opt, f)
        (minCost, optTheta, status) = optimize!(opt, theta)
        println("Cost = $minCost (returned $status)")
        optTheta, minCost, status

        writedlm("data/stl10features.txt", optTheta)
    end
    
    optTheta = readdlm("data/stl10features.txt")

    W = reshape(optTheta[1:nv * nh], nh, nv)
    b = optTheta[2*nh*nv+1:2*nh*nv+nh]
    displayColorNetwork((W*z)', "output/lindecoder_features.jpg")

end

main()


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

function displayColorNetwork(A::Matrix)
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
end

function whitenZCA!(x, m, epsilon)
    sigma = x * x' / m
    u, s, v = svd(sigma)
    w = (1 ./ sqrt(s + epsilon)) .* (u'*x)
    x[:] = u*w
end

# checkGrad()

patches = loadSTL10Images()
displayColorNetwork(patches[:,1:100])

whitenZCA!(patches, nPatches, epsilon)
displayColorNetwork(patches[:,1:100])


# TODO optimize theta and have a look at the features. 
# TODO move patches out of global scope.

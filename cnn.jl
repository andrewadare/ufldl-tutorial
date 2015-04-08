using MAT
using Images, ImageView

include("dltUtils.jl")

const nChannels = 3                # r,g,b
const imgDim = 64                  # Images are 64x64
const patchDim = 8                 # Patches are 8x8
const poolDim = 19                 # Size of pooling region
const nPatches = 50_000            
const nv = patchDim^2 * nChannels
const nh = 400
const epsilon = 0.1

function main()
    optTheta  = readdlm("data/stl10features.txt")
    meanPatch = readdlm("data/patch_means.txt")
    zcaWhite  = readdlm("data/zcawhite.txt")

    W = reshape(optTheta[1:nv * nh], nh, nv)
    b = optTheta[2*nh*nv+1:2*nh*nv+nh]
    displayColorNetwork((W*zcaWhite)')
end

main()
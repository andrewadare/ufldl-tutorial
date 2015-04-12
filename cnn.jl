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

# Load paramaters that were saved in linearDecoder.jl
const optTheta  = readdlm("data/stl10features.txt")
const meanPatch = readdlm("data/patch_means.txt")
const zcaWhite  = readdlm("data/zcawhite.txt")
const W = reshape(optTheta[1:nv * nh], nh, nv)
const b = optTheta[2*nh*nv+1:2*nh*nv+nh]

# Load training and test data (provided as part of UFLDL tutorial)
const trainDataFile = matopen("data/stlTrainSubset.mat")
const testDataFile  = matopen("data/stlTestSubset.mat")
const trainLabels   = read(trainDataFile, "trainLabels")
const testLabels    = read(testDataFile, "testLabels")
const trainImages   = read(trainDataFile, "trainImages")
const testImages    = read(testDataFile, "testImages")
const nTrain        = read(trainDataFile, "numTrainImages") # 2000 images
const nTest         = read(testDataFile, "numTestImages")   # 3200 images

function testConvolution()
    # Test cnnConvolve function on the first 8 training images. 
    # (convImages is 64 x 64 x 3 x 8).
    convImages = trainImages[:, :, :, 1:8]
    for i = 1:8
        view(convImages[:,:,:,i])
    end
    convolvedFeatures = 
    cnnConvolve(patchDim, nh, convImages, W, b, zcaWhite, meanPatch)

    nCorrect = 0
    for i = 1:1000
        fNum  = rand([1:nh;])
        imNum = rand([1:8;])
        imRow = rand([1:imgDim - patchDim + 1;])
        imCol = rand([1:imgDim - patchDim + 1;])    
 
        patch = convImages[imRow:imRow + patchDim - 1, 
                           imCol:imCol + patchDim - 1, :, imNum]
        patch = patch[:]            
        patch = patch - meanPatch
        patch = zcaWhite * patch
  
        features = feedForwardAutoencoder(optTheta, nv, nh, patch)
        diff = features[fNum, 1] - convolvedFeatures[fNum, imNum, imRow, imCol]
        if abs(diff) > 1e-9
            error("Convolved feature != autoencoder activation")
            println("Feature Number    $fNum")
            println("Image Number      $imNum")
            println("Image Row         $imRow")
            println("Image Column      $imCol")
            print("Convolved feature ")
            println("$(convolvedFeatures(fNum, imNum, imRow, imCol)))")
            println("Sparse AE feature $(features(fNum, 1)))")
        else
            nCorrect += 1
        end
    end
    println("$nCorrect/1000 convolved features match autoencoder activations.")
    convolvedFeatures
end

function testPooling(convolvedFeatures)
    pooledFeatures = cnnPool(poolDim, convolvedFeatures)
    testMatrix = reshape(1:64., 8, 8)
    expectedMatrix = [mean(mean(testMatrix[1:4, 1:4])) 
                      mean(mean(testMatrix[5:8, 1:4]))
                      mean(mean(testMatrix[1:4, 5:8]))
                      mean(mean(testMatrix[5:8, 5:8]))]
    testMatrix = reshape(testMatrix, 1, 1, 8, 8)
    pooledFeatures = cnnPool(4, testMatrix)[:]
    if isequal(pooledFeatures, expectedMatrix)
        println("Congratulations! Your pooling code passed the test.")
    else
        println("Pooling incorrect")
        println("Expected")
        println(expectedMatrix)
        println("Got")
        println(pooledFeatures)
    end
end

function main()
    if false
        displayColorNetwork((W*zcaWhite)')
        tcFeatures = testConvolution()
        testPooling(tcFeatures)
    end

end

main()
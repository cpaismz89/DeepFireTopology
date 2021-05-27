# Importations
from argparse import ArgumentParser
from tqdm import tqdm
import sys
from collections import Counter
sys.path.append('utils')

from configtf2 import *
import CPutils_tf2
import gradcam_utils_tf2

# Gradcam, GuidedBackprop, Guided GradCam for individual images
def GradCam_Outs(layers=[0,5,8,13,16,19], 
                 inputSize=(31,38),
                 channels=3,
                 save=False):
    # Loop
    for l in layers:
        print("Layer", l)
        gradcam, gb, guided_gradcam = gradcam_utils_tf2.compute_saliency(model, 
                                                                         guided_model, 
                                                                         imagePath, 
                                                                         layer_name='layer_' + str(l), 
                                                                         cls=-1,
                                                                         inputSize=inputSize,
                                                                         channels=channels,
                                                                         visualize=True, 
                                                                         save=save)

# Simpson index
def simpson_di(data):

    """ Given a hash { 'species': count } , returns the Simpson Diversity Index
    
    >>> simpson_di({'a': 10, 'b': 20, 'c': 30,})
    0.3888888888888889
    """

    def p(n, N):
        """ Relative abundance """
        if n is  0:
            return 0
        else:
            return float(n)/N

    N = sum(data.values())
    
    return sum(p(n, N)**2 for n in data.values() if n is not 0)

# Zonal statistics
def zonal_stats(TIFs, GCAMs, ImageIDs, verbose=False):
  
    def MaskFunction(percentage=0.3):
        ## INTERNAL UTILS
        # Components
        maxRow = 500
        maxCol = 500
        visited = np.zeros((maxCol, maxRow)) 

        # Function that return true if mat[row][col] 
        # is valid and hasn't been visited 
        def isSafe(M, row, col, c, n, l): 

            # If row and column are valid and element 
            # is matched and hasn't been visited then 
            # the cell is safe 
            return ((row >= 0 and row < n) and \
                    (col >= 0 and col < l) and \
                    (M[row][col] == c and not \
                     visited[row][col]))

        # Function for depth first search 
        def DFS(M, row, col, c, n, l, nn=4): 

            # These arrays are used to get row 
            # and column numbers of 4 neighbours 
            # of a given cell 
            if nn == 4:
                rowNbr = [ -1, 1, 0, 0 ] 
                colNbr = [ 0, 0, 1, -1 ]
            elif nn == 8:
                rowNbr = [-1, -1, -1,  0, 0,  1, 1, 1]; 
                colNbr = [-1,  0,  1, -1, 1, -1, 0, 1]; 
            else:
                raise ValueError

            # Mark this cell as visited 
            visited[row][col] = True; 

            # Recur for all connected neighbours 
            for k in range(nn) : 
                if (isSafe(M, row + rowNbr[k], col + colNbr[k], c, n, l)): 
                    DFS(M, row + rowNbr[k], col + colNbr[k], c, n, l) 

        # Function to return the number of 
        # connectewd components in the matrix 
        def connectedComponents(M, n, nn=4): 

            connectedComp = 0; 
            l = len(M[0]); 

            for i in range(n): 
                for j in range(l): 
                    if (not visited[i][j]): 
                        c = M[i][j]; 
                        DFS(M, i, j, c, n, l, nn)
                        connectedComp += 1 

            return connectedComp

        
        # Filtered masks 30%
        Mask = gcam.copy()
        Mask[Mask < percentage] = 0
        MaskR = cv2.resize(Mask, (tifFiles[idx].shape[1], tifFiles[idx].shape[0]))
        Aux = MaskR
        Aux[Aux != 0] = 1
        Aux[Aux == 0] = -1
        Filtered = (Aux * tifFiles[idx]).astype(np.int)
        FilteredMA = np.ma.masked_where(Filtered < 0, Filtered, copy=True) #Filtered[Filtered < 0] = np.nan
        if verbose:
            plt.imshow(FilteredMA, vmin=0, vmax=10)
            plt.show()
        
        # Area percentage
        features.append(len(Aux[Aux==1].flatten()) / len(Aux.flatten()))
        if verbose:
            print("Area %:", features[-1])
        
        # Components
        visited = np.zeros((maxCol, maxRow)) 
        I = Filtered.copy()
        I[I < 0] = 0
        n = I.shape[0]
        ncomponents8 = connectedComponents(I, n, nn=8)
        if verbose:
            print("NComp:", ncomponents8 - 1)   # minus the empty patch
        features.append(ncomponents8 - 1)
        
        # Indexes
        LCAux = Counter(Filtered[Filtered >= 0].flatten())
        if verbose:
            print("LCCounter:", LCAux)
        
        ## MN
        features.append(np.sum(FilteredMA) / ncomponents8)
        if verbose:
            print("MN:", features[-1])
        
        ## AMN
        features.append(0)
        if verbose:
            print("AMN:", features[-1])
        
        ## Simpson
        features.append(simpson_di(LCAux))
        if verbose:
            print("Simpson:", features[-1])
        
        # LCovers
        for k in range(0, 11):
            if k in LCAux.keys():
                features.append(LCAux[k])
            else:
                features.append(0)
        if verbose:
            print("LCovers:", features[-11:])
            
        # LCovers proportion
        TotalSum = np.sum([val for val in LCAux.values()])
        for k in range(0, 11):
            if k in LCAux.keys():
                features.append(LCAux[k] / TotalSum)
            else:
                features.append(0)
        if verbose:
            print("LCovers %:", features[-11:])
        
                
        # return calculated features
        return features
            
    # Containers to make DF   
    Columns = ['ID', 'area0', 'ncomp0', 'MN0', 'AMN0', 'SIMP0']
    Columns += ['lcV' + str(i) + '_0' for i in range(0, 11)]
    Columns += ['lcV' + str(i) + '_0%' for i in range(0, 11)]
    Columns +=  ['area30', 'ncomp30', 'MN30', 'AMN30', 'SIMP30']
    Columns += ['lcV' + str(i) + '_30' for i in range(0, 11)]
    Columns += ['lcV' + str(i) + '_30%' for i in range(0, 11)]
    Columns += ['area50', 'ncomp50', 'MN50', 'AMN50', 'SIMP50',] 
    Columns += ['lcV' + str(i) + '_50' for i in range(0, 11)]
    Columns += ['lcV' + str(i) + '_50%' for i in range(0, 11)]
    Columns += ['area75', 'ncomp75', 'MN75', 'AMN75', 'SIMP75']
    Columns += ['lcV' + str(i) + '_75' for i in range(0, 11)]
    Columns += ['lcV' + str(i) + '_75%' for i in range(0, 11)]
    StatsData = []
     
    # Stats loop
    for idx in tqdm(range(len(GCAMs))):
        # Features
        features = []
        features.append(ImageIDs[idx])
        
        # Get gcam mask (full)
        gcam = GCAMs[idx].copy()
        
        # Masks
        for alpha in [0.0, 0.3,0.5,0.75]:
            features = MaskFunction(percentage=alpha)
        
        StatsData.append(features)
        
    # DF
    DF = pd.DataFrame(StatsData, columns=Columns)
    
    # Return DF
    return DF


        
# Argument (set and layer)
parser = ArgumentParser()
parser.add_argument("--set",
                    help="Folder inside /data to process a set of images",
                    dest="set",
                    type=str,
                    default=None)                
parser.add_argument("--layer",
                    help="CONV layer to output from",
                    dest="layer",
                    type=int,
                    default=19)                
args = parser.parse_args()                                                                         


# Model Path
PRE_TRAINED = os.path.join('..', 'pretrained_models')
DATA_PATH = os.path.join('..', 'data', 'dataset')


# Model from scratch and then load weights
from nets_tf2.firenet import FireNet
dims = (38, 31, 3)
model = FireNet.build_model(width=dims[0], height=dims[1], depth=dims[2], classes=2)

# Load weights
model.load_weights(os.path.join(PRE_TRAINED, 'M2.h5'))
for i, layer in enumerate(model.layers):
    layer._name  = 'layer_' + str(i)
model.summary()

def model_constructor(PRE_TRAINED):
    # Model from scratch and then load weights
    from nets_tf2.firenet import FireNet
    dims = (38, 31, 3)
    model = FireNet.build_model(width=dims[0], height=dims[1], depth=dims[2], classes=2)

    # Load weights
    model.load_weights(os.path.join(PRE_TRAINED, 'M2.h5'))
    for i, layer in enumerate(model.layers):
        layer._name  = 'layer_' + str(i)

    return model    
    
# Model
H, W = 31, 38
model = gradcam_utils_tf2.build_model(model, model_constructor=model_constructor(PRE_TRAINED))
guided_model = gradcam_utils_tf2.build_guided_model(model)
print("Model loaded...")

# Load images
Set = args.set
BASE_PATH = os.path.join('..', 'data', Set)
imagePaths = sorted(list(CPutils_tf2.paths.list_images(BASE_PATH)))

# Shuffle
seed = 42
random.seed(seed)
random.shuffle(imagePaths)

# Sample
sampleSize = len(imagePaths)           # Whole directory
imagePaths = imagePaths[:sampleSize]

# Container
rawimages = []
imageNames = []

# Loop
for imagePath in tqdm(imagePaths):
    image = cv2.imread(imagePath, -1)
    image = cv2.resize(image, (38, 31))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    rawimages.append(image)
    imageNames.append(imagePath.split(os.path.sep)[-1][:-4])
    
# To array and process
rawimages = np.array(rawimages)
processedimages = rawimages/255.

print("Processedimages:", processedimages.shape)
print("Rawimages:", rawimages.shape)

# GradCAM
layer = 'layer_' + str(args.layer)
print("GradCAM on layer", args.layer)
gcam = gradcam_utils_tf2.generate_gradCAM(batch_size=32, 
                                          layer=layer,
                                          model=model,
                                          processedimages=processedimages, 
                                          rawimages=rawimages,
                                          save=True,
                                          savefile=Set + '_l' + str(args.layer) + '.lzma',
                                          showID=-1,
                                          title='Test', )

# Load geotif associated
TIF_PATH = os.path.join('..', 'data', 'geotif', 'dataset')
tifFiles = []
for image in imageNames:
    tif = image + '.tif'
    tif = cv2.imread(os.path.join(TIF_PATH,tif), -1)
    #tif = cv2.resize(tif, (38,31))
    tifFiles.append(tif)
#tifFiles = np.array(tifFiles)
#tifFiles.shape

# Zonal statistics function
DF = zonal_stats(TIFs=tifFiles, GCAMs=gcam, ImageIDs=imageNames, verbose=False)
DF.to_excel(Set + '_l' + str(args.layer) + '.xlsx', index=False)
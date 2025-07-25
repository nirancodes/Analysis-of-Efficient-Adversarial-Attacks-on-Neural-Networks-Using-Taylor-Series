import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import os
from PIL import Image
import matplotlib.pyplot as plt


# SETUP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# checks if a GPU is available because running models on them is much faster compared to CPU, but if not it defaults to using CPU
model =models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
# loads the pretrained MobileNetV2 model (has default weights)
model=model.to(device)
# model is moved to the selected GPU/CPU device
model.eval()
lossFunction=nn.CrossEntropyLoss() #loss function used by model to find how far predictions are from actual target outputs (labels)
#CrossEntropyLoss is for multi-class classification problems like this


# DATA - IMAGE PREPROCESSING AND DATASET LOADING, PREPARING RAW IMAGE FILES FOR MODEL TO CORRECTLY USE
transform=T.Compose([T.Resize((224,224)),
                     T.ToTensor()])
#a transformation pipeline applied to each dataset image: first resizes to the required input size for the model,
#then turns pixel values normalized to the range between 0 and 1 to prepare images for training
#model throws errors without this because of incorrect shape/ type
dataset=ImageFolder(root='./test_images',transform=transform)
#images are loaded from a folder and the transformation pipeline is applied to each one
load=DataLoader(dataset,batch_size=1,shuffle=False)
#an iterator loads one image at a time from the data set in the same order every time without shuffling


#ATTACK (with functions)
#an essential for any gradient based attack (ex. FGSM or TEAM)
def compute_gradient(image,model,lossFunction,actualOutput):
    image= image.clone()
    #copies the tensor used in PyTorch so that the initial one isn't overwritten when finding the gradient
    image = image.detach()
    #the new copy is detached from prior computations to allow reusing
    image.requires_grad_(True)
    #allows to track the gradient for PyTorch to calculate it in terms of the image pixels
    #the input image is considered like a variable to calculate how the model's prediction changes based on how each pixel is changed
    prediction =model(image)
    #the model is fed with this image, and the output is calculated
    loss = lossFunction(prediction,actualOutput)
    #followed by which the loss is found between prediction and expected actual target output (aka label)
    model.zero_grad()
    #gradients accumulated in the past are cleared
    loss.backward()
    #now, the loss's gradient is calculated with respect to the pixels and is returned in the next line
    return image.grad.data
#overall, finding each pixel's contribution to a wrong model


def fgsmAttack(image,epsilon, gradient):
    #follows the FGSM formulation - the gradient shows which direction increases the loss, so FGSM moves the image in that direction
     #x_adversarial = x + epsilon*sign(deltaJ(x,y))
     perturbation = epsilon*gradient.sign() #adversarial noise, produced by the product of the direction of steepest ascent in loss (1,-1,0) * epsilon
     imgAdv=image+perturbation #adds the perturbation to the image to generate a perturbed image
     imgAdv=torch.clamp(imgAdv,0,1) #for normalized images, pixel values are kept in the valid range (1=white, 0=black)
     return imgAdv


def teamAttack(image, epsilon, model, lossFunction, actualOutput, delta=1e-4):
    #follows the TEAM formululation , check Part 1's implementation section
    # the first order gradient of the loss is found in terms of the input
    grad1=compute_gradient(image.clone(), model,lossFunction, actualOutput)
    perturbed=(image.clone()+delta*grad1).detach().requires_grad_(True)
    #then, the image is perturbed a little in the gradient's direction, but at this new point, the gradient is calculated again
    gradagain=compute_gradient(perturbed,model,lossFunction,actualOutput)
    #the hessian vector product is approximation because using Hessian matrix is computationally very heavy
    hessianApprox=(gradagain-grad1) / delta
    #this is a finite difference approximation (of the 2nd derivative)
    perturbation=grad1+ (0.5)*epsilon *hessianApprox
    #this is the perturbation executed by the TEAM formula
    imgAdv=image+epsilon*perturbation
    imgAdv=torch.clamp(imgAdv,0,1)
    return imgAdv
     #for normalized images, pixel values are kept in the valid range (1=white, 0=black)


#VISUAL COMPARISONS
def compare(input_img,fgsm_adv,team_adv,epsilon,imgs):
    """Creates side-by-side comparison image"""
    # tensor is converted to a NumPy array format to create the comparisons
    #batch dimension of tensor is removed, reordered channels to display the image, tensor is moved to cpu and is made compatible to create comparisons on matplotlib


    def toNumPyArr(tensor):
        return tensor.squeeze().permute(1,2,0).cpu().numpy()
    inputNP=toNumPyArr(input_img)
    fgsmNP=toNumPyArr(fgsm_adv)
    teamNP=toNumPyArr(team_adv)
   
    # Now 3 plots are made side-by-side to represent original image with the perturbed images by FGSM and TEAM attacks
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5 ))
    fig.suptitle(f'Epsilon={epsilon:.2f}',fontsize=14) #title is also added to show current value of epsilon
   
    #each image of the 2 is shown without axis ticks and only with its respective title


    #ORIGINAL
    ax1.imshow(inputNP)
    ax1.axis('off')
    ax1.set_title('Original')
    #FGSM ATTACK
    ax2.imshow(fgsmNP)
    ax2.axis('off')
    ax2.set_title('FGSM Attack')
    #TEAM ATTACK
    ax3.imshow(teamNP)
    ax3.axis('off')
    ax3.set_title('TEAM Attack')
    #trial and error: avoids overlap, and saves the images, frees memory by closing
    plt.tight_layout()
    plt.savefig(f'comparisons/img_{imgs}_e_{epsilon:.2f}.png', bbox_inches='tight')
    plt.close()


# MAIN LOOP
os.makedirs('comparisons', exist_ok=True)
#folder/directory is made to store the comparison images
epsilons = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3] #control variable
#allowable max perturbations control the strength of the modification of the image, applied one by one


for imgs, (inputs,labels) in enumerate(load):
    if imgs>=2:  #Only process first 2 images for visual demonstration for speed
        break
#input image and its true actual output is moved to device to process
    input_tensor = inputs.to(device)
    actualOutput = labels.to(device)
   
    for e in epsilons: #for every element in epsilons, each value of e, loop to compute the loss's gradient with respect to the image
        grad=compute_gradient(input_tensor, model, lossFunction, actualOutput)
        #to show the attacks, two perturbed/adversarial versions of the image are generated
        fgsm_adv = fgsmAttack(input_tensor,e,grad)
        team_adv = teamAttack(input_tensor,e, model,lossFunction,actualOutput)
        #comparison images are saved
        compare(input_tensor,fgsm_adv,team_adv,e,imgs)

#GUI 만들기
#참고1: https://computer-choco.tistory.com/89
#참고2: https://076923.github.io/posts/Python-tkinter-1/
#모듈 호출
import tkinter
from tkinter import*
from tkinter import filedialog
#from PyQt5.QtGui import *
# import matplotlib
# matplotlib.use(“TkAgg”)
import matplotlib
matplotlib.use('Agg')

window = tkinter.Tk()
window.title("DeepArtClass")

# 초기 창 위치 설정
window.geometry("800x600+300+100")
window.configure(background='black')

# 화면크기 변경 불가
window.resizable(False, False)

wall = PhotoImage(file = "./design_source/deepartclass.gif")
wall_label = Label(image = wall)
wall_label.place(x=-2, y=0)


# train 파일과 test 파일을 불러오기 위한 버튼 액션
def style_fileopen():
    global style_url #check
    style_url = filedialog.askopenfilename() #자료형 str
    print(style_url)
    style_url_parse = style_url.split('/')
    print(style_url_parse)
    style_url = 'data/' + style_url_parse[-1]
    print("style_url에 담긴 str은: ", style_url)
    print(type(style_url))

def content_fileopen():
    global content_url# check
    content_url = filedialog.askopenfilename()
    print(content_url)
    content_url_parse = content_url.split('/')
    print(content_url_parse)
    content_url = 'data/'+ content_url_parse[-1]
    print("content_url에 담긴 str은: ", content_url)
    print(type(content_url))
    
def close():
    window.quit()
    window.destroy()

# 필요한 텍스트 및 버튼 구현
# 폰트 및 위치, 색상 변경
# 커서를 가져다 댔을 때의 버튼 효과 변경
# 버튼 위치 및 인터페이스 변경
lbl = Label(window, text="  Deep Art Class  ", width=23, font=("Lobster", 50), bg='#de6a2d', fg="#ffffff")
lbl.grid(column=0, row=0)
lbl = Label(window, text="", bg='#de6a2d')
lbl.grid(column=0, row=1)

lbl_1 = Label(window, text="[ STEP ① ]", font=("Righteous", 11), bg='black', fg='white')
lbl_1.grid(column=0, row=2, sticky='w', padx=50)
btn_1 = Button(window, text="화풍 선택하기", font=("함초롬돋움", 11), width=11, anchor='center', command=style_fileopen, bg="white", fg='black', overrelief='ridge', relief='raised', borderwidth='3', padx=5, pady=1)
btn_1.grid(column=0, row=3, sticky='w', padx=30, pady=10)

lbl_2 = Label(window, text="[ STEP ② ]", font=("Righteous", 11), bg='black', fg='white')
lbl_2.grid(column=0, row=2)
btn_2 = Button(window, text="사진 선택하기", font=("함초롬돋움", 11), width=11, anchor='center', command=content_fileopen, bg='white', fg='black', overrelief='ridge', relief='raised', borderwidth='3', padx=5, pady=1)
btn_2.grid(column=0, row=3, padx=10, pady=10)

lbl_3 = Label(window, text="[ STEP ③ ]", font=("Righteous", 11), bg='black', fg='white')
lbl_3.grid(column=0, row=2, sticky='e', padx=60)
btn_3 = Button(window, text="변환 START", font=("함초롬돋움", 11), width=11, anchor='center', command=close, bg='white', fg='black', overrelief='ridge', relief='raised', borderwidth='3', padx=5, pady=1)
btn_3.grid(column=0, row=3, sticky='e', padx=40, pady=10)

# 윈도우 창 생성
window.mainloop()

print('success')

#from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import os
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image #size조절
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

style_img = image_loader(style_url)
content_img = image_loader(content_url)
print("style_size: ", style_img.shape)
print("content_size: ", content_img.shape)

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"


unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
    
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
    
    
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
    
cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
    
# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

input_img = content_img.clone()
# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)

# add the original input image to the figure:
plt.figure()
imshow(input_img, title='Input Image')

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()

#이미지 저장
from torchvision.utils import save_image
import datetime
now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
print(nowDatetime)  # 2015-04-19 12:11:32

stime = nowDatetime
stime_change = stime.replace(':', '_', 2)

result_path = './result/result' + stime_change + '.jpg'

print(result_path)

save_image(output, result_path)
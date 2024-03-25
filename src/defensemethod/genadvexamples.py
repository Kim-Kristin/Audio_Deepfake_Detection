import torch
import torchattacks
from torch.autograd import Variable
from tqdm import tqdm
import copy

def gen_adv(input, label , model):
    model = model
    #model_point = torch.load(resnet_path)
    #model.load_state_dict(model_point["state_dict"])
    #normal_data, adv_data, label= None, None, None

    model_cp = copy.deepcopy(model)
    for p in model_cp.parameters():
        p.requires_grad = False
    model_cp.eval()
    attack = torchattacks.Pixle(model_cp, x_dimensions=(0.1, 0.2), restarts=10, max_iterations=50)
    adv_input = attack(input.cpu(), label.cpu())
    return adv_input

    """print("Generating adversarial examples...")
    for i, (input, label) in tqdm(enumerate(trainloader,0)):
        input, label = Variable(input.to(device)), Variable(label.to(device))
        adv_input = attack(input.cpu(), label.cpu())
        input, adv_input, label = input.data, adv_input.data, label.data

        #if normal_data is None:
        normal_data, adv_data, label = input, adv_input, label
        #else:
        #normal_data = torch.cat((normal_data, input))
        #adv_data = torch.cat((adv_data, adv_input))
        #label = torch.cat((label, label))
    # save normal and adv data for Adversarial Training
    torch.save({"normal": normal_data, "adv": adv_data, "label_normal": label}, filename)"""

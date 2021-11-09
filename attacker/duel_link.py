import torch
import torch.nn as nn
import torch.nn.functional as F

class LinfPGDTargetAttack(nn.Module):
    """Projected Gradient Decent(PGD) attack.
    Can be used to adversarial training.
    """
    def __init__(self, model, num_class=10, epsilon=8/255, step=2/255, iterations=20, criterion=None, random_start=True, targeted=True):
        super(LinfPGDTargetAttack, self).__init__()
        # Arguments of PGD
        self.device = next(model.parameters()).device
        self.model = model
        self.num_class = num_class

        self.epsilon = epsilon
        self.step = step
        self.iterations = iterations
        self.random_start = random_start
        self.targeted = targeted

        self.criterion = criterion
        if self.criterion is None:
            self.criterion = lambda model, input, target: nn.functional.cross_entropy(model(input), target)

        # Model status
        self.training = self.model.training

    def project(self, perturbation):
        # Clamp the perturbation to epsilon Lp ball.
        return torch.clamp(perturbation, -self.epsilon, self.epsilon)

    def compute_perturbation(self, adv_x, x):
        # Project the perturbation to Lp ball
        perturbation = self.project(adv_x - x)
        # Clamp the adversarial image to a legal 'image'
        perturbation = torch.clamp(x+perturbation, 0., 1.) - x

        return perturbation

    def onestep(self, x, perturbation, target):
        # Running one step for 
        adv_x = x + perturbation
        adv_x.requires_grad = True
        
        atk_loss = self.criterion(self.model, adv_x, target)

        self.model.zero_grad()
        atk_loss.backward()
        grad = adv_x.grad
        # Essential: delete the computation graph to save GPU ram
        adv_x.requires_grad = False

        adv_x = adv_x.detach() + self.step * torch.sign(grad)
        perturbation = self.compute_perturbation(adv_x, x)

        return perturbation
    
    def fakerstep(self, x, perturbation, target, faker, times=1):
        # Running one step for 
        adv_x = x + perturbation
        adv_x.requires_grad = True
        
        pred = self.model(adv_x)
        
        ## L2 Loss PGD
        target_onehot = F.one_hot(target, num_classes=self.num_class).float()
        faker_onehot = F.one_hot(faker, num_classes=self.num_class).float()
        atk_loss = (torch.mean(torch.norm(pred-target_onehot, dim=1), dim=0)-torch.mean(torch.norm(pred-faker_onehot, dim=1), dim=0))/2
        
        # atk_loss = (nn.functional.cross_entropy(pred, target) - nn.functional.cross_entropy(pred, faker))/2
        # atk_loss = self.criterion(self.model, adv_x, target) - self.criterion(self.model, adv_x, faker)

        self.model.zero_grad()
        atk_loss.backward()
        grad = adv_x.grad
        # Essential: delete the computation graph to save GPU ram
        adv_x.requires_grad = False

        adv_x = adv_x.detach() + times * self.step * torch.sign(grad)
        perturbation = self.compute_perturbation(adv_x, x)

        return perturbation
    
    def _model_freeze(self):
        for param in self.model.parameters():
            param.requires_grad=False

    def _model_unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad=True

    def random_perturbation(self, x):
        perturbation = torch.rand_like(x).to(device=self.device)
        perturbation = self.compute_perturbation(x+perturbation, x)

        return perturbation

    def attack(self, x, target):
        x = x.to(self.device)
        target = target.to(self.device)

        self.training = self.model.training
        self.model.eval()
        self._model_freeze()

        perturbation = torch.zeros_like(x).to(self.device)
        if self.random_start:
            perturbation = self.random_perturbation(x)

        with torch.enable_grad():
            for i in range(self.iterations):
                perturbation = self.onestep(x, perturbation, target)

        self._model_unfreeze()
        if self.training:
            self.model.train()

        return x + perturbation
    
    def target_attack(self, x, target):
        faker = torch.randint(low=0, high=self.num_class-1, size=target.shape, device=self.device)
        # Ensure target != true_target
        faker += (faker >= target).int()

        x = x.to(self.device)
        target = target.to(self.device)

        self.training = self.model.training
        self.model.eval()
        self._model_freeze()

        perturbation = torch.zeros_like(x).to(self.device)
        if self.random_start:
            perturbation = self.random_perturbation(x)

        with torch.enable_grad():
            perturbation = self.fakerstep(x, perturbation, target, faker, 8)
            perturbation = self.fakerstep(x, perturbation, target, faker, 4)
            perturbation = self.fakerstep(x, perturbation, target, faker, 2)
            perturbation = self.fakerstep(x, perturbation, target, faker, 2)
            for i in range(self.iterations):
                perturbation = self.fakerstep(x, perturbation, target, faker)

        self._model_unfreeze()
        if self.training:
            self.model.train()

        return x + perturbation
    
    def perturb(self, x, target):
        if self.targeted:
            return self.target_attack(x, target)
        else:
            return self.attack(x, target)

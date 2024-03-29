import torch
import torch.nn as nn

class LinfPGD(nn.Module):
    """Projected Gradient Decent(PGD) attack.
    Can be used to adversarial training.
    """
    def __init__(self, model, epsilon=8/255, step=2/255, iterations=20, criterion=None, random_start=True, targeted=False):
        super(LinfPGD, self).__init__()
        # Arguments of PGD
        self.device = next(model.parameters()).device

        self.model = model
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

        if self.targeted:
            adv_x = adv_x.detach() - self.step * torch.sign(grad)
        else:
            adv_x = adv_x.detach() + self.step * torch.sign(grad)
        perturbation = self.compute_perturbation(adv_x, x)

        return perturbation
    

    def _model_freeze(self):
        for param in self.model.parameters():
            param.requires_grad=False

    def _model_unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad=True

    def random_perturbation(self, x):
        return self.epsilon * (torch.rand_like(x)*2-1).to(device=self.device)
        # perturbation = torch.rand_like(x).to(device=self.device)
        # perturbation = self.compute_perturbation(x+perturbation, x)

        # return perturbation

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
    
    def perturb(self, x, target):
        return self.attack(x, target)


class L2PGD(nn.Module):
    """Projected Gradient Decent(PGD) attack.
    Can be used to adversarial training.
    """
    def __init__(self, model, epsilon=5, step=1, iterations=20, criterion=None, random_start=True, targeted=False):
        super(L2PGD, self).__init__()
        # Arguments of PGD
        self.device = next(model.parameters()).device

        self.model = model
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
        return perturbation.renorm(p=2, dim=0, maxnorm=self.epsilon)

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
        g_norm = torch.norm(grad.view(x.shape[0], -1), p=2, dim=1).view(-1, *([1]*(len(x.shape)-1)))
        grad = grad / (g_norm + 1e-10)
        # Essential: delete the computation graph to save GPU ram
        adv_x.requires_grad = False

        if self.targeted:
            adv_x = adv_x.detach() - self.step * grad
        else:
            adv_x = adv_x.detach() + self.step * grad
        perturbation = self.compute_perturbation(adv_x, x)

        return perturbation
    
    def double_onestep(self, x, perturbation, target_0, target_1):
        # Running one step for 
        adv_x = x + perturbation
        adv_x.requires_grad = True
        
        # atk_loss = self.criterion(self.model, adv_x, target_0) + self.criterion(self.model, adv_x, target_1)
        this_output = self.model(adv_x)
        atk_loss = nn.CrossEntropyLoss()(this_output, target_0) + nn.CrossEntropyLoss()(this_output, target_1)

        self.model.zero_grad()
        atk_loss.backward()
        grad = adv_x.grad
        g_norm = torch.norm(grad.view(x.shape[0], -1), p=2, dim=1).view(-1, *([1]*(len(x.shape)-1)))
        grad = grad / (g_norm + 1e-10)
        # Essential: delete the computation graph to save GPU ram
        adv_x.requires_grad = False

        if self.targeted:
            adv_x = adv_x.detach() - self.step * grad
        else:
            adv_x = adv_x.detach() + self.step * grad
        perturbation = self.compute_perturbation(adv_x, x)

        return perturbation

    def mesa_onestep(self, x, perturbation, target):
        # Running one step for 
        adv_x = x + perturbation
        adv_x.requires_grad = True
        
        # atk_loss = self.criterion(self.model, adv_x, target_0) + self.criterion(self.model, adv_x, target_1)
        this_output = self.model(adv_x)
        atk_loss = nn.MSELoss()(this_output, target)

        self.model.zero_grad()
        atk_loss.backward()
        grad = adv_x.grad
        g_norm = torch.norm(grad.view(x.shape[0], -1), p=2, dim=1).view(-1, *([1]*(len(x.shape)-1)))
        grad = grad / (g_norm + 1e-10)
        # Essential: delete the computation graph to save GPU ram
        adv_x.requires_grad = False

        if self.targeted:
            adv_x = adv_x.detach() - self.step * grad
        else:
            adv_x = adv_x.detach() + self.step * grad
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

    def double_attack(self, x, target_0, target_1):
        x = x.to(self.device)
        target_0 = target_0.to(self.device)
        target_1 = target_1.to(self.device)

        self.training = self.model.training
        self.model.eval()
        self._model_freeze()
        
        perturbation = torch.zeros_like(x).to(self.device)
        if self.random_start:
            perturbation = self.random_perturbation(x)
        
        with torch.enable_grad():
            for i in range(self.iterations):
                perturbation = self.double_onestep(x, perturbation, target_0, target_1)

        self._model_unfreeze()
        if self.training:
            self.model.train()

        return x + perturbation
    
    def mesa_attack(self, x, target):
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
                perturbation = self.mesa_onestep(x, perturbation, target)

        self._model_unfreeze()
        if self.training:
            self.model.train()

        return x + perturbation


class LinfPGD_lipz(nn.Module):
    """Projected Gradient Decent(PGD) attack.
    Can be used to adversarial training.
    """
    def __init__(self, model, epsilon=8/255, step=2/255, iterations=20, criterion=None, random_start=True, targeted=False):
        super(LinfPGD_lipz, self).__init__()
        # Arguments of PGD
        self.device = next(model.parameters()).device

        self.model = model
        self.epsilon = epsilon
        self.step = step
        self.iterations = iterations
        self.random_start = random_start
        self.targeted = targeted

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        # if self.criterion is None:
        #     self.criterion = lambda model, input, target: nn.functional.cross_entropy(model(input), target)

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

    def anorm(self, perturb):
        perabs = torch.abs(perturb)
        return torch.sum(torch.sum(torch.sum(perabs, dim=1), dim=1), dim=1)
        return torch.norm(torch.norm(torch.norm(perturb, dim=1), dim=1), dim=1)

    def onestep(self, x, perturbation, target):
        # Running one step for 
        adv_x = x + perturbation
        adv_x.requires_grad = True
        
        atk_loss = self.criterion(self.model(adv_x), target) / (1e-8 + self.anorm(perturbation.detach()))
        atk_loss = torch.sum(atk_loss)

        self.model.zero_grad()
        atk_loss.backward()
        grad = adv_x.grad
        # Essential: delete the computation graph to save GPU ram
        adv_x.requires_grad = False

        if self.targeted:
            adv_x = adv_x.detach() - self.step * torch.sign(grad)
        else:
            adv_x = adv_x.detach() + self.step * torch.sign(grad)
        perturbation = self.compute_perturbation(adv_x, x)

        return perturbation
    

    def _model_freeze(self):
        for param in self.model.parameters():
            param.requires_grad=False

    def _model_unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad=True

    def random_perturbation(self, x):
        return self.epsilon * (torch.rand_like(x)*2-1).to(device=self.device)
        # perturbation = torch.rand_like(x).to(device=self.device)
        # perturbation = self.compute_perturbation(x+perturbation, x)

        # return perturbation

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
    
    def perturb(self, x, target):
        return self.attack(x, target)

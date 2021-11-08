def target_attack(adversary, inputs, true_target, num_class, device, gamma=0.):
    target = torch.randint(low=0, high=num_class-1, size=true_target.shape, device=device)
    # Ensure target != true_target
    target += (target >= true_target).int()
    adversary.targeted = True

    ## Reach Plain
    adv_inputs = adversary.perturb(inputs, target).detach()
    
    ## Rand Gamma
    rand_lambda = tower_Rand((true_target.shape[0],1,1,1), device=device)
    ret_inputs = rand_lambda*adv_inputs + (1-rand_lambda)*inputs
    ret_target = F.one_hot(true_target, num_class).float().to(device)
    
    return ret_inputs, ret_target

def multar_adv_step(self, progress):
        self.model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=self.desc("Adv train", progress))
        for inputs_0, labels_0 in self.train_loader:

            batchSize = labels_0.shape[0]
            inputs_0 = inputs_0.to(self.device)
            labels_0 = labels_0.to(self.device)

            inputs_1, _ = target_attack(self.attacker, inputs_0, labels_0, self.num_class, self.device, self.gamma)
            inputs_2, _ = target_attack(self.attacker, inputs_0, labels_0, self.num_class, self.device, self.gamma)

            # Create inputs & labels
            rand_vector = rain_Rand((batchSize, 1), device=self.device)
            inputs = inputs_1 * rand_vector.unsqueeze(2).unsqueeze(3) + inputs_2 * (1-rand_vector.unsqueeze(2).unsqueeze(3))
            labels = F.one_hot(labels_0, self.num_class).float().to(self.device)
            del inputs_0, inputs_1, inputs_2, labels_0, rand_vector
            
            # print(inputs.requires_grad, inputs.shape)
            outputs = self.model(inputs)
            loss = soft_loss(outputs, labels)
            pbar.set_postfix_str("Loss {:.6f}".format(loss.item()))
            loss_meter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            pbar.update(1)
        pbar.close()

        return loss_meter.report()

def multar_train(self, writer, adv=True):
        
        ## Add a Writer
        self.add_writer(writer, 0)
        for epoch_idx in range(self.epochs):
            if adv:
                avg_loss = self.multar_adv_step("{}/{}".format(epoch_idx, self.epochs))
            else:
                avg_loss = self.clean_step("{}/{}".format(epoch_idx, self.epochs))
            avg_loss = collect(avg_loss, self.device)
            if torch.distributed.get_rank() == 0:
                if adv:
                    tqdm.write("Adv training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
                else:
                    tqdm.write("Clean training procedure {} (total {}), Loss avg. {:.6f}".format(epoch_idx, self.epochs, avg_loss))
            
            ## Add a Writer
            if epoch_idx % self.eval_interval == (self.eval_interval-1):
                self.add_writer(writer, epoch_idx+1)
            
            if self.scheduler is not None:
                self.scheduler.step()

        tqdm.write("Finish training on rank {}!".format(torch.distributed.get_rank()))
class Scheduler_2():
    def __init__(self, scheduler1, scheduler2):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
    def step(self):
        self.scheduler1.step()
        self.scheduler2.step()

class Scheduler_3():
    def __init__(self, scheduler1, scheduler2, scheduler3):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.scheduler3 = scheduler3
    def step(self):
        self.scheduler1.step()
        self.scheduler2.step()
        self.scheduler3.step()

class Scheduler_List():
    def __init__(self, mylist):
        self.mylist = mylist

    def step(self):
        for scheduler in self.mylist:
            scheduler.step()

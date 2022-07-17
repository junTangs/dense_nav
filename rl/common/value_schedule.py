from abc import ABCMeta,abstractmethod


class ValueSchedule(metaclass= ABCMeta):
    def __init__(self):
        super(ValueSchedule, self).__init__()
        self.step = 0
        return


    def reset(self):
        self.step = 0

    @abstractmethod
    def value(self):
        raise NotImplementedError



class LinearSchedule(ValueSchedule):
    def __init__(self,start,end,iter_nums = 1000):
        super(LinearSchedule, self).__init__()
        self.start = start
        self.end = end
        self.iter_nums = iter_nums

        self._d = (self.end - self.start)/(self.iter_nums)


    def value(self):
        val =  self.start+self.step*self._d if self.step <= self.iter_nums else self.end
        self.step+=1
        return val


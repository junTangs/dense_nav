import json
from matplotlib import pyplot as plt
import random as rnd

class TraceRecorder:
    
    color = ['r','g','b','y','c','m','k']
    def __init__(self,name,path,length = 10,width = 10) -> None:
        # {id:[data0,data1,...]}

        self.trace = {'length':length,'width':width,'trace':{}}
        self.name = name
        self.path = path
        self.length = length
        self.width = width
        
    def add_trace(self,id,x,y,t,wp = True):
        if id not in self.trace['trace']:
            self.trace['trace'][id] = []
        self.trace['trace'][id].append([x,y,t,wp])

    def save_trace(self):
        with open(self.path+self.name+'.json','w') as f:
            json.dump(self.trace,f)
    
    def load_trace(self):
        with open(self.path+self.name+'.json','r') as f:
            self.trace = json.load(f)
        self.length = self.trace['length']
        self.width = self.trace['width']

    def trace_sort(self):
        for id in self.trace['trace']:
            self.trace['trace'][id].sort(key = lambda x:x[2])
    
    def draw_trace(self,save = False):
        self.trace_sort()
        
    
        alpha = 1/len(self.trace['trace'])
        color = rnd.choice(self.color)
        
        # plot config
        plt.figure(figsize=(5, 5))
        plt.xlim(0,self.length)
        plt.ylim(0,self.width)
        
        # plot trace
        for id in self.trace['trace']:
            x = [i[0] for i in self.trace['trace'][id]]
            y = [i[1] for i in self.trace['trace'][id]]
            plt.plot(x,y,color,alpha = alpha)
            plt.scatter(x[0],y[0],color = color,alpha = alpha,marker = '^')
            plt.scatter(x[-1],y[-1],color = color,alpha = alpha,marker = '*')

        if save == True:
            plt.savefig(self.path+self.name+'.png',dpi = 300)
        plt.show()
        plt.close()
    
if __name__ == "__main__":
    import math
    tr = TraceRecorder('test','./')
    
    for i in range(1000):
        x = math.cos(i)+(i//10)
        y = math.sin(i)+(i//10)
        tr.add_trace(1,x,y,i)

    
    tr.save_trace()
    tr2 = TraceRecorder('test','./')
    tr2.load_trace()

    print(tr.trace)
    tr.trace_sort()
    print(tr.trace)
    print(tr2.trace)

    tr.draw_trace(save = True)
    tr2.draw_trace()
    
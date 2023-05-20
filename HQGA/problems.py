import math
import random
import itertools

from HQGA import discretization as dis

class Problem():
    """Class that defines the problem to be solved"""
    def __init__(self, name, dimension=3, num_bit_code=5, maxp=False):
        self.name = name
        self.dim = dimension
        self.num_bit_code = num_bit_code
        self.maxp = maxp

    def convert(self, chr):
        pass

    def computeFitness(self, phenotype):
        pass

    def evaluate(self, chr):
        phenotype = self.convert(chr)
        return self.computeFitness(phenotype)

    def isMaxProblem(self):
        return self.maxp

    def setMaxProblem(self, boolean):
        self.maxp = boolean


class BinaryProblem(Problem):
    """Class that defines the binary problem to be solved"""
    def __init__(self, name, dimension, maxp=False):
        super().__init__(name, dimension, 1, maxp)


    def convert(self, chr):
        return chr

    def getOptimum(self):
        fitnesses=[]

        list_all_values = []
        for i in range(self.dim):
            list_all_values.append(['0','1'])
        all_space = list(itertools.product(*list_all_values))

        for chr in all_space:
            fitnesses.append(self.computeFitness(''.join(chr)))

        if self.isMaxProblem():
            optimal_fit=max(fitnesses)
        else:
            optimal_fit=min(fitnesses)
        list_index=[]

        for ind in range(len(fitnesses)):
            if fitnesses[ind]==optimal_fit:
                list_index.append(ind)

        optimal_values = []
        for ind in list_index:
            optimal_values.append(all_space[ind])
        return optimal_fit, optimal_values

    def getWholeSpace(self):
        fitnesses = []
        sols= []
        list_all_values = []
        for i in range(self.dim):
            list_all_values.append(['0','1'])
        all_sols = list(itertools.product(*list_all_values))
        for chr in all_sols:
            fitnesses.append(self.computeFitness(chr))
            sols.append(''.join(chr))
        return sols, fitnesses


class RealProblem(Problem):
    """Class that defines the real problem to be solved"""
    def __init__(self, name, dimension, num_bit_code, lower_bounds, upper_bounds):
        super().__init__(name, dimension, num_bit_code)
        self.lower_bounds= lower_bounds
        self.upper_bounds= upper_bounds

    def convertToReal(self, chr):
        #return value as the left point of the interval
        return dis.convertFromBinToFloat(chr, self.lower_bounds, self.upper_bounds, self.num_bit_code, self.dim)

    def convert(self, chr):
        return self.convertToReal(chr)


    def getOptimum(self):
        fitnesses=[]

        list_all_values = []
        for i in range(self.dim):
            list_all_values.append(dis.getAllValues(self.lower_bounds[i], self.upper_bounds[i], self.num_bit_code))
        all_space = list(itertools.product(*list_all_values))
        for chr in all_space:
            fitnesses.append(self.computeFitness(chr))
        if self.isMaxProblem():
            optimal_fit=max(fitnesses)
        else:
            optimal_fit=min(fitnesses)
        list_index=[]
        for ind in range(len(fitnesses)):
            if fitnesses[ind]==optimal_fit:
                list_index.append(ind)
        optimal_values = []
        gray_optimal_sols =[]
        for ind in list_index:
            optimal_values.append(all_space[ind])
            l = []
            for i in range(self.dim):
                l.append(
                    dis.convertFromFloatToBin(all_space[ind][i], self.lower_bounds[i], self.upper_bounds[i], self.num_bit_code))
            gray_optimal_sols.append(l)
        return optimal_fit, optimal_values, gray_optimal_sols

    def getWholeSpace(self):
        fitnesses=[]
        gray_sols=[]
        list_all_values =[]
        for i in range(self.dim):
            list_all_values.append(dis.getAllValues(self.lower_bounds[i], self.upper_bounds[i], self.num_bit_code))
        sols = list(itertools.product(*list_all_values))
        for chr in sols:
            fitnesses.append(self.computeFitness(chr))
            l = []
            for i in range(self.dim):
                l.append(dis.convertFromFloatToBin(chr[i], self.lower_bounds[i], self.upper_bounds[i], self.num_bit_code))
            gray_sols.append(l)
        return sols, gray_sols, fitnesses





class RastriginProblem(RealProblem):
    def __init__(self, dim = 2, num_bit_code = 4, lower_bounds = [-5.12,-5.12], upper_bounds = [5.12,5.12], A=10):
        super().__init__("Rastrigin", dim, num_bit_code, lower_bounds, upper_bounds)
        self.A=A

    def computeFitness(self, chr_real):
        #compute fitness value
        res=self.A*self.dim+sum([x**2-10*math.cos(2*math.pi*x) for x in chr_real])
        #print("fitness value ",res)
        return res

class SphereProblem(RealProblem):
    def __init__(self,dim = 2, num_bit_code = 4, lower_bounds = [-5.12,-5.12], upper_bounds = [5.12,5.12]):
        super().__init__("Sphere", dim, num_bit_code, lower_bounds, upper_bounds)

    def computeFitness(self, chr_real):
        res = sum([i ** 2 for i in chr_real])
        #print("fitness value ", res)
        return res



class OneMaxProblem(BinaryProblem):
    def __init__(self, dim=5):
        super().__init__("OneMax", dim, maxp=True)


    def computeFitness(self, chr):
        return chr.count("1")

    def getOptimumFitness(self):
        return self.dim

class BealeProblem(RealProblem):
    def __init__(self,dim = 2, num_bit_code = 4, lower_bounds = [-4.5,-4.5], upper_bounds = [4.5,4.5]):
        super().__init__("Beale", dim, num_bit_code, lower_bounds, upper_bounds)

    def computeFitness(self, chr_real):
        x1=chr_real[0]
        x2=chr_real[1]
        #compute fitness value
        res=(1.5 -x1+x1*x2)**2 + (2.25 -x1 + x1*x2**2)**2 + (2.625 -x1 + x1*x2**3)**2
        #print("fitness value ",res)
        return res


class RosenbrockProblem(RealProblem):
    def __init__(self, dim = 2, num_bit_code = 4, lower_bounds = [-2.048,-2.048], upper_bounds = [2.048, 2.048]):
        super().__init__("Rosenbrock", dim, num_bit_code, lower_bounds, upper_bounds)


    def computeFitness(self, chr_real):
        #compute fitness value
        x1 = chr_real[0]
        x2 = chr_real[1]
        res=100*(x2 - x1**2)**2+(1-x1)**2
        #print("fitness value ",res)
        return res

class StepProblem(RealProblem):
    def __init__(self, dim = 2, num_bit_code = 4, lower_bounds = [-5.12,-5.12], upper_bounds = [5.12,5.12]):
        super().__init__("Step", dim, num_bit_code, lower_bounds, upper_bounds)

    def computeFitness(self, chr_real):
        res = sum([int(i) for i in chr_real])
        #print("fitness value ", res)
        return res

class QuarticProblem(RealProblem):
    def __init__(self, dim = 2, num_bit_code = 4, lower_bounds = [-1.28,-1.28], upper_bounds = [1.28, 1.28]):
        super().__init__("Quartic", dim, num_bit_code, lower_bounds, upper_bounds)

    def computeFitness(self, chr_real):
        res = sum([(i+1)*chr_real[i]**4+random.gauss(0,1) for i in range(len(chr_real))])
        #print("fitness value ", res)
        return res

class SchwefelProblem(RealProblem):
    def __init__(self,dim = 2, num_bit_code = 4, lower_bounds = [-500,-500], upper_bounds = [500,500], V = 418.9829):
        super().__init__("Schwefel", dim, num_bit_code, lower_bounds, upper_bounds)
        self.V = V

    def computeFitness(self, chr_real):
        res = self.dim*self.V + sum([-x*math.sin(math.sqrt(abs(x))) for x in chr_real])
        #print("fitness value ", res)
        return res


class GriewangkProblem(RealProblem):
    def __init__(self, dim = 2,num_bit_code = 4, lower_bounds = [-600,-600], upper_bounds = [600,600]):
        super().__init__("Griewangk", dim, num_bit_code, lower_bounds, upper_bounds)

    def computeFitness(self, chr_real):
        res =1 + sum([x**2/4000 for x in chr_real])- math.prod([math.cos(chr_real[i]/math.sqrt(i+1)) for i in range(len(chr_real))])
        #print("fitness value ", res)
        return res

class AckleyProblem(RealProblem):
    def __init__(self, dim = 2,num_bit_code = 4, lower_bounds = [-32.768, -32.768], upper_bounds = [32.768, 32.768], a=20, b=0.2, c=2*math.pi):
        super().__init__("Ackley", dim, num_bit_code, lower_bounds, upper_bounds)
        self.a=a
        self.b=b
        self.c=c

    def computeFitness(self, chr_real):
        firstSum=sum([x**2 for x in chr_real])
        secondSum=sum([math.cos(self.c*x) for x in chr_real])
        res = -self.a*math.exp(-self.b*math.sqrt(firstSum/self.dim)) - math.exp(secondSum/self.dim) + self.a + math.exp(1)
        #print("fitness value ", res)
        return res

class BohachevskyProblem(RealProblem):
    def __init__(self, num_bit_code = 4, lower_bounds = [-100,-100], upper_bounds = [100,100]):
        super().__init__("Bohachevsky", 2, num_bit_code, lower_bounds, upper_bounds)


    def computeFitness(self, chr_real):
        x1=chr_real[0]
        x2=chr_real[1]
        res = x1**2 + 2*x2**2 -0.3*math.cos(3*math.pi*x1) -0.4*math.cos(4*math.pi*x2)+0.7
        #print("fitness value ", res)
        return res

class BirdProblem(RealProblem):
    def __init__(self, num_bit_code = 4, lower_bounds = [-2*math.pi,-2*math.pi], upper_bounds = [2*math.pi,2*math.pi]):
        super().__init__("Bird", 2, num_bit_code, lower_bounds, upper_bounds)


    def computeFitness(self, chr_real):
        #print("cromosome ", chr_real)
        x=chr_real[0]
        y=chr_real[1]
        res = math.sin(x)*math.exp((1-math.cos(y))**2) + math.cos(y)*math.exp((1-math.sin(x))**2) + (x-y)**2
        #print("fitness value ", res)
        return res

class BoothProblem(RealProblem):
    def __init__(self, num_bit_code = 4, lower_bounds = [-10,-10], upper_bounds = [10,10]):
        super().__init__("Booth", 2, num_bit_code, lower_bounds, upper_bounds)


    def computeFitness(self, chr_real):
        x=chr_real[0]
        y=chr_real[1]
        res = (x+2*y-7)**2 + (2*x + y - 5)**2
        #print("fitness value ", res)
        return res

class GoldeinsteinProblem(RealProblem):
    def __init__(self, num_bit_code = 4, lower_bounds = [-2,-2], upper_bounds = [2,2]):
        super().__init__("Goldeinstein", 2, num_bit_code, lower_bounds, upper_bounds)


    def computeFitness(self, chr_real):
        X=chr_real[0]
        Y=chr_real[1]
        res = (1 + ((X + Y + 1)**2) * (19 - (14 * X) + (3 * (X**2)) - 14*Y + (6 *X*Y) + (3 * (Y**2)))) *(30 + ((2 * X - 3 * Y)**2) * (18 - 32 * X + 12 * (X**2) + 48 * Y - (36* X*Y) + (27 * (Y**2))) )
        #print("fitness value ", res)
        return res

class HolderTableProblem(RealProblem):
    def __init__(self, num_bit_code = 4, lower_bounds = [-10,-10], upper_bounds = [10,10]):
        super().__init__("HolderTable", 2, num_bit_code, lower_bounds, upper_bounds)


    def computeFitness(self, chr_real):
        X=chr_real[0]
        Y=chr_real[1]
        res = -(abs(math.sin(X)*math.cos(Y)*math.exp(abs(1-(math.sqrt(X**2+Y**2)/math.pi)))))
        #print("fitness value ", res)
        return res

class GramacyProblem(RealProblem):
    def __init__(self, num_bit_code = 8, lower_bounds = [0.5], upper_bounds = [2.5]):
        super().__init__("Gramacy", 1, num_bit_code, lower_bounds, upper_bounds)


    def computeFitness(self, chr_real):
        X=chr_real[0]
        res = (math.sin(10*math.pi*X)/2*X) +(X-1)**4
        #print("fitness value ", res)
        return res

class ForresterProblem(RealProblem):
    def __init__(self, num_bit_code = 8, lower_bounds = [0], upper_bounds = [1]):
        super().__init__("Forrester", 1, num_bit_code, lower_bounds, upper_bounds)


    def computeFitness(self, chr_real):
        x=chr_real[0]
        res = (6*x-2)**2 * math.sin(12*x - 4)
        #print("fitness value ", res)
        return res


class PeriodicProblem(RealProblem):
    def __init__(self, dim = 2, num_bit_code = 4, lower_bounds = [-2, -2], upper_bounds = [2,2]):
        super().__init__("Periodic", dim, num_bit_code, lower_bounds, upper_bounds)


    def computeFitness(self, chr_real):
        firstSum=sum([math.sin(x)**2 for x in chr_real])
        secondSum=sum([x**2 for x in chr_real])
        res = 1 + firstSum - 0.1 * math.e**(-secondSum)
        #print("fitness value ", res)
        return res



class Problem02(RealProblem):
    def __init__(self, num_bit_code = 8, lower_bounds = [2.7], upper_bounds = [7.5]):
        super().__init__("Problem02", 1, num_bit_code, lower_bounds, upper_bounds)


    def computeFitness(self, chr_real):
        x=chr_real[0]
        res = math.sin(x) + math.sin(10/3*x)
        #print("fitness value ", res)
        return res


class Problem03(RealProblem):
    def __init__(self, num_bit_code = 8, lower_bounds = [-10], upper_bounds = [10]):
        super().__init__("Problem03", 1, num_bit_code, lower_bounds, upper_bounds)


    def computeFitness(self, chr_real):
        x=chr_real[0]
        res = -sum([k*math.sin((k+1)*x + k) for k in range(1,7)])
        #print("fitness value ", res)
        return res

class Problem04(RealProblem):
    def __init__(self, num_bit_code = 8, lower_bounds = [1.9], upper_bounds = [3.9]):
        super().__init__("Problem04", 1, num_bit_code, lower_bounds, upper_bounds)

    def computeFitness(self, chr_real):
        x=chr_real[0]
        res = -(16*x**2 - 24*x + 5)* math.e**(-x)
        #print("fitness value ", res)
        return res

class Problem05(RealProblem):
    def __init__(self, num_bit_code = 8, lower_bounds= [0.0], upper_bounds = [1.2]):
        super().__init__("Problem05", 1, num_bit_code, lower_bounds, upper_bounds)


    def computeFitness(self, chr_real):
        x=chr_real[0]
        res = -(1.4 -3*x)*math.sin(18*x)
        #print("fitness value ", res)
        return res

class Problem06(RealProblem):
    def __init__(self, num_bit_code = 8, lower_bounds = [-10], upper_bounds = [10]):
        super().__init__("Problem06", 1, num_bit_code, lower_bounds, upper_bounds)


    def computeFitness(self, chr_real):
        x=chr_real[0]
        res = -(x + math.sin(x))*math.e**(-x**2)
        #print("fitness value ", res)
        return res

class Problem07(RealProblem):
    def __init__(self, num_bit_code = 8, lower_bounds = [2.7], upper_bounds = [7.5]):
        super().__init__("Problem07", 1, num_bit_code, lower_bounds, upper_bounds)


    def computeFitness(self, chr_real):
        x=chr_real[0]
        res = math.sin(x) + math.sin(10/3*x) + math.log(x) - 0.84*x + 3
        #print("fitness value ", res)
        return res

class Problem08(RealProblem):
    def __init__(self, num_bit_code = 8, lower_bounds = [-10], upper_bounds = [10]):
        super().__init__("Problem08", 1, num_bit_code, lower_bounds, upper_bounds)

    def computeFitness(self, chr_real):
        x=chr_real[0]
        res = -sum([k*math.cos((k+1)*x + k) for k in range(1,7)])
        #print("fitness value ", res)
        return res

class Problem09(RealProblem):
    def __init__(self, num_bit_code = 8, lower_bounds = [3.1], upper_bounds = [20.4]):
        super().__init__("Problem09", 1, num_bit_code, lower_bounds, upper_bounds)


    def computeFitness(self, chr_real):
        x=chr_real[0]
        res = math.sin(x) + math.sin(2/3*x)
        #print("fitness value ", res)
        return res

class Problem10(RealProblem):
    def __init__(self, num_bit_code = 8, lower_bounds = [0], upper_bounds = [10]):
        super().__init__("Problem10", 1, num_bit_code, lower_bounds, upper_bounds)


    def computeFitness(self, chr_real):
        x=chr_real[0]
        res = - x * math.sin(x)
        #print("fitness value ", res)
        return res

class Problem13(RealProblem):
    def __init__(self, num_bit_code = 8, lower_bounds = [0.001], upper_bounds= [0.99]):
        super().__init__("Problem13", 1, num_bit_code, lower_bounds, upper_bounds)


    def computeFitness(self, chr_real):
        x=chr_real[0]
        res = -x**(2/3) - (1 - x**2)**(1/3)
        #print("fitness value ", res)
        return res

class Problem14(RealProblem):
    def __init__(self, num_bit_code = 8, lower_bounds = [0], upper_bounds = [4]):
        super().__init__("Problem14", 1, num_bit_code, lower_bounds, upper_bounds)

    def computeFitness(self, chr_real):
        x=chr_real[0]
        res = -math.e**(-x)*math.sin(2*math.pi*x)
        #print("fitness value ", res)
        return res

class Problem15(RealProblem):
    def __init__(self, num_bit_code = 8, lower_bounds = [-5], upper_bounds = [5]):
        super().__init__("Problem15", 1, num_bit_code, lower_bounds, upper_bounds)

    def computeFitness(self, chr_real):
        x=chr_real[0]
        res = (x**2 -5*x + 6)/(x**2 + 1)
        #print("fitness value ", res)
        return res


if __name__ == "__main__":
    p=SphereProblem(num_bit_code=5)
    sols, gray_sols,fitnesses=p.getWholeSpace()
    for i, j, l in zip(sols,gray_sols,fitnesses):
        print(i,j,l)

    p = OneMaxProblem()
    sols, fitnesses = p.getWholeSpace()
    for i, j in zip(sols,fitnesses):
        print(i,j)


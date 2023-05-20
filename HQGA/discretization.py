from sympy.combinatorics.graycode import gray_to_bin
from sympy.combinatorics.graycode import bin_to_gray


def convertFromBinToFloat(chr, lower_bounds, upper_bounds,num_bit_code,dim):
    """Function that converts the number in gray code to the corresponding float number

      Args:
          chr (string): a binary individual in gray code
          lower_bounds (list): list of real values, each one representing the lower bound for a gene of the individual
          upper_bounds (list): list of real values, each one representing the upper bound for a gene of the individual
          num_bit_code (int): the number of bits to use for coding one dimension
          dim (int): the number of dimensions

    Returns:
          chr_real (list): list of real values, each representing the allele for a gene of the individual
    """
    genes = []
    chr_real = []
    step = [(x-y)/(2 ** num_bit_code - 1) for x,y in zip(upper_bounds,lower_bounds)]
    k = 0
    #print(dim)
    #print(chr)
    for i in range(dim):
        var = chr[k:k + num_bit_code]
        #print("cromosome ", var)
        #print(gray_to_bin(var))
        #print(int(gray_to_bin(var), 2))
        genes.append(int(gray_to_bin(var), 2))
        gene_real = lower_bounds[i] + genes[i] * step[i]
        chr_real.append(gene_real)
        k = k + num_bit_code
    return chr_real


def convertFromBinToInterval(chr, lower_bounds_input, upper_bounds_input,num_bit_code,dim):
    """Function that converts the number in gray code to the corresponding interval of values

      Args:
          chr (string): a binary individual in gray code
          lower_bounds_input (list): list of real values, each one representing the lower bound for a gene of the individual
          upper_bounds_input (list): list of real values, each one representing the upper bound for a gene of the individual
          num_bit_code (int): the number of bits to use for coding one dimension
          dim (int): the number of dimensions

    Returns:
          lower_bounds (list): list of real values, each one representing the lower bound of the interval related to a dimension
          upper_bounds (list): list of real values, each one representing the upper bound of the interval related to a dimension
    """
    genes = []
    lower_bounds = []
    upper_bounds = []
    step = [(x-y)/(2 ** num_bit_code - 1) for x,y in zip(upper_bounds_input,lower_bounds_input)]
    k = 0
    #print(dim)
    #print(chr)
    for i in range(dim):
        var = chr[k:k + num_bit_code]
        #print("chromosome ", var)
        #print(gray_to_bin(var))
        #print(int(gray_to_bin(var), 2))
        genes.append(int(gray_to_bin(var), 2))
        gene_real = lower_bounds_input[i] + genes[i] * step[i]
        #print(gene_real)
        lower_bounds.append(gene_real)
        upper_bounds.append(gene_real+step[i])
        k = k + num_bit_code
    return lower_bounds,upper_bounds

def convertFromFloatToBin(value, lower_bound, upper_bound, num_bit_code):
    """Function that converts the real number to the corresponding number in gray code

      Args:
          value (float): a real number representing a gene of the individual
          lower_bound (float): real value representing the lower bound for the gene of the individual
          upper_bound (float): real value representing the upper bound for the gene of the individual
          num_bit_code (int): the number of bits to use for coding one dimension

    Returns:
          gray_value (string): string representing the real value in a binary number in gray code
    """
    step = (upper_bound - lower_bound) / (2 ** num_bit_code - 1)
    index= int(round((value - lower_bound)/step,1))
    binary="{0:b}".format(index)
    gray_value=bin_to_gray(binary)
    if len(gray_value) <num_bit_code:
        gray_value=(num_bit_code-len(gray_value))*'0'+gray_value
    return gray_value



#all values for a gene within [lower_bound,upper_bound] and encoded with num_bit_code bits
def getAllValues(lower_bound, upper_bound,num_bit_code):
    """Function that computes all real values in the range [lower_bound, upper_bound] coded with num_bit_code bits

      Args:
          lower_bound (real): real value representing the lower bound for a gene of the individual
          upper_bound (real): real value representing the upper bound for a gene of the individual
          num_bit_code (int): the number of bits to use for coding one dimension

    Returns:
          list_values (list): list of real values in the range [lower_bound, upper_bound] coded with num_bit_code bits
    """
    list_values = []
    step = (upper_bound - lower_bound) / (2 ** num_bit_code - 1)
    for i in range(2 ** num_bit_code):
        gene_real = lower_bound + i * step
        list_values.append(gene_real)
    return list_values

if __name__ == "__main__":
    gray='000010110000'
    float_v=convertFromBinToFloat(gray, [-1,-1,-1], [1,1,1],4,3)
    print(float_v)
    print(convertFromFloatToBin(0.7333333333333334, -1, 1,4)+'')
    print(getAllValues(-1,1,4))
    for k in getAllValues(-1,1,4):
        print(convertFromFloatToBin(k, -1, 1, 4) + '')

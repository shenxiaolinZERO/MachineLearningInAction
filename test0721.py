
def SumOfLqa():
    inputIntegers = input("Please input 6 positive integers[<100],the first integer is a： ")
    list = inputIntegers.split(" ")
    a = int(list[0])
    sum = 0
    for i in range(len(list)-1):
        b = int(list[i+1])
        if b < a:
            sum += b
    print("The sum is : ",sum)
    return sum




if __name__ == '__main__':
    # zhangsan = Person('zhangsan',24)
    # print(zhangsan)
    SumOfLqa()
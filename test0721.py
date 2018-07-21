
def SumOfLqa():
    inputIntegers = input("Please input 6 positive integers[<100],the first integer is a： ")
    list = inputIntegers.split(" ")
    a = list[0]
    sum = 0
    for i in range(len(list)):
        if list[i+1] < a:
            sum += list[i+1]
    print("The sum is : ",sum)
    return sum


if __name__ == '__main__':
    # zhangsan = Person('zhangsan',24)
    # print(zhangsan)
    SumOfLqa()
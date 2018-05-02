# !/usr/bin/env python
# encoding: utf-8
__author__ = 'Administrator'

import random
import copy
import time
import math

class Rlso:

    @staticmethod
    def go(user_sessions_data, session_item_data, item_session_data, aspects_num, ITERATION):
        session_user_dic, session_index_dic, item_list, item_index_dic=\
            data_prepare(user_sessions_data,session_item_data,item_session_data)

        #随机初始化U（各行需加和为1）
        user_num=len(user_sessions_data) #用户数
        U=dict()
        for i in user_sessions_data:
            U[i]=list()
            #随机数种子
            ls=random_init_KDvector(aspects_num,3)
            U[i]=normalize(ls)

        #随机初始化V
        item_num=len(item_list)
        V=dict()
        for item in item_list:
            V[item]=list()

        #s为随机数种子
        s=1
        for item_id in item_list:
            V[item_id]=random_init_KDvector(aspects_num,s)
            s +=1

        #归一化：对一个k，所有的v下,v_k加和为1
        V =normalize_items(V,aspects_num)

        V_update =dict()
        for item in item_list:
            V_update[item] = list()
            V_update[item] = copy.deepcopy(V[item])

        #随机初始化theta（需大于1）
        theta=1.1
        session_num=len(session_item_data)

        gamma=dict()
        likelihood = list()

        for t in range(ITERATION):
            start =time.time()

            #gamma 的计算
            for i in range(session_num):
                gamma_start=time.time()
                gamma[i]=list()
                #注意这里的i已经唯一确定对应的session（ID）了。
                #获取当前session（ID），以便获得对应的user信息（用于u_k中）
                d=session_item_data[i][0]
                user=session_user_dic[d]
                #需先求出分母“和”的值才能进行计算——进行加和的每个元素的值保存在temp_list中
                #temp_list保存（当前session）所有K个temp_w_product
                temp_list=list()
                scale_list=list()
                for k in range(aspects_num):
                    #当前session购买的商品
                    W_d=session_item_data[i][1]
                    #当前session点击但没有购买的商品
                    L_d=session_item_data[i][2]
                    #temp_w_product为每个gamma(d,k)计算式的分子项
                    temp_w_product=U[user][k]
                    scale=0
                    #遍历当前session的所有商品（ID）pairs
                    for w in W_d:
                        for v in L_d:
                            #获取item的feature值
                            w_k = V[w][k]
                            v_k = V[v][k]

                            temp_s_product =1
                            for s in range(aspects_num):
                                if s !=k:
                                    w_s=V[w][s]
                                    v_s=V[v][s]
                                    #原来直接计算的方式
                                    temp_s_product *=(theta*w_s)/(v_s+theta*w_s)

                            if temp_s_product==0:
                                print('session idx:',i)
                                print('stop here temp_s_product ==0')
                            if temp_s_product !=temp_s_product:
                                print('session idx:', i)
                                print('stop here temp_s_product !=temp_s_product')

                            if temp_w_product >1e10:
                                temp_w_product *=(w_k/(w_k+theta*v_k))*temp_s_product
                            else:
                                temp_scale =judge_scale(temp_s_product)
                                temp_w_product *=(w_k / (w_k +theta*v_k)) * temp_s_product *math.pow(10,temp_scale)
                                scale +=temp_scale
                    temp_list.append(temp_w_product)
                    scale_list.append(scale)

                max_scale =max(scale_list)
                temp_list_sum=0

                #特殊情况处理：判断gamma计算公式中分母加和各项是否存在某一项的值的规模比其中最小的值的规模大很多（处理temp_s_product为nan）
                exception_flag =0
                exception_k=0
                for k in range(aspects_num):
                    if max_scale -scale_list[k] >250:
                        exception_flag =1
                        exception_k =k
                        break
                    cur_temp =temp_list[k] * math.pow(10,max_scale-scale_list[k])
                    temp_list_sum +=cur_temp
                #若为异常情况
                if exception_flag ==1:
                    for k in range(aspects_num):
                        if k ==exception_k:
                            gamma[i].append(1.0)
                        else:
                            gamma[i].append(0.0)
                #一般情况下
                else:
                    for k in range(aspects_num):
                        cur_temp=temp_list[k] *math.pow(10,max_scale-scale_list[k])
                        gamma[i].append(cur_temp/temp_list_sum)

            end_all_gamma=time.time()
            all_gamma_spend=end_all_gamma-start
            print("迭代一轮全部session的gamma计算耗时：%fs"%all_gamma_spend)

            # u的计算
            for u in user_sessions_data:
                #需先求解出分母“和”的值才能进行计算——进行加和的每个元素的值保存在gamma_u_sum_list中
                #gamma_u_sum_list(当前user)所有k个gamma_u_sum
                temp_list=list()
                for k in range(aspects_num):
                    #取出当前用户下的所有session（ID）
                    cur_user_sessions_data = user_sessions_data[u]
                    #求出各个k值下的ud_gamma_sum
                    temp_ud_sum=0
                    for d in cur_user_sessions_data:
                        d_index = session_index_dic[d]
                        temp_ud_sum +=gamma[d_index][k]
                    temp_list.append(temp_ud_sum)

                temp_list_sum =sum(temp_list)
                for k in range(aspects_num):
                    U[u][k]=temp_list[k]/temp_list_sum
            end_all_u= time.time()
            all_u_spend =end_all_u -end_all_gamma
            print("迭代一轮全部session的u计算耗时为：%fs"%all_u_spend)


            #v的计算——注意：这里要等所有的v计算完再更新
            for v in item_list:
                for k in range(aspects_num):
                    # 这里的v_index用于item_session_data获取数据中
                    v_index =item_index_dic[v]
                    defeated_sum =0
                    # Wv表示购买了v 的session
                    Wv= item_session_data[v_index][1]
                    for d in Wv:
                        d_index=session_index_dic[d]
                        #L_d表示该session中点击但没有购买的所有商品
                        L_d=session_item_data[d_index][2]
                        defeated_sum +=len(L_d)
                    sum1 = 0
                    for d in Wv:
                        d_index = session_index_dic[d]
                        # L_d表示该session中点击但没有购买的所有商品
                        L_d = session_item_data[d_index][2]
                        for v_pie in L_d:
                            v_pie_index=item_index_dic[v_pie]
                            alpha1 = calculate_alpha(V[v][k],V[v_pie][k],theta)
                            alpha2 = calculate_alpha(V[v_pie][k],V[v][k],theta)
                            k_pie_temp = 0
                            for s in range(aspects_num):
                                if s !=k:
                                    k_pie_temp +=(theta *gamma[d_index][s]) / alpha2
                            sum1 +=gamma[d_index][k] / alpha1 +k_pie_temp
                    sum2=0
                    #Lv 表示点击但没有购买v的session
                    Lv=item_session_data[v_index][2]
                    for d in Lv:
                        d_index =session_index_dic[d]
                        #W_d表示该session中购买的所有商品
                        W_d = session_item_data[d_index][1]
                        for v_pie in W_d:
                            v_pie_index =item_index_dic[v_pie]
                            alpha1=calculate_alpha(V[v_pie][k],V[v][k],theta)
                            alpha2=calculate_alpha(V[v][k],V[v_pie][k],theta)
                            k_pie_temp=0
                            for s in range(aspects_num):
                                if s !=k:
                                    k_pie_temp +=gamma[d_index][s] /alpha2
                            sum2 +=(theta*gamma[d_index][k]) / alpha1 +k_pie_temp
                    V_update[v][k]=defeated_sum/(sum1+sum2)
            end_all_v=time.time()
            all_v_spend =end_all_v-end_all_u
            print("迭代一轮全部session的v计算耗时为：%fs"%all_v_spend)

            #theta 的计算——注意： \theta的计算用的是上一轮的v
            sum1=0
            for i in range(session_num):
                W_d =session_item_data[i][1]
                L_d =session_item_data[i][2]
                sum1 +=len(W_d) *len(L_d)
            sum2=0
            for i in range(session_num):
                for k in range(aspects_num):
                    w_v_temp = 0
                    W_d =session_item_data[i][1]
                    for w in W_d:
                        L_d=session_item_data[i][2]
                        for v in L_d:
                            alpha1 =calculate_alpha(V[w][k],V[v][k],theta)
                            k_pie_temp=0
                            for s in range(aspects_num):
                                if s !=k:
                                    alpha2 =calculate_alpha(V[v][s],V[w][s],theta)
                                    k_pie_temp +=V[w][s]/alpha2
                            w_v_temp +=V[v][k] / alpha1 +k_pie_temp
                    sum2+=gamma[i][k]*w_v_temp
            theta = (aspects_num-1)*sum1 /sum2
            print("theta为：",theta)
            end_all_theta =time.time()
            all_theta_spend= end_all_theta -end_all_v
            print("迭代一轮全部session的 theta 计算耗时：%fs" % all_theta_spend)

            # Normalize V——归一化：对一个k，所有的v下,v_k加和为1.
            for k in range(aspects_num):
                # 先计算出一个k下，所有v的加和
                v_k_sum = 0
                for item in item_list:
                    v_k_sum += V_update[item][k]
                for item in item_list:
                    V_update[item][k] = V_update[item][k] / v_k_sum

            # 更新V，为下一轮的计算做准备
            for item in item_list:
                V[item] = copy.deepcopy(V_update[item])

            # evaluation with likelihood value
            # session_aspect_dic = recommendation.calc_favorite_aspect(V, aspects_num, session_item_data)
            # val = evaluate2(U, V, theta, session_item_data, session_user_dic, session_aspect_dic, aspects_num)
            c = time.time() - start
            # 改正了似然的计算方式
            val = t
            # val = evaluate3(U, V, theta, session_item_data, session_user_dic, aspects_num)
            # if t%50 == 0:
            #     print("iteration", t, ",likelihood value:", val, ",耗时：%0.2f" % c, 's')
            # 哦哦，没错，原来是没每50个输出一次
            if t % 1 == 0:
                print("iteration", t, ",耗时：%0.2f" % c, 's')
            # # 提前退出迭代判断条件
            # if t > 1 and abs(val - likelihood[-1]) <= 0.001:
            #     likelihood.append(val)
            #     return U, V, theta, likelihood
            likelihood.append(val)

            # print("U: ", U)
            # print("V: ", V)
            # print("theta: ", theta)
        return U, V, theta, likelihood, gamma


def data_prepare(user_sessions_data,session_item_data,item_session_data):

    # 提取结果如下：
    # session_user_dic = {100:0,101:0,102:0}        #可以由 user_sessions_data求解得到
    # session_index_dic = {100:0,101:1,102:2}       #可以由 session_item_data求解得到
    # item_list = [10,11,12,13,14]                  #可以由 item_session_data求解得到
    # item_index_dic = {10:0,11:1,12:2,13:3,14:4}   #可以由 item_session_data求解得到

    session_user_dic = dict()
    session_index_dic = dict()
    item_list =list()
    item_index_dic=dict()
    for u in user_sessions_data:
        for session in user_sessions_data[u]:
            session_user_dic[session]=u

    for i in range(len(session_item_data)):
        d=session_item_data[i][0]
        session_index_dic[d]=i

    for i in range(len(item_session_data)):
        item= item_session_data[i][0]
        item_list.append(item)
        item_index_dic[item]=i

    return session_user_dic,session_index_dic,item_list,item_index_dic


#返回一个list，该list包含k个整数随机数，s 为随机数的种子
def random_init_KDvector(k,s):
    list=[]
    #设置随机种子
    random.seed(s)
    for i in range(0,k):
        tmp=random.randint(1,10000)
        list.append(tmp)
    return list

#归一化list，使加和为1
def normalize(list,sum=0):
    if sum==0:
        for i in range(0,len(list)):
            sum+=list[i]
    num =0
    for i in range(0,len(list)-1):
        list[i]=list[i]*1.0 /sum
        num +=list[i]
    list[len(list)-1] =1-num
    return list

#归一化整个V，使其在每个k上加和为1
def normalize_items(V,K):
    for k in range(K):
        #先计算出一个k下，所有v的加和
        v_k_sum =0
        for item in V.keys():
            v_k_sum +=V[item][k]
        for item in V.keys():
            V[item][k]=V[item][k] / v_k_sum
    return V

#初始化。产生一个加和为1， 长度为dim的list
#参数：长度dim
def initialize(dim):
    lst=list()
    for i in range(dim):
        lst.append(1.0/dim)
    return lst

#计算alpha
def calculate_alpha(v,v_pie,theta):
    return v+theta*v_pie

#判断一个浮点数有多小，如0.0010为3,0.01为2
def judge_scale(x):
    scale=0
    if x==0:
        print("x==0 in judge_scale")
        exit()
    if x!=x:
        print("x!=x in judge_scale")
        exit()
    while True:
        if x >=1.0:
            break
        else:
            x *=10
            scale +=1
        if x !=x:
            print("x !=x in judge_scale")
            exit()
    return scale


if __name__ == '__main__':

    # 开始计时
    startRlso = time.time()
    #读取数据
    train_file_dir = '..\\data\\dataset1\\train'
    user_sessions_data = rff.get_int_list_dict(train_file_dir + '\\user_session.txt')
    # print("user_sessions_data为：",user_sessions_data[5])
    # print("------------------")
    session_item_data = rff.get_data_lists(train_file_dir + '\\session_item.txt')

    # print("session_item_data为：", session_item_data[0][0])
    # print("------------------")
    item_session_data = rff.get_data_lists(train_file_dir + "\\item_session.txt")
    # print("item_session_data为：", item_session_data[0][0])
    # print("item_session_data为：", item_session_data[1][0])
    print('Starting training~~~~~~')
    print('完整数据中训练数据的session数是：', len(session_item_data))
    U, V, theta, likelihood,gamma=Rlso.go(user_sessions_data, session_item_data, item_session_data, 5, 100)

    endRlso = time.time()
    c2 = endRlso - startRlso
    print('Rlso 程序运行总耗时:%0.2f' % c2, 's')


    data_file_dir = '..\\data'  # 应该是唯一需要修改路径
    dataset = '1'

    # Zero：训练出的模型参数的存放路径（更新上面）
    write_file_dir = data_file_dir + '\\result\\TrainedParameters-0424'
    if not os.path.exists(write_file_dir):
        os.makedirs(write_file_dir)

    #
    print2file_list = [[theta], likelihood]
    # 输出结果到文件中
    file_name = ['theta.txt', 'likelihood.txt']
    idx = 0
    for cur_list in print2file_list:
        cur_file_path = write_file_dir + '\\' + file_name[idx]
        p2f.print_list_to_file(cur_list, cur_file_path)
        idx += 1

    gamma_file_path = write_file_dir + '\\' + 'gamma.txt'
    p2f.print_list_dict_to_file(gamma, gamma_file_path)
    print('训练出参数gamma')

    U_file_path = write_file_dir + '\\' + 'U.txt'
    p2f.print_list_dict_to_file(U, U_file_path)
    print('训练出参数U')
    V_file_path = write_file_dir + '\\' + 'V.txt'
    p2f.print_list_dict_to_file(V, V_file_path)
    print('训练出参数V')
    endTrained = time.time()
    c3 = endTrained - startRlso
    print('参数训练结束，总共耗时:%0.2f' % c3, 's')
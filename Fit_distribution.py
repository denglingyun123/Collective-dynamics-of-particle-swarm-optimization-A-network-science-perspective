import csv
import powerlaw
from fitter import Fitter
from collections import Counter
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import geom
from scipy.stats import logistic
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
# #################################拟合优度R^2的计算######################################
def __sst(y_no_fitting):
    """
    计算SST(total sum of squares) 总平方和
    :param y_no_predicted: List[int] or array[int] 待拟合的y
    :return: 总平方和SST
    """
    y_mean = sum(y_no_fitting) / len(y_no_fitting)
    s_list =[(y - y_mean)**2 for y in y_no_fitting]
    sst = sum(s_list)
    return sst


def __ssr(y_fitting, y_no_fitting):
    """
    计算SSR(regression sum of squares) 回归平方和
    :param y_fitting: List[int] or array[int]  拟合好的y值
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :return: 回归平方和SSR
    """
    y_mean = sum(y_no_fitting) / len(y_no_fitting)
    s_list =[(y - y_mean)**2 for y in y_fitting]
    ssr = sum(s_list)
    return ssr


def __sse(y_fitting, y_no_fitting):
    """
    计算SSE(error sum of squares) 残差平方和
    :param y_fitting: List[int] or array[int] 拟合好的y值
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :return: 残差平方和SSE
    """
    s_list = [(y_fitting[i] - y_no_fitting[i])**2 for i in range(len(y_fitting))]
    sse = sum(s_list)
    return sse


def goodness_of_fit(y_fitting, y_no_fitting):
    """
    计算拟合优度R^2
    :param y_fitting: List[int] or array[int] 拟合好的y值
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :return: 拟合优度R^2
    """
    SSR = __ssr(y_fitting, y_no_fitting)
    SST = __sst(y_no_fitting)
    rr = SSR /SST
    return rr

def get_column_by_name(filename, column_name):
    """
    根据列名获取CSV文件中的列数据。
    :param filename: CSV文件名
    :param column_name: 列名
    :return: 列中的所有数据
    """
    column_data = []
    with open(filename, 'r',encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if column_name in row:
                column_data.append(row[column_name])
        column_data = [int(item) if item.isdigit() else item for item in column_data]  # 将列表中的字符串类型转化为数字
    return column_data

# 泊松分布的似然函数
def poisson_likelihood(params):
    lmbda = params
    return -np.sum(poisson.logpmf(column_data, lmbda))

# 几何分布的似然函数
def geom_likelihood(params):
    p = params
    return -np.sum(geom.logpmf(column_data, p))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Step1: 读取csv文件的列
    filename = 'C:\\Users\\dengl\\Desktop\\Degree on Func1.csv'
    column_name = 'degree'
    column_data = get_column_by_name(filename, column_name)  # 通过列名访问列
    degree_counts = Counter(column_data)
    degree_list = []
    count_list = []
    for degree, count in degree_counts.items():  # 得到了度以及与度对应的频数
        degree_list.append(degree)
        count_list.append(count)

    Frequency = []
    for item in count_list:
        Frequency.append(item/sum(count_list))

    # Step2: 绘制原始数据的度分布散点图
    plt.scatter(degree_list, Frequency, facecolors='none', edgecolors='b', marker='o',label='Original data')

    # Step3: 拟合度分布
    # 使用最大似然估计拟合泊松分布(Poisson Distribution)
    result_poisson = minimize(poisson_likelihood, x0=[1], method='BFGS')
    lmbda_fit = result_poisson.x[0]
    # 使用最大似然估计拟合正态分布 (Normal Distribution)
    x_mean, x_std = norm.fit(column_data)
    # 使用最大似然估计拟合指数分布 (Exponential Distribution)
    loc1, scale1 = expon.fit(column_data, floc=0)
    # 使用最大似然估计拟合伽马分布 (Gamma Distribution)
    shape_fit, loc_fit, scale_fit = gamma.fit(column_data)
    # 使用最大似然估计拟合几何分布 (Geometric Distribution)
    result = minimize(geom_likelihood, x0=[1], method='BFGS')
    p_fit = result.x[0]
    # 使用最大似然估计拟合逻辑分布 (Logistic Distribution)
    loc2, scale2 = logistic.fit(column_data)
    # 使用极大似然估计拟合幂律分布 (Power-law Distribution)
    f = Fitter(column_data, distributions='powerlaw')
    f.fit()
    f.plot_pdf(names='powerlaw')

    # 得到拟合后的度分布序列
    degree_poisson_fit = poisson.pmf(degree_list, lmbda_fit)
    degree_norm_fit = norm.pdf(degree_list, x_mean, x_std)
    degree_expon_fit = expon.pdf(degree_list, loc1, scale=scale1)
    degree_gamma_fit = gamma.pdf(degree_list, shape_fit, scale=scale_fit)
    degree_geom_fit = geom.pmf(degree_list, p_fit)
    degree_logistic_fit = logistic.pdf(degree_list, loc2, scale2)



    # Step4: 计算残差平方和(SSE)
    SSE_poisson = __sse(degree_poisson_fit, Frequency)
    SSE_norm = __sse(degree_norm_fit, Frequency)
    SSE_expon = __sse(degree_expon_fit, Frequency)
    SSE_gamma = __sse(degree_gamma_fit, Frequency)
    SSE_geom = __sse(degree_geom_fit, Frequency)
    SSE_logistic = __sse(degree_logistic_fit, Frequency)
    # SSE_powerlaw = __sse(degree_powerlaw_fit, Frequency)

    print(f"泊松分布关于度分布序列的残差平方和(SSE_poisson): {SSE_poisson:.2f}")
    print(f"正态分布关于度分布序列的残差平方和(SSE_norm): {SSE_norm:.2f}")
    print(f"指数分布关于度分布序列的残差平方和(SSE_expon): {SSE_expon:.2f}")
    print(f"伽马分布关于度分布序列的残差平方和(SSE_gamma): {SSE_gamma:.2f}")
    print(f"几何分布关于度分布序列的残差平方和(SSE_geom): {SSE_geom:.2f}")
    print(f"逻辑分布关于度分布序列的残差平方和(SSE_logistic): {SSE_logistic:.2f}")
    # print(f"幂律分布关于度分布序列的残差平方和(SSE_powerlaw): {SSE_powerlaw:.2f}")
    print("____________________________________________")

    # Step5: 计算拟合度(R^2)
    r_squared_poisson = goodness_of_fit(degree_poisson_fit, Frequency)
    r_squared_norm = goodness_of_fit(degree_norm_fit, Frequency)
    r_squared_expon = goodness_of_fit(degree_expon_fit, Frequency)
    r_squared_gamma = goodness_of_fit(degree_gamma_fit, Frequency)
    r_squared_geom = goodness_of_fit(degree_geom_fit, Frequency)
    r_squared_logistic = goodness_of_fit(degree_logistic_fit, Frequency)
    # r_squared_powerlaw = goodness_of_fit(degree_powerlaw_fit, Frequency)

    print(f"泊松分布关于度分布序列的拟合度(r_squared_poisson): {r_squared_poisson:.2f}")
    print(f"正态分布关于度分布序列的拟合度(r_squared_norm): {r_squared_norm:.2f}")
    print(f"指数分布关于度分布序列的拟合度(r_squared_expon): {r_squared_expon:.2f}")
    print(f"伽马分布关于度分布序列的拟合度(r_squared_gamma): {r_squared_gamma:.2f}")
    print(f"几何分布关于度分布序列的拟合度(r_squared_geom): {r_squared_geom:.2f}")
    print(f"逻辑分布关于度分布序列的拟合度(r_squared_geom): {r_squared_logistic:.2f}")
    # print(f"幂律分布关于度分布序列的拟合度(r_squared_powerlaw): {r_squared_powerlaw:.2f}")

    # 绘制拟合曲线
    degree_list1 = sorted(degree_list)

    # plt.plot(degree_list1, poisson.pmf(degree_list1, lmbda_fit), 'r--', linewidth=1.5,label='Poisson')
    # plt.plot(degree_list1, norm.pdf(degree_list1, x_mean, x_std), 'k--', linewidth=1.5, label='Norm')
    # plt.plot(degree_list1, expon.pdf(degree_list1, loc=loc1, scale=scale1), 'y--', linewidth=1.5, label='Exponential')
    # plt.plot(degree_list1, gamma.pdf(degree_list1, shape_fit,  scale=scale_fit), 'g--', linewidth=1.5, label='Gamma')
    # plt.plot(degree_list1, geom.pmf(degree_list1, p_fit), 'm--', linewidth=1.5, label='Geometric')
    # plt.plot(degree_list1, logistic.pdf(degree_list1, loc2, scale2), 'c--', linewidth=1.5, label='Logistic')

    plt.xlabel('Degree')
    plt.ylabel('Probability density P(k)')
    plt.title("Degree Distribution and Fitted Curve")
    plt.grid(True)
    # # 获取当前图形的轴对象
    ax = plt.gca()
    # # 将轴设置为对数比例
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.legend()
    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

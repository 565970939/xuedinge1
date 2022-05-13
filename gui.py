import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
from tkinter import ttk
import tkinter
import matplotlib.pyplot as plt;
import numpy as np;


root = Tk()
root.title("一维定态薛定谔方程求解")
root.geometry("1000x350")
#分页
note = ttk.Notebook()
note.place(relx=0.02,rely=0.02,relwidth=0.9,relheight=0.9)
#第一页
frame1 = tkinter.Frame()
note.add(frame1,text='第一种势能')
#第二页
frame2 = tkinter.Frame()
note.add(frame2,text='第二种势能')
#第三页
frame3 = tkinter.Frame()
note.add(frame3,text='第三种势能')
#第四页
frame4 = tkinter.Frame()
note.add(frame4,text='第四种势能')
#……………………………………………………………………………………………………
'''将一维定态薛定谔方程的数值解封装成一个类'''
class Schrodinger():
    def __init__(self, potential_func,mass=1, hbar=1,xmin=-5, xmax=5, ninterval=1000):
        self.x = np.linspace(xmin, xmax, ninterval)
        self.U = np.diag(potential_func(self.x), 0)
        self.Lap = self.laplacian(ninterval)
        self.H = - hbar ** 2 / (2 * mass) * self.Lap + self.U
        self.eigE, self.eigV = self.eig_solve()

    '''构造二阶微分算子：Laplacian'''
    def laplacian(self, N):
        dx = self.x[1] - self.x[0]
        return (-2 * np.diag(np.ones((N), np.float32), 0)+ np.diag(np.ones((N - 1), np.float32), 1) + np.diag(np.ones((N - 1), np.float32), -1)) / (dx ** 2)

    '''解哈密顿矩阵的本征值，本征向量；并对本征向量排序'''
    def eig_solve(self):
        w, v = np.linalg.eig(self.H)
        idx_sorted = np.argsort(w)
        return w[idx_sorted], v[:, idx_sorted]

    '''波函数'''
    def wave_func(self, n=0):
        return self.eigV[:, n]

    '''本征值'''
    def eigen_value(self, n=0):
        return self.eigE[n]

    '''检查是否满足薛定谔方程'''
    def check_eigen(self, n=7):
        HPsi = np.dot(self.H, self.eigV[:, n])
        EPsi = self.eigE[n] * self.eigV[:, n]
        plt.subplot(132)
        plt.plot(self.x, HPsi, label=r'$H|\psi_{%s} \rangle$' % n)
        plt.plot(self.x, EPsi, '-.', label=r'$E |\psi_{%s} \rangle$' % n)
        plt.legend(loc='upper center')
        plt.xlabel(r'$x$')
        plt.ylim(EPsi.min(), EPsi.max() * 1.6)

    '''粒子在势能中的分布概率密度'''
    def plot_density(self, n=7):
        rho = self.eigV[:, n] * self.eigV[:, n]
        plt.subplot(133)
        plt.plot(self.x, rho)
        plt.title(r'$E_{%s}=%.2f$' % (n, self.eigE[n]))
        plt.ylabel(r'$\rho_{%s}(x)=\psi_{%s}^*(x)\psi_{%s}(x)$' % (n, n, n))
        plt.xlabel(r'$x$')

    '''势能可视化'''
    def plot_potential(self,n=7):
        plt.subplot(131)
        plt.plot(self.x, np.diag(self.U))
        plt.ylabel(r'potential')
        plt.xlabel(r'$x$')
#…………………………………………………………………………………………………………………………………………………………………………

#获值
def get_value(entry):
    value = entry.get()
    return int(value)


# 定义谐振子势
def harmonic_potential(x, k=100):
    return 0.5 * k * x ** 2

# 创建谐振子势下的薛定谔方程
schro_harmonic = Schrodinger(harmonic_potential)

'''势能可视化'''

def func_potential(i):
    schro_harmonic.plot_potential(i)
    plt.show()

'''检查是否满足薛定谔方程'''

def func_check(i):
    schro_harmonic.check_eigen(n=i)
    plt.show()

'''粒子在势能中的分布概率密度'''

def func_density(i):
    schro_harmonic.plot_density(n=i)
    plt.show()





# 添加标签控件
label11 = Label(frame1 , text="势能可视化输入", font=("宋体", 25), fg="red")
label11 .grid(row=0, column=0)
# 添加输入框
entry11 = Entry(frame1 , font=("宋体", 25), fg="red")
entry11.grid(row=0, column=1)
# 添加按钮

buttton11=Button(frame1 , text='点击', command=lambda :func_potential(get_value(entry11)))
buttton11.grid(row=0, column=2)

# 添加标签控件
label12 = Label(frame1 , text="检查是否满足薛定谔方程", font=("宋体", 25), fg="red")
label12.grid(row=2, column=0)
# 添加输入框
entry12 = Entry(frame1 , font=("宋体", 25), fg="red")
entry12.grid(row=2, column=1)
#

button12=Button(frame1 , text='点击', command=lambda :func_check(get_value(entry12)))
button12.grid(row=2, column=2)

# 添加标签控件
label13 = Label(frame1 , text="势能中分布概率密度", font=("宋体", 25), fg="red")
label13.grid(row=4, column=0)
# 添加输入框
entry13 = Entry(frame1 , font=("宋体", 25), fg="red")
entry13.grid(row=4, column=1)
# 添加按钮
buttton13=Button(frame1 , text='点击', command=lambda :func_density(get_value(entry13)))
buttton13.grid(row=4, column=2)

#……………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………
def woods_saxon_potential(x, R0=6.2, surface_thickness=0.5):
    sigma = surface_thickness
    return  -1 / (1 + np.exp((np.abs(x) - R0)/sigma))

# 创建谐振子势下的薛定谔方程
ws_schro = Schrodinger(woods_saxon_potential)
'''势能可视化'''

def func_potential1(i):
    ws_schro.plot_potential(i)
    plt.show()

'''检查是否满足薛定谔方程'''

def func_check1(i):
    ws_schro.check_eigen(n=i)
    plt.show()

'''粒子在势能中的分布概率密度'''
def func_density1(i):
    ws_schro.plot_density(n=i)
    plt.show()

# 添加标签控件
label21 = Label(frame2 , text="势能可视化", font=("宋体", 25), fg="red")
label21 .grid(row=0, column=0)
# 添加输入框
entry21 = Entry(frame2 , font=("宋体", 25), fg="red")
entry21.grid(row=0, column=1)
# 添加按钮
buttton21=Button(frame2 , text='pic', command=lambda :func_potential1(get_value(entry21)))
buttton21.grid(row=1, column=1)

# 添加标签控件
label22 = Label(frame2 , text="检查是否满足薛定谔方程", font=("宋体", 25), fg="red")
label22.grid(row=2, column=0)
# 添加输入框
entry22 = Entry(frame2 , font=("宋体", 25), fg="red")
entry22.grid(row=2, column=1)
#
button22=Button(frame2 , text='pic2', command=lambda :func_check1(get_value(entry22)))
button22.grid(row=3, column=1)

# 添加标签控件
label23 = Label(frame2 , text="粒子在势能中的分布概率密度", font=("宋体", 25), fg="red")
label23.grid(row=4, column=0)
# 添加输入框
entry23 = Entry(frame2 , font=("宋体", 25), fg="red")
entry23.grid(row=4, column=1)
# 添加按钮
buttton23=Button(frame2 , text='3', command=lambda :func_density1(get_value(entry23)))
buttton23.grid(row=5, column=1)

#……………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………
def double_well(x, xmax=5, N=100):
    w = xmax / N
    a = 3 * w
    return -100 * (np.heaviside(x + w - a, 0.5) - np.heaviside(x - w - a, 0.5)+np.heaviside(x + w + a, 0.5) - np.heaviside(x - w + a, 0.5))
dw = lambda x: double_well(x, xmax=5, N=1000)
dw_shro = Schrodinger(double_well)
'''势能可视化'''

def func_potential2(i):
    dw_shro.plot_potential(i)
    plt.show()

'''检查是否满足薛定谔方程'''

def func_check2(i):
    dw_shro.check_eigen(n=i)
    plt.show()

'''粒子在势能中的分布概率密度'''

def func_density2(i):
    dw_shro.plot_density(n=i)
    plt.show()


# 添加标签控件
label31 = Label(frame3 , text="势能可视化", font=("宋体", 25), fg="red")
label31 .grid(row=0, column=0)
# 添加输入框
entry31 = Entry(frame3 , font=("宋体", 25), fg="red")
entry31.grid(row=0, column=1)
# 添加按钮
buttton31=Button(frame3 , text='pic', command=lambda :func_potential2(get_value(entry31)))
buttton31.grid(row=1, column=1)

# 添加标签控件
label32 = Label(frame3, text="检查是否满足薛定谔方程", font=("宋体", 25), fg="red")
label32.grid(row=2, column=0)
# 添加输入框
entry32 = Entry(frame3 , font=("宋体", 25), fg="red")
entry32.grid(row=2, column=1)
#
button32=Button(frame3 , text='pic2', command=lambda :func_check2(get_value(entry32)))
button32.grid(row=3, column=1)

# 添加标签控件
label33 = Label(frame3 , text="粒子在势能中的分布概率密度", font=("宋体", 25), fg="red")
label33.grid(row=4, column=0)
# 添加输入框
entry33 = Entry(frame3 , font=("宋体", 25), fg="red")
entry33.grid(row=4, column=1)
# 添加按钮
buttton33=Button(frame3 , text='3', command=lambda :func_density2(get_value(entry33)))
buttton33.grid(row=5, column=1)


#……………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………

# 定义谐振子势
def harmonic_potential3(x, k=1):
    return 0.5 * k * x ** 2

# 创建谐振子势下的薛定谔方程
schro_harmonic3 = Schrodinger(harmonic_potential3)
'''势能可视化'''

def func_potential3(i):
    schro_harmonic3.plot_potential(i)
    plt.show()

'''检查是否满足薛定谔方程'''

def func_check3(i):
    schro_harmonic3.check_eigen(n=i)
    plt.show()

'''粒子在势能中的分布概率密度'''

def func_density3(i):
    schro_harmonic3.plot_density(n=i)
    plt.show()


# 添加标签控件
label41 = Label(frame4 , text="势能可视化", font=("宋体", 25), fg="red")
label41 .grid(row=0, column=0)
# 添加输入框
entry41 = Entry(frame4 , font=("宋体", 25), fg="red")
entry41.grid(row=0, column=1)
# 添加按钮
buttton41=Button(frame4 , text='pic', command=lambda :func_potential3(get_value(entry41)))
buttton41.grid(row=1, column=1)

# 添加标签控件
label42 = Label(frame4 , text="检查是否满足薛定谔方程", font=("宋体", 25), fg="red")
label42.grid(row=2, column=0)
# 添加输入框
entry42 = Entry(frame4 , font=("宋体", 25), fg="red")
entry42.grid(row=2, column=1)
#
button42=Button(frame4 , text='pic2', command=lambda :func_check3(get_value(entry42)))
button42.grid(row=3, column=1)

# 添加标签控件
label43 = Label(frame4 , text="粒子在势能中的分布概率密度", font=("宋体", 25), fg="red")
label43.grid(row=4, column=0)
# 添加输入框
entry43 = Entry(frame4 , font=("宋体", 25), fg="red")
entry43.grid(row=4, column=1)
# 添加按钮
buttton43=Button(frame4 , text='3', command=lambda :func_density3(get_value(entry43)))
buttton43.grid(row=5, column=1)

#…………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………………
root.mainloop()
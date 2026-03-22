import time
import numpy as np


class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self, mode="last", printout=True):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        if mode == "last":
            result = self.lastT()
        elif mode == "avg":
            result = self.avg()
        elif mode == "sum":
            result = self.sum()
        elif mode == "cumsum":
            result = self.cumsum()
        elif mode == "reset":
            self.reset()
            return True
        else:
            return "Can not find matching return type!"

        if printout:
            print(f'time consumption: {result:.2f} sec')
            return True

        return False

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

    def reset(self):
        """重设时间列表"""
        self.times = []
        return True

    def lastT(self):
        """返回上一次时间"""
        return self.times[-1]


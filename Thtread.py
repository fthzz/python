import threading
import queue

'''
1.线程和进程的区别：一个操作系统可以开启多个进程，一个进程可以开启多个线程
'''
'''
2.一个线程的状态：
New 创建
Runnable 就绪，等待调度
Running 运行
Blocked 堵塞
Dead 消亡
'''
'''
3.分类：
主线程
子线程
守护线程
前台线程
'''

'''
join
锁
线程通讯
'''


class Thread:
    def __init__(self):
        self.count = 0
        self.q=queue.Queue()

        self.lock = threading.Lock() #一个同步锁

        self.thread1 = threading.Thread(target=self.display1, args=(self.q,))
        self.thread1.start()
        self.thread1.join() #用于堵塞主线程，等子线程结束后主线程再结束

        #对于两个线程访问公共的数据时，会存在数据不统一或数据被同时修改的问题，此时使用锁解决
        self.thread2 = threading.Thread(target=self.display2,args=(self.q,))
        '''
        (self.q,)这里的括号表示的是一个元组，传递的是一个元组
        (self.q)而这种括号只是普通用法，传递的是参数q
        这里args需要interable，可迭代对象，所以队列类型不行，但元组可以，所以, 是一定要加的
        '''
        self.thread2.start()
        self.thread2.join()

    def display1(self):
        #同步锁
        self.lock.acquire() #同步锁的作用就是在锁进程块运行时，其他线程暂停运行
        for i in range(5):
            self.count += i
            print(self.count)
        self.lock.release() #一个acquire()需要一个release()。多次调用release()或者acquire()会导致死锁问题：程序堵塞，无法运行

    def display2(self):
        with self.lock: #这样写可以自动加锁和解锁
            for i in range(10,15):
                self.count += i
                print(self.count)

if __name__ == "__main__":
    print("start")
    thread = Thread()
    print("end")
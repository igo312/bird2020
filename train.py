
def taowa1(s='taowa beginning'):
    def taowa2(func):
        def taowa3(*args, **kwargs):
            print('%s func name is %s' %(s, func.__name__))
            print("taowa3's args is %s" %(args))
            return func(*args, **kwargs)
        return taowa3
    return taowa2

@taowa1()
def hello(s):
    print('we are in taowa program')
    return

taowa1()(hello)('heihei')
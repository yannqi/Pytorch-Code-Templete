import logging
from logging import handlers

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=7,fmt='%(asctime)s- %(levelname)s: %(message)s', split=True):
        """logging 

        Args:
            filename (str): the log file name.
            level (str, optional): save level. Defaults to 'info'.
            when (str, optional): 分时间何时分开保存. Defaults to 'D'.
            backCount (int, optional): backup Count, 分时间保存的个数,超过该数目,则覆盖保存(实现log的回滚). Defaults to 7. 这里设置成一周回滚。
            fmt (str, optional): the log format. Defaults to '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'.
        """
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        
        
        #切割日志文件
        if split:
            th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
            #实例化TimedRotatingFileHandler
            #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
            # S 秒
            # M 分
            # H 小时、
            # D 天、
            # W 每星期（interval==0时代表星期一）
            # midnight 每天凌晨
            th.suffix = "%Y-%m-%d.log"  #更改分割格式，末尾后缀为Log，便于gitignore管理
            th.setFormatter(format_str)#设置文件里写入的格式
            self.logger.addHandler(th)
        self.logger.addHandler(sh) #把对象加到logger里
        
if __name__ == '__main__':
    log = Logger('all.log',level='debug')
    log.logger.debug('debug')
    log.logger.info('info')
    log.logger.warning('警告')
    log.logger.error('报错')
    log.logger.critical('严重')
    Logger('error.log', level='error').logger.error('error')
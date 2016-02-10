# encoding: utf8
import sys, time, contextlib


RED = '\033[91m%s\033[0m'    # ANSI escape code Bright Red
YELLOW = '\033[91m%s\033[0m' # ANSI escape code Bright Yellow


class SimpleLog(object):
    def __init__(self, logfile=None, console=True):
        """
        SimpleLog lets you print messages to a file and to the 
        console at the same time. Errors will be written to
        console in red and warnings in yellow. The text "ERROR: "
        and "WARNING: " will be prepended to each line of the
        respective type of messages. 
        """
        self.files = []
        if logfile:
            self.files.append(open(logfile, 'wt'))
        if console:
            self.files.append(sys.stdout)
            
    def info(self, message):
        for f in self.files:
            f.write(message)
    
    def warning(self, message):
        message_warn = message[:-1].replace('\n', '\nWARNING: ')
        message = 'WARNING: %s%s' % (message_warn, message[-1])
        for f in self.files:
            if f.fileno() == 1:
                f.write(YELLOW % message)
            else:
                f.write(message)
    
    def error(self, message):
        message_err = message[:-1].replace('\n', '\nERROR: ')
        message = 'ERROR: %s%s' % (message_err, message[-1])
        for f in self.files:
            if f.fileno() == 1:
                f.write(RED % message)
            else:
                f.write(message)
    
    @contextlib.contextmanager
    def timer(self, pre_message='Timer: ', post_message='%4.2fs'):
        self.info(pre_message)
        t_start = time.time()
        yield
        duration = time.time() - t_start
        self.info(post_message % duration)

import time

from loguru import logger

start_time=time.time()
frame_count=0

class GETFPS:
    def __init__(self,stream_id):
        global start_time
        self.start_time=start_time
        self.is_first=True
        global frame_count
        self.frame_count=frame_count
        self.stream_id=stream_id
    def get_fps(self):
        end_time=time.time()
        if(self.is_first):
            self.start_time=end_time
            self.is_first=False
        if(end_time-self.start_time>5):
            logger.info("**********************FPS*****************************************")
            logger.info(f"Fps of stream {self.stream_id} is {float(self.frame_count)/5.0}")
            self.frame_count=0
            self.start_time=end_time
        else:
            self.frame_count=self.frame_count+1
    def print_data(self):
        print(f"frame_count= {self.frame_count}")
        logger.info(f"start_time: {self.start_time}")


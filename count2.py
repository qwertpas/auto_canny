#!/bin/env python

import cv2
from threading import Thread
import time

class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 480);
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 640);


        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True




#vs = cv2.VideoCapture(0)
vs = WebcamVideoStream(src=0).start()

a=0
count=0

while True:

    last = time.process_time_ns()
    frame = vs.read()

    cv2.imshow("frame", frame)
    cv2.imshow("fram2e", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elapsed = ((time.process_time_ns()-last))
    a = a + elapsed
    count+=1
    print(float(a)*1e-9/float(count))







cv2.destroyAllWindows()
vs.stop()



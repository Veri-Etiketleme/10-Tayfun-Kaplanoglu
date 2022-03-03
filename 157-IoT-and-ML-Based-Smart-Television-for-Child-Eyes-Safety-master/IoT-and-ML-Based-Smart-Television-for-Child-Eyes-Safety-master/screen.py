import os
import time

while 1:
 os.system("xset dpms force off")
 time.sleep(5)
 os.system("xset dpms force on")
 time.sleep(5)


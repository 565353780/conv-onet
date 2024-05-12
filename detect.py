from time import sleep
from conv_onet.Demo.detector import demo as demo_detect

if __name__ == "__main__":
    while True:
        demo_detect()
        sleep(10)

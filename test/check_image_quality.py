import time

import cv2
from picamera2 import Picamera2

cam = Picamera2(tuning=Picamera2.load_tuning_file("imx708_noir.json"))
cam.configure(cam.create_video_configuration(
    main={"size": (1536, 864), "format": "XBGR8888"},
    lores={"size": (640, 360), "format": "XBGR8888"},
    sensor={"output_size": (1536, 864)}))
cam.start()
time.sleep(1)                                          # let exposure settle

req = cam.capture_request()
full = req.make_array("main")[:, :, :3]
isp = req.make_array("lores")[:, :, :3]
md = req.get_metadata()
req.release()
cam.stop()

print("sensor mode:", cam.camera_configuration()["sensor"])
print("ScalerCrop:", md["ScalerCrop"])                 # part of the sensor actually used
cpu = cv2.resize(full, (640, 360), interpolation=cv2.INTER_AREA)
print("PSNR ISP vs CPU resize:", cv2.PSNR(isp, cpu))   # > ~35 dB means practically identical

for name, img in (("full.png", full), ("isp.png", isp), ("cpu.png", cpu)):
    cv2.imwrite(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

cam.close()

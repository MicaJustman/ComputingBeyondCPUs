import GPUtil
import time
import numpy as np


timeToRun = 10
sleepTime = 0

samplesPerSecond = 20

memValues = np.array([], dtype = float)
times = []

startTime = time.time()

while(time.time() - startTime < timeToRun):
    memValues = np.append(memValues, GPUtil.getGPUs()[0].memoryUsed)
    times.append(time.time() - startTime)
    time.sleep(1 / samplesPerSecond)

print("Done collecting results printing in MB used")


print(f"Memory Usage list: {list(memValues)}")

memValues = memValues[memValues != 0]

print(f"Memory Usage list excluding 0's: {list(memValues)}")

print(f"Max mem usage: {np.max(memValues)}")
print(f"Average mem usage: {np.mean(memValues):.2f}")
#print(f"Time Values: {times}")


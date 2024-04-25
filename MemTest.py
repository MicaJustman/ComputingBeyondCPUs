import GPUtil
import time
import matplotlib.pyplot as plt


timeToRun = 30
sleepTime = 0

samplesPerSecond = 20

memValues = []
times = []

startTime = time.time()

while(time.time() - startTime < timeToRun):
    memValues.append(GPUtil.getGPUs()[0].memoryUsed)
    times.append(time.time() - startTime)
    time.sleep(1 / samplesPerSecond)

print("Done collecting results printing in MB used")

print(f"Memory Usage: {memValues}")
print(f"Time Values: {times}")

plt.plot(times, memValues)
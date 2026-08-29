import sys
from retry_policy import retry_delay
print(retry_delay(int(sys.argv[1]), sys.argv[2]))

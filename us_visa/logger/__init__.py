import logging
import os
import sys
from datetime import datetime

# Format to how log file is created
log_file_name = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

log_dir = 'logs'

# Path for a log file
logs_file_path = os.path.join(os.getcwd(), log_dir, log_file_name)

# Even there is file keep on appending
os.makedirs(log_dir, exist_ok=True)  



# If handlers is used inside basicConfig then filename and filemode is not used else vice-versa
logging.basicConfig(
    # filename=logs_file_path,
    # filemode='w',
    level=logging.DEBUG,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(logs_file_path), # Output logs to the file 
        logging.StreamHandler(sys.stdout) # Output logs in the terminal/console
    ]
)

# Logging with Multiple Loggers
logger = logging.getLogger('us_visa')

# Log message with different severity levels
'''
logging.debug("This is dubug message")
logging.info("This is info message")
logging.warning("This is warning message")
logging.error("This is error message")
logging.critical("This is critical message")
'''
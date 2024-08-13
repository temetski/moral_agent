import csv

class Logger:
    def __init__(self, filename):
        self.filename = filename
        self.columns = ['step', 'question', 'response', 'reward', 'action']
        self.create_file()

    def create_file(self):
        """Create the CSV file with the headers."""
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.columns)
            writer.writeheader()

    def log(self, step, question, response, reward, action):
        """Log a new row of data into the CSV file."""
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.columns)
            writer.writerow({
                'step': step,
                'question': question,
                'response': response,
                'reward': reward,
                'action': action
            })

# # Example usage:
# if __name__ == "__main__":
#     logger = Logger('log.csv')

#     # Example data to log
#     logger.log(step=1, question="What is 2+2?", response="4", reward=1.0, action="Correct")
#     logger.log(step=2, question="What is the capital of France?", response="Paris", reward=1.0, action="Correct")
#     logger.log(step=3, question="What is 5*6?", response="30", reward=1.0, action="Correct")

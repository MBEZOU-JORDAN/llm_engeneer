import math
import matplotlib.pyplot as plt

# ANSI escape codes for terminal colors.
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
# Mapping from color names to ANSI codes for easy lookup.
COLOR_MAP = {"red":RED, "orange": YELLOW, "green": GREEN}

class Tester:
    """
    A class for testing a price prediction model.
    It runs the model on a dataset, calculates various error metrics,
    and generates a scatter plot to visualize the results.
    """

    def __init__(self, predictor, data, title=None, size=250):
        """
        Initializes the Tester.

        Args:
            predictor: The prediction function or method to be tested.
            data: The dataset to test the predictor on.
            title: The title for the test run and chart. If not provided, it's generated from the predictor's name.
            size: The number of data points to use from the dataset.
        """
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = size
        self.guesses = []
        self.truths = []
        self.errors = []
        self.sles = [] # Squared Logarithmic Errors
        self.colors = []

    def color_for(self, error, truth):
        """
        Determines the color for a data point on the plot based on the prediction error.
        Green for low error, orange for medium, and red for high error.

        Args:
            error: The absolute error of the prediction.
            truth: The ground truth price.

        Returns:
            A string representing the color ("green", "orange", or "red").
        """
        if error<40 or error/truth < 0.2:
            return "green"
        elif error<80 or error/truth < 0.4:
            return "orange"
        else:
            return "red"
    
    def run_datapoint(self, i):
        """
        Runs a single data point through the predictor, calculates metrics, and stores the results.

        Args:
            i: The index of the data point in the dataset.
        """
        datapoint = self.data[i]
        guess = self.predictor(datapoint)
        truth = datapoint.price
        error = abs(guess - truth)
        log_error = math.log(truth+1) - math.log(guess+1)
        sle = log_error ** 2
        color = self.color_for(error, truth)
        title = datapoint.title if len(datapoint.title) <= 40 else datapoint.title[:40]+"..."
        
        # Store the results for later analysis and plotting.
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)
        
        # Print the results for the current data point.
        print(f"{COLOR_MAP[color]}{i+1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} Error: ${error:,.2f} SLE: {sle:,.2f} Item: {title}{RESET}")

    def chart(self, title):
        """
        Generates and displays a scatter plot of the model's predictions vs. the ground truth.
        
        Args:
            title: The title for the chart.
        """
        max_error = max(self.errors)
        plt.figure(figsize=(12, 8))
        max_val = max(max(self.truths), max(self.guesses))
        # Plot a diagonal line representing a perfect prediction.
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6)
        # Scatter plot of the actual predictions.
        plt.scatter(self.truths, self.guesses, s=3, c=self.colors)
        plt.xlabel('Ground Truth')
        plt.ylabel('Model Estimate')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.title(title)
        plt.show()

    def report(self):
        """
        Calculates and reports the overall performance metrics of the model,
        and generates the final chart.
        """
        average_error = sum(self.errors) / self.size
        rmsle = math.sqrt(sum(self.sles) / self.size)
        hits = sum(1 for color in self.colors if color=="green")
        title = f"{self.title} Error=${average_error:,.2f} RMSLE={rmsle:,.2f} Hits={hits/self.size*100:.1f}%"
        self.chart(title)

    def run(self):
        """
        Runs the full test suite.
        It iterates through the specified number of data points, runs the predictor on each,
        and then generates a report.
        """
        self.error = 0
        for i in range(self.size):
            self.run_datapoint(i)
        self.report()

    @classmethod
    def test(cls, function, data):
        """
        A class method to conveniently create a Tester instance and run the test.
        """
        cls(function, data).run()

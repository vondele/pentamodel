from math import exp, log10
import random


class PentaModel:
    """
    PentaModel simulates chess match outcomes using a logistic function to estimate win, loss,
    and draw probabilities based on evaluation scores and Elo difference.

    Attributes:
        a (float): Normalized score, where 100 cp = 50% win chance in self play.
        b (float): Spread of the logistic function, roughly corresponding to LTC value.
        opponentElo (float): Elo difference of the opponent.
        s (float): Shift value to account for Elo difference.
        gamePairProbs (dict): Precomputed probabilities for all possible game pairs.
        pentanomialProbs (list): Precomputed probabilities for pentanomial outcomes.
    """

    def __init__(self, opponentElo):
        """
        Initialize the model with opponent Elo difference.
        Precomputes shift and probability tables.

        Args:
            opponentElo (float): Elo difference of the opponent.
        """
        self.a = 100  # Normalized score: 100 cp = 50% win chance in self play
        self.b = 22  # Spread of the logistic function, roughly LTC value
        self.opponentElo = opponentElo
        # Compute shift to account for Elo difference
        self.s = self.fix_s_from_Elo(opponentElo)
        # Precompute probabilities
        self.gamePairProbs = self.game_pair_probs()
        self.pentanomialProbs = self.pentanomial_probs()

    def win(self, v, s):
        """
        Probability of a win given evaluation v and shift s.

        Args:
            v (float): Evaluation score.
            s (float): Shift value.

        Returns:
            float: Probability of a win.
        """
        return 1 / (1 + exp((self.a + s - v) / self.b))

    def loss(self, v, s):
        """
        Probability of a loss given evaluation v and shift s.

        Args:
            v (float): Evaluation score.
            s (float): Shift value.

        Returns:
            float: Probability of a loss.
        """
        return 1 - (1 / (1 + exp((-self.a + s - v) / self.b)))

    def draw(self, v, s):
        """
        Probability of a draw given evaluation v and shift s.

        Args:
            v (float): Evaluation score.
            s (float): Shift value.

        Returns:
            float: Probability of a draw.
        """
        return 1 - self.win(v, s) - self.loss(v, s)

    def score(self, v, s):
        """
        Expected score (win + 0.5 * draw) for evaluation v and shift s.

        Args:
            v (float): Evaluation score.
            s (float): Shift value.

        Returns:
            float: Expected score.
        """
        return self.win(v, s) + self.draw(v, s) / 2

    def normalized_score_uho_pair(self, s):
        """
        Average expected score for both sides (UHO pair) at shift s.

        Args:
            s (float): Shift value.

        Returns:
            float: Average expected score for both sides.
        """
        return (self.score(self.a, s) + self.score(-self.a, s)) / 2

    @staticmethod
    def elo_diff_from_score(match_score):
        """
        Compute Elo difference from a match score.

        Args:
            match_score (float): Match score.

        Returns:
            float: Elo difference.
        """
        return 400 * log10(1 / match_score - 1)

    def fix_s_from_Elo(self, Elo):
        """
        Find the shift s that matches a given Elo difference using bisection.

        Args:
            Elo (float): Desired Elo difference.

        Returns:
            float: Shift value s.
        """
        assert Elo > -300 and Elo < 300, "Elo must be between -300 and 300"
        s_lower = -200
        s_upper = 200
        # Bisection search to find s value for desired Elo difference
        while s_upper - s_lower > 0.0000001:
            s_mid = (s_lower + s_upper) / 2
            elo_mid = self.elo_diff_from_score(self.normalized_score_uho_pair(s_mid))
            if elo_mid < Elo:
                s_lower = s_mid
            else:
                s_upper = s_mid
        return (s_upper + s_lower) / 2

    def game_pair_probs(self):
        """
        Returns a dictionary of probabilities for all possible game pairs.

        Keys: "WW", "WL", "WD", "DW", "DL", "DD", "LW", "LL", "LD"

        Returns:
            dict: Probabilities for each game pair outcome.
        """
        w_a = self.win(self.a, self.s)
        l_a = self.loss(self.a, self.s)
        d_a = self.draw(self.a, self.s)
        w_b = self.win(-self.a, self.s)
        l_b = self.loss(-self.a, self.s)
        d_b = self.draw(-self.a, self.s)
        return {
            "WW": w_a * w_b,
            "WL": w_a * l_b,
            "WD": w_a * d_b,
            "DW": d_a * w_b,
            "DL": d_a * l_b,
            "DD": d_a * d_b,
            "LW": l_a * w_b,
            "LL": l_a * l_b,
            "LD": l_a * d_b,
        }

    def pentanomial_probs(self):
        """
        Returns a list of probabilities for pentanomial outcomes.

        Order: [LL, LD+DL, DD+WL+LW, WD+DW, WW]

        Returns:
            list: Probabilities for each pentanomial outcome.
        """
        p = self.game_pair_probs()
        return [
            p["LL"],
            p["LD"] + p["DL"],
            p["DD"] + p["WL"] + p["LW"],
            p["WD"] + p["DW"],
            p["WW"],
        ]

    def return_game_pair(self):
        """
        Randomly samples a game pair outcome based on probabilities.

        Returns:
            str: Key representing the sampled game pair outcome.
        """
        uniformRandom = random.random()
        lastKey = None
        for [k, v] in self.gamePairProbs.items():
            lastKey = k
            if uniformRandom < v:
                return k
            uniformRandom -= v
        # Unlikely numeric event: random number larger than sum of probabilities
        return lastKey

    def return_match_pentanomial(self, rounds):
        """
        Simulates a match of 'rounds' game pairs and returns the pentanomial result counts.

        Args:
            rounds (int): Number of game pairs to simulate.

        Returns:
            list: Counts for each pentanomial outcome.
        """
        results = [0, 0, 0, 0, 0]
        for _ in range(0, rounds):
            uniformRandom = random.random()
            lastKey = 0
            for i, v in enumerate(self.pentanomialProbs):
                lastKey = i
                if uniformRandom < v:
                    break
                uniformRandom -= v
            results[lastKey] += 1
        return results


if __name__ == "__main__":
    # Example usage and demonstration
    import matplotlib.pyplot as plt
    import numpy as np
    from pprint import pprint

    # Generate a penta model with an opponent Elo of 100
    p = PentaModel(100)

    # Print UHO Game pair probabilities
    print("UHO Game pair probabilities:")
    pprint(p.gamePairProbs)
    print("Sum of probs values:", sum(p.gamePairProbs.values()))

    # Print UHO Pentanomial probabilities
    print("UHO Pentanomial probabilities:")
    pprint(p.pentanomialProbs)
    print("Sum of probs values:", sum(p.pentanomialProbs))

    # Generate 10 random game pairs
    print("Randomly generated game pairs:")
    for _ in range(10):
        print(p.return_game_pair())

    # Return a sample pentanomial of a match of N rounds
    print("Randomly generated pentanomial of a match:")
    rounds = 100000
    match_pentanomial = p.return_match_pentanomial(rounds)
    pprint(match_pentanomial)
    score = (
        match_pentanomial[0] * 0
        + match_pentanomial[1] * 0.5
        + match_pentanomial[2] * 1
        + match_pentanomial[3] * 1.5
        + match_pentanomial[4] * 2
    )
    print(f"Elo diff from score {p.elo_diff_from_score(score / (rounds * 2)):.2f} Elo")

    # Plot the relationship between book exit value v and win/loss/draw probabilities
    v_values = np.linspace(-400, 400, 200)
    win = [p.win(v, p.s) for v in v_values]
    plt.plot(v_values, win, label="Win probability")
    loss = [p.loss(v, p.s) for v in v_values]
    plt.plot(v_values, loss, label="Loss probability")
    draw = [p.draw(v, p.s) for v in v_values]
    plt.plot(v_values, draw, label="Draw probability")

    plt.ylabel("Probability")
    plt.xlabel("v")
    plt.title("Probabilities as a function of book exit")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    input("Press Enter to close the plot window...")
    plt.close()


class AlphaBetaGammaFilter():

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, gamma: float = 0.5) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def measure(self, pos_measure: np.ndarray):
        self.pos_measure = pos_measure

class KalmanFilter():

    def __init__(self) -> None:
        pass

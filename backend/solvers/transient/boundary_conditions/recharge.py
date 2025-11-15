# backend/solvers/transient/boundary_conditions/recharge.py

class RechargeBC:
    """
    Adds distributed recharge (L/T) per cell.
    """

    def __init__(self, recharge_array):
        self.recharge = recharge_array

    def apply(self, W):
        """
        Adds recharge directly to W vector (not A,b)
        """
        return W + self.recharge

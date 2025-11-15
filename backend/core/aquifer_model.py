@dataclass
class AquiferModel:
    name: str
    grid: Grid
    properties: AquiferProperties
    wells: list = field(default_factory=list)
    boundaries: list = field(default_factory=list)

    def transmissivity_tensor(self, i, j, k=0):
        """Return (Tx,Ty) tuple for a cell."""
        Tx = self.properties.transmissivity_x(k, i, j)
        Ty = self.properties.transmissivity_y(k, i, j)
        return Tx, Ty

    def storage(self, i, j, k=0):
        """Return storage coefficient depending on aquifer type."""
        if self.properties.confined:
            return self.properties.Ss[k] * self.properties.thickness[k]
        else:
            return self.properties.Sy[k]

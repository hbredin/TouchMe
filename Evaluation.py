
"""Permet d'evaluer les resultats par rapport aux references -> Calcul le rappel, la precision et la fmesure."""

class Evaluation():
	
	#### Calcul le nombre de frontieres bien detectees
	def reussis(self, hypotheses, references) : # [], []
		# Calcul du pourcentage de reussite sur la segmentation
		reussi = 0
		dist = np.array([[np.inf for u in range(len(hypotheses))] for v in range(len(references)) ])
		for i, val1 in enumerate(hypotheses):
			for j, val2 in enumerate(references):
				temp = abs(val1-val2)
				if temp < 8:
					dist[j][i] = temp
				
		while dist.min() != np.inf and reussi < len(references):
			reussi += 1
			tp = dist.argmin()
			x = tp/len(maxi)
			y = tp%len(maxi)
			for i in range(len(hypotheses)) : 
				dist[x][i] = np.inf
			for j in range(len(references)) : 
				dist[j][y] = np.inf		
		return reussi # int

	#### Calcul la precision	
	def precision(self, nbHyp, reussi) : # int, int 
		return float(reussi)/nbHyp # float


	#### Calcul le rappel
	def rappel(self, reussi, nbRef) : # int, int
		return float(reussi)/npRef # float
	

	#### Calcul la Fmesure
	def fmesure(self, prec, rapp) : # int, int
		temp = 0.0
		if prec+rapp != 0.0 :
			temp = ((2*prec*rapp)/(prec+rapp)) # float
		return temp

	#### Evalue les resultats par rapport aux verites terrain 
	def evalu(self, references, hypotheses) : # [], [] 
		reussi = self.reussis(maxi, veritesTerrain)
		prec = self.precision(len(hypotheses), reussi)
		rapp = self.rappel(reussi, len(veritesTerrain))
		fmesure = self.fmesure(prec, rapp)
		return prec, rapp, fmesure # float, float, float



